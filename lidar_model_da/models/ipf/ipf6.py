from turtle import pos
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ipf.edsr_lidar import EDSRLiDAR
from models.ipf.tf_weight import WeightTransformer
from einops import rearrange

from models.model_utils import register_model

from models.ipf.ipf_utils import coords_to_rays, MLP
from models.ipf.positional_encoding import get_embedder

def make_coord(shape, ranges=None, flatten=True):
    # Make coordinates at grid centers.
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


@register_model('ipf6')
class IPF(nn.Module):
    def __init__(self, L=None, lidar=None):
        super().__init__()
        # Encoder: EDSR
        self.encoder = EDSRLiDAR(n_resblocks=16, n_feats=64, res_scale=1.0)

        assert lidar != None, 'lidar specification required'
        self.lidar = lidar

        self.embed_fn, input_ch = get_embedder(L)
        self.embeddirs_fn, input_ch_view = get_embedder(L)

        dim = self.encoder.out_dim

        self.offset_prediction = MLP(in_dim=input_ch_view+2*input_ch+dim, out_dim=1+dim, hidden_list=[256, 256, 256])

        # self.offset_prediction = MLP(in_dim=input_ch_view+2*input_ch+dim, out_dim=1, hidden_list=[256, 256, 256])

        self.linear = nn.Linear(input_ch, dim)
        self.attention = WeightTransformer(num_classes=1, dim=dim,
                                           depth=2, heads=8, mlp_dim=dim,
                                           dim_head=(dim//8), dropout=0.1)
        
        # self.weight_mlp = MiniPointNet(dims = [input_ch+64, 256, 256, 256, 1]) # ipf3
        # self.mlp = SharedMLP(dims = [input_ch+64, 64, 64, 64, 1])
        # self.softmax = nn.Softmax(dim=2)

    def gen_feat(self, inp):
        self.inp_img = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_detection(self, coord):
        feat = self.feat
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:])

        # bs: batch size
        # q: query size
        # d: feature dimension
        # t: neighbors
        preds = torch.empty((4, coord.shape[0], coord.shape[1]), device='cuda')
        q_feats = torch.empty((4, feat.shape[0], coord.shape[1], feat.shape[1]), device='cuda') # [t, bs, q, d]
        
        bs, q = coord.shape[:2]
        ray = coords_to_rays(coord, self.lidar) # [bs, q, 3]
        dir_embedding = self.embeddirs_fn(ray)
        t = 0
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift  # vertical
                coord_[:, :, 1] += vy * ry + eps_shift  # horizontal
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [bs, q, d]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [bs, q, 2]

                q_ray = coords_to_rays(q_coord, self.lidar) # [bs, q, 3]
                r = F.grid_sample(self.inp_img, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [bs, q, 1]
                # print('t(r):', r[0,0,:].item())
                
                # [-1.0 ~ 1.0 -> 0.0 ~ 1.0]
                r = (r + 1.0) * 0.5

                q_point = r * q_ray # [bs, q, 3]
                proj_d = torch.matmul(q_point.view(bs*q, 1, 3), ray.view(bs*q, 3, 1)).view(bs, q, 1) # [bs, q, 1]
                proj = proj_d * ray # [bs, q, 3]
                rej = q_point - proj # [bs, q, 3]

                rej_embedding = self.embed_fn(-rej) # [bs, q, input_ch]
                proj_embedding = self.embed_fn(proj)
                inp = torch.cat((dir_embedding, rej_embedding, proj_embedding, q_feat), dim=-1) # [bs, q, input_ch+d]

                offset = self.offset_prediction(inp) # [bs, q, 1+d]
                range_offset = offset[:,:,:1]
                feat_offset = offset[:,:,1:]

                pred = proj_d + range_offset # [bs, q, 1]
                
                # duppy projection
                pred[r + 1e-6 > 1.0] = 1.0

                pred_point = pred * ray # [bs, q, 3]

                # [0.0 ~ 1.0] -> [-1.0 ~ 1.0]
                pred = pred * 2.0 - 1.0

                q_feat = q_feat + feat_offset

                q_feats[t, :, :, :] = q_feat + self.linear(self.embed_fn(pred_point))        # [4, bs, q, d]
                preds[t, :, :] = pred.view(bs, q)    # [4, bs, q]
                t = t + 1

        q_feats = rearrange(q_feats, "t bs q d -> (bs q) t d")

        weights = self.attention(q_feats) # [bs*q, t, 1]
        preds = rearrange(preds, "t bs q -> (bs q) t").unsqueeze(1) # [bs*q, 1, t]
        
        ret = torch.matmul(preds, weights)  # [bs*Q, 1]
        ret = ret.view(bs, q, -1)            # [bs, Q, 1]
        # _, mi = torch.max(weights.squeeze(-1), dim=1)
        # preds = rearrange(preds, "t bs q -> (bs q) t")
        # preds = torch.gather(preds, 1, mi.unsqueeze(1))
        # ret = preds.view(bs, q, -1)
        return ret

    def forward(self, inp, coord):
        self.gen_feat(inp)
        return self.query_detection(coord)
