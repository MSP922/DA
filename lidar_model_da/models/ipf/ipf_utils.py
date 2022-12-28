import torch
import torch.nn as nn
from utils.dataset_utils import denormalization_queries, normalization_ranges, denormalization_ranges

def coords_to_rays(coords, lidar):
    angles = denormalization_queries(coords.clone(), lidar)

    x = torch.sin(angles[:, :, 1]) * torch.cos(angles[:, :, 0])
    y = torch.cos(angles[:, :, 1]) * torch.cos(angles[:, :, 0])
    z = torch.sin(angles[:, :, 0])

    xyz = torch.stack((x, y, z), dim=-1)
    return xyz

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)