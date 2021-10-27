import numpy as np
import io_data as SemanticKittiIO

from base_setting import RepresentationBase
import VRI_utils

class VoxelRayInteraction(RepresentationBase):
    def __init__(self):
        super().__init__()
        self.VOXEL_PARAMS = self.voxel_params()
        self.RAY_PARAMS = self.ray_params()
    
    def func_valid_idx(self, xyz):
        x_vidx = np.logical_and(
            xyz[:,0] >= self.VOXEL_PARAMS['RANGE']['x'][0],
            xyz[:,0] < self.VOXEL_PARAMS['RANGE']['x'][1]
        )
        y_vidx = np.logical_and(
            xyz[:,1] >= self.VOXEL_PARAMS['RANGE']['y'][0],
            xyz[:,1] < self.VOXEL_PARAMS['RANGE']['y'][1]
        )
        z_vidx = np.logical_and(
            xyz[:,2] >= self.VOXEL_PARAMS['RANGE']['z'][0],
            xyz[:,2] < self.VOXEL_PARAMS['RANGE']['z'][1]
        )

        horz_ang = np.arctan2(xyz[:,1], xyz[:,0])
        horz_vidx = np.logical_and(
            horz_ang >= self.RAY_PARAMS['RANGE']['h'][0],
            horz_ang < self.RAY_PARAMS['RANGE']['h'][1]
        )
        return np.logical_and.reduce((x_vidx, y_vidx, z_vidx, horz_vidx))
        
    def KITTI_voxel_to_LiDAR(self, voxel):
        occupied_idx = voxel > 0
        occupied_voxel = voxel[occupied_idx]
        occupied_fidx = self.VOXEL_PARAMS['FIDX'][occupied_idx]
        
        voxel_idx_dict = self.func_voxel_fidx_to_xyzidx(occupied_fidx, return_dict=True)
        voxel_center_xyz = self.func_voxel_idx_to_center(voxel_idx_dict)

        val_idx = self.func_valid_idx(voxel_center_xyz)
        voxel_center_xyz = voxel_center_xyz[val_idx,:]

        vox_r = VRI_utils.func_range(voxel_center_xyz)
        vox_vert_ang, vox_horz_ang = VRI_utils.func_angle(voxel_center_xyz)
        vox_horz_idx = self.func_horz_angle_to_idx(vox_horz_ang)

        PC_LIST = []
        for j in range(self.RAY_PARAMS['DIMS']['h']):
            tmp_voxel_vert_ang = vox_vert_ang[vox_horz_idx == j]
            tmp_voxel_r = vox_r[vox_horz_idx == j]

            r_sort_idx = np.argsort(tmp_voxel_r)

            tmp_voxel_vert_ang = tmp_voxel_vert_ang[r_sort_idx]
            tmp_voxel_r = tmp_voxel_r[r_sort_idx]

            tmp_vert_ang = self.RAY_PARAMS['ANGLE']['v'][self.RAY_PARAMS['IDX']['h'] == j]
            if len(tmp_voxel_vert_ang) > 0:
                vert_ang_expand = np.tile(tmp_vert_ang.reshape(-1,1), (1,len(tmp_voxel_vert_ang))) # 64 x vox_VN
                voxel_vert_ang_expand = np.tile(tmp_voxel_vert_ang.reshape(1,-1), (len(tmp_vert_ang), 1))

                ang_res = np.abs(vert_ang_expand - voxel_vert_ang_expand)

                idx_base = np.arange(len(tmp_voxel_vert_ang))
                association_idx = np.zeros(len(tmp_vert_ang))
                for jj in range(ang_res.shape[0]):
                    cand_idx = ang_res[jj,:] < 0.005
                    if np.sum(cand_idx) > 0:
                        association_idx[jj] = idx_base[cand_idx][0]
                    else:
                        association_idx[jj] = -1

                val_idx = association_idx != -1

                tmp_vert_ang = tmp_vert_ang[val_idx]
                min_idx = np.uint(association_idx[val_idx])

                r_result = tmp_voxel_r[min_idx]
                x = r_result * np.cos(tmp_vert_ang) * np.cos(self.RAY_PARAMS['ANGLE']['h'][j])
                y = r_result * np.cos(tmp_vert_ang) * np.sin(self.RAY_PARAMS['ANGLE']['h'][j])
                z = r_result * np.sin(tmp_vert_ang)

                xyz = np.concatenate((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)), axis=1)
                PC_LIST.append(xyz)

        return np.concatenate(PC_LIST, axis=0)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    B = VoxelRayInteraction()

    PRED_VOXEL = SemanticKittiIO._read_label_SemKITTI('./pred_voxel_example/example.label')

    PC = B.KITTI_voxel_to_LiDAR(PRED_VOXEL)
    ax = plt.axes(projection = '3d')
    ax.scatter3D(PC[:,0], PC[:,1], PC[:,2], color = 'k', s=1)
    plt.show()