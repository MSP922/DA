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

    def KITTI_nth_intersection_idx(self, n, axis='x'):
        c = n * self.VOXEL_PARAMS['GRID_SIZE'] + self.VOXEL_PARAMS['RANGE'][axis][0]
        c = np.full(self.RAY_PARAMS['DIMS']['h']*self.RAY_PARAMS['DIMS']['v'], c)
        r = c / self.RAY_PARAMS['RAY'][axis]

        x = r * self.RAY_PARAMS['RAY']['x']
        y = r * self.RAY_PARAMS['RAY']['y']
        z = r * self.RAY_PARAMS['RAY']['z']
        return r, x, y, z

    def func_intersection_idx(self, xidx, yidx, zidx, axis='x', oidx=0):
        if axis == 'y':
            yidx[yidx < oidx] -= 1
        elif axis == 'z':
            zidx[zidx < oidx] -= 1
        return xidx, yidx, zidx
    
    def KITTI_axis_intersection(self, occupancy, axis='x'):
        if axis == 'x':
            S = 'D'
        elif axis == 'y':
            S = 'W'
        elif axis == 'z':
            S = 'H'
        
        o_idx = np.floor(-self.VOXEL_PARAMS['RANGE'][axis][0]/self.VOXEL_PARAMS['GRID_SIZE'])
        
        r_in = np.full(self.RAY_PARAMS['FDIM'], 60)
        xyz_in = np.zeros((self.RAY_PARAMS['FDIM'],3))
        flag = np.full(self.RAY_PARAMS['FDIM'], False)

        for i in range(1,self.VOXEL_PARAMS['DIMS'][S]):
            r, x, y, z = self.KITTI_nth_intersection_idx(i, axis=axis)
            xidx, yidx, zidx = self.func_voxel_xyz_to_xyzidx(x, y, z)
            xidx, yidx, zidx = self.func_intersection_idx(xidx, yidx, zidx, axis=axis, oidx=o_idx)
            fidx = self.func_voxel_xyzidx_to_fidx(xidx, yidx, zidx)
            val_idx = self.func_valid_index(xidx, yidx, zidx)

            val_fidx = fidx * val_idx
            occupancy_now = occupancy[val_fidx] * val_idx

            d_idx = (~flag) * occupancy_now
            r_in[d_idx] = r[d_idx]
            xyz_in[d_idx, 0] = x[d_idx]
            xyz_in[d_idx, 1] = y[d_idx]
            xyz_in[d_idx, 2] = z[d_idx]
            flag[d_idx] = True

        return r_in, xyz_in, flag
    
    def KITTI_intersection2(self, voxel):
        occupancy = np.full(self.VOXEL_PARAMS['FDIM'], False)
        occupancy[self.VOXEL_PARAMS['FIDX'][voxel>0]] = True
        r_axisX, xyz_axisX, flag_axisX = self.KITTI_axis_intersection(occupancy, axis='x')
        r_axisY, xyz_axisY, flag_axisY = self.KITTI_axis_intersection(occupancy, axis='y')
        r_axisZ, xyz_axisZ, flag_axisZ = self.KITTI_axis_intersection(occupancy, axis='z')

        r_all = np.concatenate((r_axisX.reshape(-1,1),r_axisY.reshape(-1,1),r_axisZ.reshape(-1,1)), axis=1)
        xyz_all = np.concatenate((xyz_axisX, xyz_axisY, xyz_axisZ), axis=0)
        flag_all = np.concatenate((flag_axisX, flag_axisY, flag_axisZ))

        min_idx = np.argmin(r_all, axis=1)
        print(min_idx.shape)
        min_idx = min_idx * self.RAY_PARAMS['FDIM'] + np.arange(self.RAY_PARAMS['FDIM'])

        xyz = xyz_all[min_idx, :]
        flag = flag_all[min_idx]

        return xyz[flag,:]

    
    def KITTI_intersection(self, voxel):
        d_in = np.zeros((self.RAY_PARAMS['FDIM'],3))
        in_checked_flag = np.full(self.RAY_PARAMS['FDIM'], False)

        occupancy = np.full(self.VOXEL_PARAMS['FDIM'], False)
        occupancy[self.VOXEL_PARAMS['FIDX'][voxel>0]] = True

        for i in range(256):
            r, x, y, z = self.KITTI_nth_intersection_idx(i)
            xidx, yidx, zidx = self.func_voxel_xyz_to_xyzidx(x, y, z)
            fidx = self.func_voxel_xyzidx_to_fidx(xidx, yidx, zidx)
            val_idx = self.func_valid_index(xidx, yidx, zidx)

            val_fidx = fidx * val_idx
            occupancy_now = occupancy[val_fidx] * val_idx

            d_idx = (~in_checked_flag) * occupancy_now
            d_in[d_idx, 0] = x[d_idx]
            d_in[d_idx, 1] = y[d_idx]
            d_in[d_idx, 2] = z[d_idx]
            in_checked_flag[d_idx] = True

        return d_in[in_checked_flag, :] #* self.RAY_PARAMS['RAY']
    
class VoxelRayInteraction_old(RepresentationBase):
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

    PC = B.KITTI_intersection2(PRED_VOXEL)
    ax = plt.axes(projection = '3d')
    ax.scatter3D(PC[:,0], PC[:,1], PC[:,2], color = 'k', s=1)
    plt.show()
