import numpy as np

class RepresentationBase():
    def __init__(self):
        self.POINT_CLOUD_PARAMS = None
        self.VOXEL_PARAMS = None
        self.RAY_PARAMS = None
    
    def voxel_params(self):
        RANGE = {'x': [0, 51.2], 'y': [-25.6, 25.6],'z':[-2.0, 4.4]}
        DIMS = {'H': 32, 'W': 256, 'D': 256}
        GRID_SIZE = 0.2

        FIDX = np.arange(DIMS['H']*DIMS['W']*DIMS['D']).reshape(DIMS['W'],DIMS['H'],DIMS['D'])
        FIDX = np.moveaxis(FIDX, [0,1,2], [0,2,1]).reshape(-1)

        IDX = self.func_voxel_fidx_to_xyzidx(FIDX, voxel_dims=DIMS, return_dict=True)

        CENTER_XYZ = self.func_voxel_idx_to_center(IDX, RANGE, GRID_SIZE)

        return {'RANGE': RANGE, 'DIMS': DIMS, 'FDIM': DIMS['W']*DIMS['H']*DIMS['D'],
                'GRID_SIZE': GRID_SIZE, 'FIDX': FIDX, 'IDX': IDX, 'CENTER': CENTER_XYZ}
    
    def func_voxel_xyz_to_xyzidx(self, x, y, z):
        xidx = np.floor((x - self.VOXEL_PARAMS['RANGE']['x'][0])/self.VOXEL_PARAMS['GRID_SIZE'])
        yidx = np.floor((y - self.VOXEL_PARAMS['RANGE']['y'][0])/self.VOXEL_PARAMS['GRID_SIZE'])
        zidx = np.floor((z - self.VOXEL_PARAMS['RANGE']['z'][0])/self.VOXEL_PARAMS['GRID_SIZE'])
        return xidx, yidx, zidx

    def func_voxel_fidx_to_xyzidx(self, fidx, voxel_dims=None, return_dict=False):
        if voxel_dims is None and self.VOXEL_PARAMS is not None:
            tmp, yidx = np.divmod(fidx, self.VOXEL_PARAMS['DIMS']['W'])
            xidx, zidx = np.divmod(tmp, self.VOXEL_PARAMS['DIMS']['H'])
        else:
            tmp, yidx = np.divmod(fidx, voxel_dims['W'])
            xidx, zidx = np.divmod(tmp, voxel_dims['H'])
        
        if return_dict:
            return {'x': xidx, 'y': yidx, 'z': zidx}
        else:
            return xidx, yidx, zidx

    def func_voxel_xyzidx_to_fidx(self, xidx, yidx, zidx, voxel_dims=None):
        if voxel_dims is None and self.VOXEL_PARAMS is not None:
            return np.int64(yidx + self.VOXEL_PARAMS['DIMS']['W']*(zidx + self.VOXEL_PARAMS['DIMS']['H'] * xidx))
        else:
            return np.int64(yidx + voxel_dims['W']*(zidx + voxel_dims['H'] * xidx))
    
    def func_voxel_idx_to_fidx(self, IDX, voxel_dims=None):
        if voxel_dims is None and self.VOXEL_PARAMS is not None:
            return np.int64(IDX['y'] + self.VOXEL_PARAMS['DIMS']['W']*(IDX['z'] + self.VOXEL_PARAMS['DIMS']['H'] * IDX['x']))
        else:
            return np.int64(IDX['y'] + voxel_dims['W']*(IDX['z'] + voxel_dims['H'] * IDX['x']))
    
    def func_voxel_idx_to_center(self, IDX, RANGE=None, GRID_SIZE=None):
        if (RANGE is None or GRID_SIZE is None) and self.VOXEL_PARAMS is not None:
            return np.concatenate((
                ((IDX['x']+0.5)*self.VOXEL_PARAMS['GRID_SIZE'] + self.VOXEL_PARAMS['RANGE']['x'][0]).reshape(-1,1),
                ((IDX['y']+0.5)*self.VOXEL_PARAMS['GRID_SIZE'] + self.VOXEL_PARAMS['RANGE']['y'][0]).reshape(-1,1),
                ((IDX['z']+0.5)*self.VOXEL_PARAMS['GRID_SIZE'] + self.VOXEL_PARAMS['RANGE']['z'][0]).reshape(-1,1)
            ), axis=1)
        else:
            return np.concatenate((
                ((IDX['x']+0.5)*GRID_SIZE + RANGE['x'][0]).reshape(-1,1),
                ((IDX['y']+0.5)*GRID_SIZE + RANGE['y'][0]).reshape(-1,1),
                ((IDX['z']+0.5)*GRID_SIZE + RANGE['z'][0]).reshape(-1,1)
            ), axis=1)

    def ray_params(self):
        RANGE = {'h': [-0.4*np.pi, 0.4*np.pi], 'v': None}
        DIMS = {'h': 1024, 'v':64}
        GRID_SIZE = (RANGE['h'][1] - RANGE['h'][0]) / DIMS['h']

        FIDX = np.arange(DIMS['h'] * DIMS['v'])

        inv_layer, horz_idx = np.divmod(FIDX, 1024)
        IDX = {'h': horz_idx, 'v': inv_layer}

        horz_ang = self.func_horz_idx_to_angle(horz_idx, RANGE, GRID_SIZE)
        vert_ang = self.init_vertical_angle(RANGE, GRID_SIZE)
        ANG = {'h': horz_ang, 'v': vert_ang}
        RAY = self.func_angle_to_ray(ANG)
        return {'RANGE': RANGE, 'DIMS': DIMS, 'FDIM': DIMS['h']*DIMS['v'], 'GRID_SIZE': GRID_SIZE,
                'FIDX': FIDX, 'IDX': IDX, 'ANGLE': ANG, 'RAY': RAY}

    def func_horz_idx_to_angle(self, horz_idx, RANGE=None, GRID_SIZE=None):
        if (RANGE is None or GRID_SIZE is None) and self.RAY_PARAMS is not None:
            return (horz_idx + 0.5) * self.RAY_PARAMS['GRID_SIZE'] + self.RAY_PARAMS['RANGE']['h'][0]
        else:
            return (horz_idx + 0.5) * GRID_SIZE + RANGE['h'][0]
        
    def func_horz_angle_to_idx(self, horz_angle, RANGE=None, GRID_SIZE=None):
        if (RANGE is None or GRID_SIZE is None) and self.RAY_PARAMS is not None:
            return  np.floor((horz_angle - self.RAY_PARAMS['RANGE']['h'][0])/self.RAY_PARAMS['GRID_SIZE'])
        else:
            return np.floor((horz_angle - RANGE['h'][0]) * GRID_SIZE)
    
    def func_angle_to_ray(self, ANG=None):
        if ANG is None and self.RAY_PARAMS is not None:
            return {'x': np.cos(self.RAY_PARAMS['ANGLE']['v']) * np.cos(self.RAY_PARAMS['ANGLE']['h']),
                    'y': np.cos(self.RAY_PARAMS['ANGLE']['v']) * np.sin(self.RAY_PARAMS['ANGLE']['h']),
                    'z': np.sin(self.RAY_PARAMS['ANGLE']['v'])}
        else:
            return {'x': np.cos(ANG['v']) * np.cos(ANG['h']),
                    'y': np.cos(ANG['v']) * np.sin(ANG['h']),
                    'z': np.sin(ANG['v'])}

    def init_vertical_angle(self, RANGE=None, GRID_SIZE=None):
        def func(x, a, b, c, d, e):
            return a*x**4 + b*x**3 + c*x**2+d*x + e
        popt_array = np.array(
            [[-7.699407e-03, 9.252107e-04,  2.307237e-02, -3.074322e-03,  4.125655e-02],
                [-7.946800e-03, 8.529778e-04,  2.358416e-02, -3.133186e-03,  3.495042e-02],
                [-8.420451e-03, 9.226977e-04,  2.440059e-02, -3.394200e-03,  2.995137e-02],
                [-8.790182e-03, 9.315611e-04,  2.504271e-02, -3.566233e-03,  2.204013e-02],
                [-9.262116e-03, 1.166157e-03,  2.578271e-02, -3.843294e-03,  1.672356e-02],
                [-9.263092e-03, 1.225525e-03,  2.616682e-02, -4.132732e-03,  9.709379e-03],
                [-9.244737e-03, 1.281019e-03,  2.568769e-02, -4.199192e-03,  4.931792e-03],
                [-9.370751e-03, 1.081307e-03,  2.594981e-02, -4.239208e-03, -1.432609e-03],
                [-9.481669e-03, 1.069290e-03,  2.592257e-02, -4.268822e-03, -7.421036e-03],
                [-9.212305e-03, 9.954280e-04,  2.539883e-02, -4.309931e-03, -1.428471e-02],
                [-9.206443e-03, 1.010167e-03,  2.517631e-02, -4.370451e-03, -1.929357e-02],
                [-9.273705e-03, 1.012454e-03,  2.528609e-02, -4.522750e-03, -2.504815e-02],
                [-9.068580e-03, 9.679239e-04,  2.457894e-02, -4.467060e-03, -3.130059e-02],
                [-8.921298e-03, 9.436505e-04,  2.424551e-02, -4.534543e-03, -3.695334e-02],
                [-8.739506e-03, 9.060497e-04,  2.391640e-02, -4.592855e-03, -4.239854e-02],
                [-8.474327e-03, 9.287716e-04,  2.318251e-02, -4.645026e-03, -4.831897e-02],
                [-8.259792e-03, 9.249098e-04,  2.253804e-02, -4.667540e-03, -5.327968e-02],
                [-8.278463e-03, 9.211392e-04,  2.240353e-02, -4.796566e-03, -5.971605e-02],
                [-8.146777e-03, 8.313116e-04,  2.175148e-02, -4.638508e-03, -6.425172e-02],
                [-8.202594e-03, 7.687341e-04,  2.193770e-02, -4.749211e-03, -7.058106e-02],
                [-8.248319e-03, 8.052791e-04,  2.179652e-02, -4.754034e-03, -7.567019e-02],
                [-7.719239e-03, 7.683338e-04,  2.123424e-02, -4.822150e-03, -8.143925e-02],
                [-7.368162e-03, 8.368159e-04,  2.033696e-02, -4.840814e-03, -8.691909e-02],
                [-7.555957e-03, 9.180778e-04,  2.053779e-02, -5.006575e-03, -9.250949e-02],
                [-6.777558e-03, 7.448700e-04,  1.930459e-02, -4.847916e-03, -9.882512e-02],
                [-6.734494e-03, 7.064188e-04,  1.907792e-02, -4.899982e-03, -1.043074e-01],
                [-6.805779e-03, 7.164313e-04,  1.890671e-02, -4.875206e-03, -1.095461e-01],
                [-6.073417e-03, 5.077428e-04,  1.773129e-02, -4.806887e-03, -1.142289e-01],
                [-5.747110e-03, 5.073423e-04,  1.688527e-02, -4.767594e-03, -1.206605e-01],
                [-5.751950e-03, 5.215343e-04,  1.714661e-02, -4.972449e-03, -1.269053e-01],
                [-5.981651e-03, 6.891430e-04,  1.714530e-02, -5.052171e-03, -1.309567e-01],
                [-5.667830e-03, 6.517169e-04,  1.652429e-02, -5.099648e-03, -1.374296e-01],
                [-4.164740e-03, 7.859862e-04,  1.169109e-02, -4.616598e-03, -1.479787e-01],
                [-3.816674e-03, 8.582787e-04,  1.093905e-02, -4.759865e-03, -1.570357e-01],
                [-3.695985e-03, 9.006785e-04,  1.070650e-02, -4.914672e-03, -1.651057e-01],
                [-3.264994e-03, 8.919408e-04,  9.703559e-03, -4.950232e-03, -1.718984e-01],
                [-2.868495e-03, 9.164527e-04,  8.815250e-03, -5.035819e-03, -1.799657e-01],
                [-2.826276e-03, 9.437194e-04,  8.976801e-03, -5.362245e-03, -1.904300e-01],
                [-2.641867e-03, 9.574289e-04,  8.476751e-03, -5.456071e-03, -1.990435e-01],
                [-2.196806e-03, 9.229600e-04,  7.581256e-03, -5.520741e-03, -2.064347e-01],
                [-1.995765e-03, 8.624547e-04,  7.558982e-03, -5.733528e-03, -2.143475e-01],
                [-1.750959e-03, 9.280025e-04,  6.907447e-03, -5.872050e-03, -2.217647e-01],
                [-1.155004e-03, 9.318191e-04,  5.843505e-03, -6.080116e-03, -2.308775e-01],
                [-1.456845e-03, 1.138026e-03,  6.434135e-03, -6.557240e-03, -2.399802e-01],
                [-9.802726e-04, 1.280609e-03,  5.191365e-03, -6.711677e-03, -2.498190e-01],
                [-1.123413e-03, 1.490753e-03,  5.351615e-03, -7.133772e-03, -2.574837e-01],
                [-5.051780e-04, 1.078740e-03,  4.995586e-03, -7.179966e-03, -2.660633e-01],
                [ 5.163216e-05, 1.099188e-03,  3.898162e-03, -7.347149e-03, -2.722231e-01],
                [ 6.007874e-04, 1.288532e-03,  2.768039e-03, -7.691218e-03, -2.825249e-01],
                [ 1.090837e-03, 1.303179e-03,  2.282467e-03, -8.056940e-03, -2.914453e-01],
                [ 1.449866e-03, 1.511490e-03,  1.441713e-03, -8.503560e-03, -3.009271e-01],
                [ 1.686088e-03, 1.892556e-03,  4.560275e-04, -8.906527e-03, -3.085986e-01],
                [ 1.731096e-03, 1.880379e-03,  6.071963e-04, -9.246729e-03, -3.172016e-01],
                [ 1.701391e-03, 2.302043e-03, -2.228498e-04, -9.503581e-03, -3.241113e-01],
                [ 1.852740e-03, 2.198880e-03, -1.478245e-04, -9.868275e-03, -3.318728e-01],
                [ 1.312292e-03, 2.642586e-03,  8.163519e-04, -1.047656e-02, -3.413507e-01],
                [ 1.177259e-03, 2.773768e-03,  1.057302e-03, -1.080804e-02, -3.501355e-01],
                [ 1.670953e-04, 3.346134e-03,  2.493561e-03, -1.146546e-02, -3.608139e-01],
                [ 1.627271e-04, 3.278658e-03,  2.716004e-03, -1.171456e-02, -3.692765e-01],
                [-2.531066e-04, 4.030194e-03,  2.680322e-03, -1.233492e-02, -3.761275e-01],
                [-3.541372e-04, 4.563040e-03,  2.655750e-03, -1.305477e-02, -3.838599e-01],
                [-1.288709e-03, 5.195932e-03,  4.332159e-03, -1.398649e-02, -3.954789e-01],
                [-1.166195e-03, 5.089843e-03,  4.249220e-03, -1.404592e-02, -4.047208e-01],
                [-2.841186e-03, 6.136462e-03,  6.666151e-03, -1.517288e-02, -4.140586e-01]])

        if (RANGE is None or GRID_SIZE is None) and self.RAY_PARAMS is not None:
            horizontal_angle = self.func_horz_idx_to_angle(np.arange(1024))
        else:
            horizontal_angle = self.func_horz_idx_to_angle(np.arange(1024), RANGE, GRID_SIZE)
        
        vert_ang_list = []
        for i in range(64):
            popt_tmp = popt_array[i,:]
            vert_ang_list.append(func(horizontal_angle, *popt_tmp))

        return np.concatenate(vert_ang_list)
    
    def func_valid_index(self, xidx, yidx, zidx):
        return np.logical_and.reduce((
            xidx>=0,
            yidx>=0,
            zidx>=0,
            xidx<self.VOXEL_PARAMS['DIMS']['D'],
            yidx<self.VOXEL_PARAMS['DIMS']['W'],
            zidx<self.VOXEL_PARAMS['DIMS']['H']))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    B = RepresentationBase()
    RAY_PARAMS = B.ray_params()
    plt.scatter(np.arange(65536),np.float32(RAY_PARAMS['ANGLE']['v']))
    plt.show()
