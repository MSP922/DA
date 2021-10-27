import numpy as np

def func_range(x):
    return np.sqrt(np.sum(x**2, axis=1))

def func_angle(xyz):
    xy2 = func_range(xyz[:,0:2])
    return np.arctan2(xyz[:,2], xy2), np.arctan2(xyz[:,1], xyz[:,0])