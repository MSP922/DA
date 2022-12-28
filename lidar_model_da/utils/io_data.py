import numpy as np
import yaml
import imageio


def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed

def pack(array):
  """ convert a boolean array into a bitwise array. """
  array = array.reshape((-1))

  #compressing bit flags.
  # yapf: disable
  compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
  # yapf: enable

  return np.array(compressed, dtype=np.uint8)

def _read_SemKITTI(path, dtype, do_unpack):
  bin = np.fromfile(path, dtype=dtype)  # Flattened array
  if do_unpack:
    bin = unpack(bin)
  return bin

def _read_rgb_SemKITTI(path):
  rgb = np.asarray(imageio.imread(path))
  return rgb


def _read_pointcloud_SemKITTI(path):
  'Return pointcloud semantic kitti with remissions (x, y, z, intensity)'
  pointcloud = _read_SemKITTI(path, dtype=np.float32, do_unpack=False)
  pointcloud = pointcloud.reshape((-1, 4))
  return pointcloud