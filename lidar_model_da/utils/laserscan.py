import numpy as np
from utils.dataset_utils import range_image_to_points, denormalization_ranges, normalization_queries

import torch

class LaserScan:
  def __init__(self, lidar_in):
    self.lidar_in = lidar_in
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # if all goes well, open pointcloud
    f = open(filename, 'r')
    input_range_image = np.fromfile(f, dtype='float64')
    f.close()

    input_range_image = input_range_image.reshape(1,1,16,1024)
    input_range_image = input_range_image.astype('float32')

    input_range = denormalization_ranges(input_range_image.flatten())
    input_range[input_range < 0.] = 0.
    input_range[input_range > self.lidar_in['norm_r']] = self.lidar_in['norm_r']

    input_pcd = range_image_to_points(input_range, self.lidar_in)
    r = np.sqrt(np.sum(input_pcd**2, axis=1))
    roi_idx = np.logical_and.reduce((input_pcd[:,2] >= -4.4, input_pcd[:,2] <= 2.0, r < self.lidar_in['max_r']))
    input_pcd = input_pcd[roi_idx, :]
    input_pcd = input_pcd[:,[1,0,2]]

    # put in attribute
    self.set_points(input_pcd)

  def set_points(self, points):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # put in attribute
    self.points = points    # get xyz
    self.unproj_range = np.sqrt(np.sum(points**2, axis=1))

class HR_LaserScan(LaserScan):
  def __init__(self, lidar_out, model):
    self.lidar_out = lidar_out
    self.model = model
    self.set_queries()
    self.reset()

  def set_queries(self):
    # Query
    v_dir = np.linspace(start=self.lidar_out['min_v'], stop=self.lidar_out['max_v'], num=self.lidar_out['channels'])
    h_dir = np.linspace(start=self.lidar_out['min_h'], stop=self.lidar_out['max_h'], num=self.lidar_out['points_per_ring'], endpoint=False)

    v_angles = []
    h_angles = []

    for i in range(self.lidar_out['channels']):
      v_angles = np.append(v_angles, np.ones(self.lidar_out['points_per_ring']) * v_dir[i])
      h_angles = np.append(h_angles, h_dir)

    queries = np.stack((v_angles, h_angles), axis=-1).astype(np.float32)
    queries = normalization_queries(queries, self.lidar_out)

    # Upsampling
    self.input_queries = torch.from_numpy(queries).cuda().unsqueeze(0)

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # if all goes well, open pointcloud
    f = open(filename, 'r')
    input_range_image = np.fromfile(f, dtype='float64')
    f.close()

    input_range_image = input_range_image.reshape(1,1,16,1024)
    input_range_image = input_range_image.astype('float32')
    input_range_image = torch.from_numpy(input_range_image).cuda()

    with torch.no_grad():
      pred_ranges = self.model(input_range_image, self.input_queries)

    pred_ranges = denormalization_ranges(pred_ranges)
    pred_ranges = pred_ranges.view(-1)
    pred_ranges[pred_ranges < 0.] = 0.
    pred_ranges[pred_ranges > self.lidar_out['norm_r']] = self.lidar_out['norm_r']

    pred_pcd = range_image_to_points(pred_ranges.detach().cpu().numpy(), self.lidar_out)
    r = np.sqrt(np.sum(pred_pcd**2, axis=1))
    roi_idx = np.logical_and.reduce((pred_pcd[:,2] >= -4.4, pred_pcd[:,2] <= 2.0, r < self.lidar_out['max_r']))
    pred_pcd = pred_pcd[roi_idx, :]
    pred_pcd = pred_pcd[:,[1,0,2]]

    # put in attribute
    self.set_points(pred_pcd)
