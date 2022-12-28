import argparse
import os
from tkinter import Frame

import torch
import torch.nn as nn
from tqdm import tqdm

# Datasets
from utils.dataset_utils import initialize_lidar
from utils.laserscan import LaserScan, HR_LaserScan
from utils.laserscanvis import LaserScanVis

# Models
from models.ipf.ipf6 import IPF
from models.model_utils import generate_model

import numpy as np

if __name__ == '__main__':
  # Parse the arguments
  parser = argparse.ArgumentParser(description="Evaluate the MAE (Mean Absolute Error) on the Carla dataset")
  parser.add_argument('-d', '--data_directory',
                      type=str,
                      required=True,
                      help='Data dataset directory')
  parser.add_argument('-f', '--frame',
                      type=int,
                      required=True,
                      help='frame number (0~4070)')
  parser.add_argument('-b', '--batch',
                      type=int,
                      required=False, 
                      default=1,
                      choices=[1, 2, 7, 11, 14, 17, 22, 34],
                      help='Batch size for network testing. (default: 1)')
  parser.add_argument('-cp', '--checkpoint',
                      type=str,
                      required=False,
                      default='./models/trained/ipf6_400.pth',
                      help='Check point filename. [.pth]')
  args = parser.parse_args()

  lidar_config_filename = os.path.join(args.data_directory, 'lidar_specification.yaml')
  lidar_in = initialize_lidar(lidar_config_filename, channels=16, points_per_ring=1024)
  lidar_out = initialize_lidar(lidar_config_filename, channels=64, points_per_ring=2048)

  scan_paths = os.path.join(args.data_directory, 'rimg')
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(scan_paths)) for f in fn]
  scan_names.sort()

  # Load the check point
  check_point = torch.load(args.checkpoint)

  # Model
  print(check_point['model']['name'])
  print(check_point['model']['args'])
  # print(aa)
  model = generate_model(check_point['model']['name'], check_point['model']['args'])
  model.load_state_dict(check_point['model']['state_dict'])

  model.lidar = lidar_in
  out_directory = 'ipf'

  num_of_gpus = torch.cuda.device_count()
  if torch.cuda.device_count() > 1:
    model = nn.parallel.DataParallel(model)
  model.eval().cuda()

  scan = LaserScan(lidar_in=lidar_in)
  # hr_scan = LaserScan(lidar_in=lidar_in)
  hr_scan = HR_LaserScan(lidar_out=lidar_out, model=model)

  vis = LaserScanVis(scan=scan,
                      hr_scan=hr_scan,
                      scan_names=scan_names,
                      offset=args.frame)

  # print instructions
  print("To navigate:")
  print("\tb: back (previous scan)")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")

  # run the visualizer
  vis.run()


        
