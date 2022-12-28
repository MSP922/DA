# Data Augmentation (DA)

## LiDAR Super-Resolution (LSR)

### Introduction

* This repository is released for LiDAR Super-Resolution (LSR).
* This works on velodyne files (*.bin) from the KITTI dataset.

### Requirements
Python 3.8.13\
Pytorch 1.12.0\
Numpy 1.21.5\
(For visualization) Vispy, Matplotlib
### Demo
```
cd ./lidar_model_da
python main.py
```

## Voxel-Ray-Interaction (VRI) module

### Introduction
This repository is released for Voxel-Ray-Interaction (VRI) module.

### Requirements
Python 3.7.5\
Numpy 1.20.3\
(For visualization) Matplotlib

### Demo
```
python ./VRI/voxel_ray_interaction.py
```

## Recent Updates
* (2021.12.12) Intersection-based code update
* (2022.12.29) LiDAR Super-Resolution code update

## Background

### Outline
This SW is related to 'data augmentation engine' in 'Development of Driving Environment Data Transformation and Data Verification Technology for the Mutual Utilization of Self-driving Learning Data for Different Vehicles', and in detail, it is SW for LiDAR data augmentation.
### Data augmentation engine
New data with changed weather, lighting, objects, etc. is generated(augmented) from the acquired real environment information or virtual information.
- Environmental data augmentation: Generating other types of data in which the environment, object elements, etc. is conducted from the collected data and the environmental information at the time of collection
- Main object data augmentation: Inserting main target objects (VRU, Vehicle) on the road into the collected data in using AI
![data_augmentation_image](https://user-images.githubusercontent.com/95835936/147022053-62dd1851-2717-41af-9233-3c5f344dc8cb.png)
