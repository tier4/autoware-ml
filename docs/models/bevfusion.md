---
icon: lucide/scan-line
---

# BEVFusion

BEVFusion is a camera-LiDAR 3D object detection model integrated under the `detection3d` task namespace. It combines a sparse-voxel LiDAR branch (hard voxelization + sparse 3D convolution encoder), a multiview image branch, a LiDAR-depth-guided `DepthLSSTransform` view transform, and a convolutional BEV fusion layer before the TransFusion detection head. LiDAR-only configurations are also provided, they skip the image branch entirely.

## Summary

| Property     | Value                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------- |
| Task         | 3D object detection                                                                         |
| Modality     | Camera, LiDAR                                                                               |
| Input        | Point cloud, optionally with synchronized multiview images                                  |
| Output       | 3D bounding boxes and class scores                                                          |
| Architecture | Sparse voxel encoder + multiview camera backbone/FPN + DepthLSS + fusion + TransFusion head |
| Datasets     | NuScenes, T4Dataset                                                                         |

## Available Configurations

| Config Name                                                                        | Dataset   | Purpose                                    |
| ---------------------------------------------------------------------------------- | --------- | ------------------------------------------ |
| `detection3d/bevfusion/lidar_voxel0075_second_secfpn_54m_nuscenes`                 | NuScenes  | LiDAR-only NuScenes 54 m configuration     |
| `detection3d/bevfusion/camera_lidar_voxel0075_second_secfpn_54m_nuscenes`          | NuScenes  | Camera-LiDAR NuScenes 54 m configuration   |
| `detection3d/bevfusion/lidar_voxel0170_second_secfpn_120m_t4dataset_j6gen2`        | T4Dataset | LiDAR-only T4Dataset 120 m configuration   |
| `detection3d/bevfusion/camera_lidar_voxel0170_second_secfpn_120m_t4dataset_j6gen2` | T4Dataset | Camera-LiDAR T4Dataset 120 m configuration |

## Training

```bash
autoware-ml train --config-name detection3d/bevfusion/camera_lidar_voxel0075_second_secfpn_54m_nuscenes
autoware-ml train --config-name detection3d/bevfusion/camera_lidar_voxel0170_second_secfpn_120m_t4dataset_j6gen2
autoware-ml train --config-name detection3d/bevfusion/lidar_voxel0075_second_secfpn_54m_nuscenes
autoware-ml train --config-name detection3d/bevfusion/lidar_voxel0170_second_secfpn_120m_t4dataset_j6gen2
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection3d/bevfusion/camera_lidar_voxel0075_second_secfpn_54m_nuscenes \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/bevfusion/camera_lidar_voxel0075_second_secfpn_54m_nuscenes \
    --weights mlruns/detection3d/bevfusion/camera_lidar_voxel0075_second_secfpn_54m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/bevfusion/camera_lidar_voxel0075_second_secfpn_54m_nuscenes \
    --weights mlruns/detection3d/bevfusion/camera_lidar_voxel0075_second_secfpn_54m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

The export follows the AWML/Autoware runtime ONNX split. Camera-LiDAR configurations export two modules: `bevfusion_image_backbone.onnx` encodes raw `uint8` multiview images (the training-time `1 / 255` normalization is baked into the graph) into `image_feats`, and `bevfusion_camera_lidar.onnx` is the main body consuming those features together with precomputed `bev_pool` metadata. LiDAR-only configurations export a single `bevfusion_lidar.onnx` main body restricted to the first three inputs below.

| Input Tensor           | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `voxels`               | Voxelized LiDAR features                             |
| `coors`                | Voxel coordinates in `(x, y, z)` order, batch-free   |
| `num_points_per_voxel` | Point count per voxel                                |
| `points`               | Raw point cloud used for LiDAR depth guidance        |
| `lidar2image`          | Raw camera projection matrices                       |
| `img_aug_matrix`       | Image augmentation (resize/crop) matrices            |
| `geom_feats`           | Precomputed BEV pooling coordinates                  |
| `kept`                 | Boolean mask for valid projected frustum points      |
| `ranks`                | Sorted BEV pooling ranks                             |
| `indices`              | Sorting indices for pooled frustum features          |
| `image_feats`          | Image backbone features from the image backbone ONNX |

Every main-body module returns the runtime detection interface: `bbox_pred` with the raw regression channels `(center, height, dim, rot, vel)` per proposal, `score` with per-proposal confidences, and `label_pred` with per-proposal class labels. Metric-space decoding happens in the runtime node.

TensorRT engine generation is disabled (`deploy.tensorrt.enabled=false`); the runtime builds engines itself using the custom sparse-convolution and `bev_pool` plugins.

## Implementation

| Path                                                          | Description                                      |
| ------------------------------------------------------------- | ------------------------------------------------ |
| `autoware_ml/models/detection3d/bevfusion.py`                 | BEVFusion model wrapper                          |
| `autoware_ml/models/detection3d/feature_extractors.py`        | LiDAR BEV and multiview image feature extractors |
| `autoware_ml/models/detection3d/view_transforms/depth_lss.py` | Multiview image-to-BEV transform                 |
| `autoware_ml/models/detection3d/fusion.py`                    | Camera-LiDAR BEV fusion layer                    |
| `autoware_ml/models/detection3d/encoders/voxel.py`            | Hard voxelization feature encoder                |
| `autoware_ml/models/detection3d/encoders/sparse.py`           | Sparse 3D convolution encoder                    |
| `autoware_ml/models/detection3d/backbones/second.py`          | SECOND backbone                                  |
| `autoware_ml/models/detection3d/necks/second_fpn.py`          | SECONDFPN neck                                   |
| `autoware_ml/models/detection3d/heads/transfusion.py`         | TransFusion detection head                       |
| `autoware_ml/models/common/backbones/resnet.py`               | ResNet multiview image backbone                  |
| `autoware_ml/models/common/necks/lss_fpn.py`                  | Multiview image neck                             |
| `autoware_ml/models/detection3d/task_modules/`                | Shared assigners, costs, coders                  |
| `autoware_ml/datamodule/nuscenes/detection3d.py`              | NuScenes detection datamodule (LiDAR-only)       |
| `autoware_ml/datamodule/t4dataset/detection3d.py`             | T4Dataset detection datamodule (LiDAR-only)      |
| `autoware_ml/datamodule/common/multiview_detection3d.py`      | Shared multiview detection dataset               |
| `autoware_ml/datamodule/nuscenes/multiview_detection3d.py`    | NuScenes multiview datamodule                    |
| `autoware_ml/datamodule/t4dataset/multiview_detection3d.py`   | T4Dataset multiview datamodule                   |
| `autoware_ml/preprocessing/detection3d/point_pillar.py`       | Pillar preprocessing                             |
| `autoware_ml/configs/tasks/detection3d/bevfusion/`            | Task configurations                              |

## Acknowledgment

The Autoware-ML BEVFusion implementation was ported from the official mmdetection3d
project by OpenMMLab.

<!-- cspell:ignore Zhijian -->
- Repository: <https://github.com/open-mmlab/mmdetection3d>
- License: Apache License 2.0
- Paper: Liu, Zhijian, et al. "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation" ICRA, 2023.
