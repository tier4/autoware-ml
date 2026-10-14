---
icon: lucide/scan-line
---

# BEVFusion

BEVFusion is a camera-LiDAR 3D object detection model integrated under the `detection3d` task namespace. It combines a sparse-voxel LiDAR branch (hard voxelization + sparse 3D convolution encoder), a multiview image branch, a LiDAR-depth-guided `DepthLSSTransform` view transform, and a convolutional BEV fusion layer before the TransFusion detection head. The bundled experiments are LiDAR-only: they leave `camera_network` unset and skip the image branch entirely.

## Summary

| Property     | Value                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------- |
| Task         | 3D object detection                                                                         |
| Modality     | LiDAR (camera-LiDAR supported by the model, no bundled experiment yet)                      |
| Input        | Point cloud, optionally with synchronized multiview images                                  |
| Output       | 3D bounding boxes and class scores                                                          |
| Architecture | Sparse voxel encoder + multiview camera backbone/FPN + DepthLSS + fusion + TransFusion head |
| Datasets     | T4Dataset                                                                                   |

## Available Configurations

Experiment configs live under `autoware_ml/configs/tasks/detection3d/bevfusion/` and are passed to the CLI as the path relative to `tasks/`.

| Config Name                                                                                                         | Dataset           | Purpose                                                              |
| ------------------------------------------------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------------------- |
| `detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_30e_t4dataset_120m_gen1_base`    | T4Dataset gen1    | LiDAR-only 120 m base training from scratch, 30 epochs               |
| `detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_30e_t4dataset_120m_jpntaxi_base` | T4Dataset jpntaxi | LiDAR-only 120 m fine-tuning from a pretrained checkpoint, 30 epochs |
| `detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base`  | T4Dataset j6gen2  | LiDAR-only 120 m fine-tuning from a pretrained checkpoint, 50 epochs |

The configs are layered so that each file owns one concern. Camera-LiDAR experiments will add a sibling directory next to `bevfusion_lidar/` that configures the camera branch and fuser.

| File                                                       | Owns                                                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `base.yaml`                                                | Model architecture, voxel preprocessor, optimizer, trainer settings, and ONNX export modules     |
| `t4dataset/base_120m.yaml`                                 | 120 m point cloud range, evaluation ranges, proposal count, NMS and score thresholds, transforms |
| `t4dataset/bevfusion_lidar/base_lidar_voxel0170_120m.yaml` | 0.17 m voxel size, sparse encoder grid shapes, and head output stride                            |
| `t4dataset/bevfusion_lidar/lidar_voxel0170_*.yaml`         | Database selection, batch size, experiment name, epochs, and optional pretrained weights         |

## Training

Train the base model from scratch:

```bash
autoware-ml train --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_30e_t4dataset_120m_gen1_base
```

The `jpntaxi` and `j6gen2` configs declare `weights: ???`, so they must be started from a pretrained checkpoint. Pass it with `--weights`; the config already lowers the learning rate for fine-tuning:

```bash
autoware-ml train \
    --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_30e_t4dataset_120m_gen1_base/<run_id>/artifacts/checkpoints/best.ckpt
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_30e_t4dataset_120m_gen1_base \
    +trainer.fast_dev_run=true
```

Checkpoints are written to `mlruns/<config_name>/<run_id>/artifacts/checkpoints/` as `best.ckpt` and `last.ckpt`. Resume an interrupted run with `--resume-checkpoint` pointing at `last.ckpt`.

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base/<run_id>/artifacts/checkpoints/best.ckpt
```

Evaluation runs on a single device by default so metrics are deterministic. Pass `--use-config-devices` to keep the `trainer.devices` from the config.

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base/<run_id>/artifacts/checkpoints/best.ckpt \
    --release v1.0.0
```

The export produces the ONNX modules consumed by `autoware_universe/perception/autoware_bevfusion`. LiDAR-only configurations export a single `bevfusion_lidar.onnx` main body restricted to the first three inputs below. Camera-LiDAR models export two modules: `bevfusion_image_backbone.onnx` encodes raw `uint8` multiview images (the training-time `1 / 255` normalization is baked into the graph) into `image_feats`, and `bevfusion_camera_lidar.onnx` is the main body consuming those features together with precomputed `bev_pool` metadata.

| Input Tensor           | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `voxels`               | Voxelized LiDAR features                             |
| `coors`                | Voxel coordinates in `(z, y, x)` order, batch-free   |
| `num_points_per_voxel` | Point count per voxel                                |
| `image_feats`          | Image backbone features from the image backbone ONNX |
| `depth_maps`           | LiDAR depth maps used to guide the view transform    |
| `geom_feats`           | Precomputed BEV pooling coordinates                  |
| `kept`                 | Boolean mask for valid projected frustum points      |
| `ranks`                | Sorted BEV pooling ranks                             |
| `indices`              | Sorting indices for pooled frustum features          |

Every main-body module returns the runtime detection interface: `bbox_pred` with the raw regression channels `(center, height, dim, rot, vel)` per proposal, `score` with per-proposal confidences, and `label_pred` with per-proposal class labels. Metric-space decoding happens in the runtime node.

Exports land in `mlruns/<config_name>/<run_id>/artifacts/exports/` of a dedicated deploy run linked to the training run. Every module is stamped with its provenance and the `--release` version; omit `--release` only for throwaway exports, which are stamped `unversioned`. TensorRT engine generation is disabled (`deploy.tensorrt.enabled=false`); the runtime builds engines itself using the custom sparse-convolution and `bev_pool` plugins.

## Implementation

| Path                                                                         | Description                                      |
| ---------------------------------------------------------------------------- | ------------------------------------------------ |
| `autoware_ml/models/detection3d/main_modules/bevfusion.py`                   | BEVFusion model wrapper                          |
| `autoware_ml/models/detection3d/main_modules/bevfusions/bevfusion_lidar.py`  | LiDAR BEV network                                |
| `autoware_ml/models/detection3d/main_modules/bevfusions/bevfusion_camera.py` | Multiview image network                          |
| `autoware_ml/models/detection3d/main_modules/bevfusions/export_wrappers.py`  | ONNX export wrappers                             |
| `autoware_ml/models/detection3d/view_transforms/depth_lss.py`                | Multiview image-to-BEV transform                 |
| `autoware_ml/models/detection3d/main_modules/bevfusions/fuser.py`            | Camera-LiDAR BEV fusion layer                    |
| `autoware_ml/models/detection3d/encoders/voxel.py`                           | Hard voxelization feature encoder                |
| `autoware_ml/models/detection3d/encoders/sparse/sparse_encoder.py`           | Sparse 3D convolution encoder                    |
| `autoware_ml/models/detection3d/backbones/second.py`                         | SECOND backbone                                  |
| `autoware_ml/models/detection3d/necks/second_fpn.py`                         | SECONDFPN neck                                   |
| `autoware_ml/models/detection3d/heads/transfusions/transfusion_head.py`      | TransFusion detection head                       |
| `autoware_ml/models/common/backbones/resnet.py`                              | ResNet multiview image backbone                  |
| `autoware_ml/models/common/necks/lss_fpn.py`                                 | Multiview image neck                             |
| `autoware_ml/models/detection3d/task_modules/`                               | Shared assigners, costs, coders                  |
| `autoware_ml/datamodule/t4dataset/dataset.py`                                | T4Dataset records as a dataset                   |
| `autoware_ml/datamodule/nuscenes/dataset.py`                                 | nuScenes dataset                                 |
| `autoware_ml/datamodule/common/detection3d.py`                               | 3D boxes of a record                             |
| `autoware_ml/transforms/camera/`                                             | Multiview image loading and augmentation         |
| `autoware_ml/configs/datamodule/`                                            | Datamodule, dataset and transform configurations |
| `autoware_ml/preprocessing/detection3d/point_pillar_preprocessor.py`         | Pillar preprocessing                             |
| `autoware_ml/configs/tasks/detection3d/bevfusion/`                           | Experiment configurations                        |

## Acknowledgment

The Autoware-ML BEVFusion implementation was ported from the official mmdetection3d
project by OpenMMLab.

<!-- cspell:ignore Zhijian -->
- Repository: <https://github.com/open-mmlab/mmdetection3d>
- License: Apache License 2.0
- Paper: Liu, Zhijian, et al. "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation" ICRA, 2023.
