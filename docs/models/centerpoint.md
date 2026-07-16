---
icon: lucide/scan-search
---

# CenterPoint

CenterPoint is a LiDAR-based 3D object detection model integrated under the `detection3d` task namespace. It uses a PointPillars-style frontend with a `SECOND` backbone, `SECONDFPN` neck, and CenterPoint detection head.

## Summary

| Property     | Value                                    |
| ------------ | ---------------------------------------- |
| Task         | 3D object detection                      |
| Modality     | LiDAR                                    |
| Input        | Point cloud                              |
| Output       | 3D bounding boxes and class scores       |
| Architecture | PointPillars + SECOND + SECONDFPN + head |
| Datasets     | NuScenes, T4Dataset                      |

## Available Configurations

| Config Name                                                            | Dataset   | Purpose                                                  |
| ---------------------------------------------------------------------- | --------- | -------------------------------------------------------- |
| `detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes`          | NuScenes  | Standard NuScenes 51 m configuration                     |
| `detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2` | T4Dataset | 120 m T4Dataset configuration (aligned with TransFusion) |

## Training

```bash
autoware-ml train --config-name detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes
autoware-ml train --config-name detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes \
    --weights mlruns/detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2 \
    --weights mlruns/detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt
```

The export produces the two ONNX modules consumed by `autoware_universe/perception/autoware_lidar_centerpoint`: `pts_voxel_encoder_centerpoint.onnx` encodes decorated pillar features into per-pillar descriptors, and `pts_backbone_neck_head_centerpoint.onnx` predicts the raw dense detection heads (`heatmap`, `reg`, `height`, `dim`, `rot`, `vel`) from the scattered BEV canvas. Voxelization, pillar decoration, BEV scatter, and box decoding all run in the runtime node.

## Implementation

| Path                                                    | Description                |
| ------------------------------------------------------- | -------------------------- |
| `autoware_ml/models/detection3d/centerpoint.py`         | CenterPoint model wrapper  |
| `autoware_ml/models/detection3d/encoders/pillar.py`     | Pillar encoder and scatter |
| `autoware_ml/models/detection3d/backbones/second.py`    | SECOND backbone            |
| `autoware_ml/models/detection3d/necks/second_fpn.py`    | SECONDFPN neck             |
| `autoware_ml/models/detection3d/heads/centerpoint.py`   | CenterPoint detection head |
| `autoware_ml/preprocessing/detection3d/point_pillar.py` | Pillar preprocessing       |
| `autoware_ml/datamodule/nuscenes/detection3d.py`        | NuScenes datamodule        |
| `autoware_ml/datamodule/t4dataset/detection3d.py`       | T4Dataset datamodule       |
| `autoware_ml/configs/tasks/detection3d/centerpoint/`    | Task configurations        |

## Acknowledgment

The Autoware-ML CenterPoint implementation was ported from the official mmdetection3d
project by OpenMMLab.

<!-- cspell:ignore Zhijian -->
- Repository: <https://github.com/open-mmlab/mmdetection3d>
- License: Apache License 2.0
- Paper: Yin, Tianwei, et al. "Center-based 3D Object Detection and Tracking" CVPR, 2021.
