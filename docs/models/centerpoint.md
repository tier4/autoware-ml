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
| Datasets     | T4Dataset                                |

## Available Configurations

Experiment configs live under `autoware_ml/configs/tasks/detection3d/centerpoint/` and are passed to the CLI as the path relative to `tasks/`.

| Config Name                                                                                    | Dataset           | Purpose                                                             |
| ---------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------- |
| `detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_30e_t4dataset_120m_gen1_base`    | T4Dataset gen1    | 120 m base training from scratch, 30 epochs                         |
| `detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_30e_t4dataset_120m_jpntaxi_base` | T4Dataset jpntaxi | 120 m fine-tuning from a pretrained checkpoint, 30 epochs, lower LR |
| `detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base`  | T4Dataset j6gen2  | 120 m fine-tuning from a pretrained checkpoint, 50 epochs, lower LR |

The configs are layered so that each file owns one concern:

| File                                | Owns                                                                                            |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- |
| `base.yaml`                         | Model architecture, pillar preprocessor, optimizer, trainer settings, and ONNX export modules   |
| `t4dataset/base_120m.yaml`          | 120 m point cloud range, evaluation ranges, NMS radius, dataloaders, and the transform pipeline |
| `t4dataset/base_voxel024_120m.yaml` | 0.24 m voxel size, BEV canvas shape, backbone strides, and head output stride                   |
| `t4dataset/voxel024_*.yaml`         | Database selection, batch size, experiment name, epochs, and optional pretrained weights        |

## Training

Train the base model from scratch:

```bash
autoware-ml train --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_30e_t4dataset_120m_gen1_base
```

The `jpntaxi` and `j6gen2` configs declare `weights: ???`, so they must be started from a pretrained checkpoint. Pass it with `--weights`; the config already lowers the learning rate for fine-tuning:

```bash
autoware-ml train \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_30e_t4dataset_120m_gen1_base/<run_id>/artifacts/checkpoints/best.ckpt
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_30e_t4dataset_120m_gen1_base \
    +trainer.fast_dev_run=true
```

Checkpoints are written to `mlruns/<config_name>/<run_id>/artifacts/checkpoints/` as `best.ckpt` and `last.ckpt`. Resume an interrupted run with `--resume-checkpoint` pointing at `last.ckpt`.

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base/<run_id>/artifacts/checkpoints/best.ckpt
```

Evaluation runs on a single device by default so metrics are deterministic. Pass `--use-config-devices` to keep the `trainer.devices` from the config.

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base/<run_id>/artifacts/checkpoints/best.ckpt \
    --release v1.0.0
```

The export produces the two ONNX modules consumed by `autoware_universe/perception/autoware_lidar_centerpoint`: `pts_voxel_encoder_centerpoint.onnx` encodes decorated pillar features into per-pillar descriptors, and `pts_backbone_neck_head_centerpoint.onnx` predicts the raw dense detection heads (`heatmap`, `reg`, `height`, `dim`, `rot`, `vel`) from the scattered BEV canvas. Voxelization, pillar decoration, BEV scatter, and box decoding all run in the runtime node.

Exports land in `mlruns/<config_name>/<run_id>/artifacts/exports/` of a dedicated deploy run linked to the training run. Every module is stamped with its provenance and the `--release` version; omit `--release` only for throwaway exports, which are stamped `unversioned`. TensorRT engine generation is disabled in the base config (`deploy.tensorrt.enabled=false`); the runtime builds engines itself.

## Implementation

| Path                                                                      | Description                                      |
| ------------------------------------------------------------------------- | ------------------------------------------------ |
| `autoware_ml/models/detection3d/main_modules/centerpoint.py`              | CenterPoint model wrapper                        |
| `autoware_ml/models/detection3d/encoders/pillars/pillar_feature_net.py`   | Pillar feature encoder                           |
| `autoware_ml/models/detection3d/encoders/pillars/point_pillar_scatter.py` | Pillar scatter to BEV                            |
| `autoware_ml/models/detection3d/backbones/second.py`                      | SECOND backbone                                  |
| `autoware_ml/models/detection3d/necks/second_fpn.py`                      | SECONDFPN neck                                   |
| `autoware_ml/models/detection3d/heads/centerhead.py`                      | CenterPoint detection head                       |
| `autoware_ml/preprocessing/detection3d/point_pillar_preprocessor.py`      | Pillar preprocessing                             |
| `autoware_ml/datamodule/t4dataset/dataset.py`                             | T4Dataset records as a dataset                   |
| `autoware_ml/datamodule/nuscenes/dataset.py`                              | nuScenes dataset                                 |
| `autoware_ml/datamodule/common/detection3d.py`                            | 3D boxes of a record                             |
| `autoware_ml/configs/datamodule/`                                         | Datamodule, dataset and transform configurations |
| `autoware_ml/configs/tasks/detection3d/centerpoint/`                      | Experiment configurations                        |

## Acknowledgment

The Autoware-ML CenterPoint implementation was ported from the official mmdetection3d
project by OpenMMLab.

<!-- cspell:ignore Zhijian -->
- Repository: <https://github.com/open-mmlab/mmdetection3d>
- License: Apache License 2.0
- Paper: Yin, Tianwei, et al. "Center-based 3D Object Detection and Tracking" CVPR, 2021.
