---
icon: lucide/radar
---

# TransFusion

TransFusion is a LiDAR-based 3D object detection model integrated under the `detection3d` task namespace. It uses a sparse-voxel frontend (hard voxelization + sparse 3D convolution encoder) with a `SECOND` backbone, `SECONDFPN` neck, and native TransFusion detection head.

## Summary

| Property     | Value                                                        |
| ------------ | ------------------------------------------------------------ |
| Task         | 3D object detection                                          |
| Modality     | LiDAR                                                        |
| Input        | Point cloud                                                  |
| Output       | 3D bounding boxes and class scores                           |
| Architecture | Sparse voxel encoder + SECOND + SECONDFPN + TransFusion head |
| Datasets     | T4Dataset                                                    |

## Available Configurations

The LiDAR-only [BEVFusion](bevfusion.md) experiments run the same sparse-voxel encoder, `SECOND` backbone, `SECONDFPN` neck, and TransFusion head stack; the TransFusion experiments below train that stack without the BEVFusion wrapper.

Experiment configs live under `autoware_ml/configs/tasks/detection3d/transfusion/` and follow the CenterPoint layout: a `base.yaml` with the model and export modules, a `t4dataset/base_<range>.yaml` with the range-dependent settings and transforms, a voxel-level base, and one leaf per database. The bundled leaves are `detection3d/transfusion/t4dataset/voxel0170_second_secfpn_b16_30e_t4dataset_120m_j6gen2_base` and `detection3d/transfusion/nuscenes/voxel0075_second_secfpn_b16_30e_nuscenes_54m`.

## Training

An experiment is trained through the standard entrypoint, with `--weights` supplying a pretrained checkpoint when the config declares `weights: ???`:

```bash
autoware-ml train --config-name detection3d/transfusion/t4dataset/voxel0170_second_secfpn_b16_30e_t4dataset_120m_j6gen2_base
```

Checkpoints are written to `mlruns/<config_name>/<run_id>/artifacts/checkpoints/` as `best.ckpt` and `last.ckpt`.

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/transfusion/t4dataset/voxel0170_second_secfpn_b16_30e_t4dataset_120m_j6gen2_base \
    --weights mlruns/detection3d/transfusion/t4dataset/voxel0170_second_secfpn_b16_30e_t4dataset_120m_j6gen2_base/<run_id>/artifacts/checkpoints/best.ckpt
```

Evaluation runs on a single device by default so metrics are deterministic. Pass `--use-config-devices` to keep the `trainer.devices` from the config.

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/transfusion/t4dataset/voxel0170_second_secfpn_b16_30e_t4dataset_120m_j6gen2_base \
    --weights mlruns/detection3d/transfusion/t4dataset/voxel0170_second_secfpn_b16_30e_t4dataset_120m_j6gen2_base/<run_id>/artifacts/checkpoints/best.ckpt \
    --release v1.0.0
```

The model exports one ONNX main body wrapping the voxel encoder, sparse encoder, backbone, neck, and head.

| Tensor          | Direction | Description                              |
| --------------- | --------- | ---------------------------------------- |
| `voxels`        | Input     | Voxelized LiDAR features                 |
| `num_points`    | Input     | Point count per voxel                    |
| `coors`         | Input     | Voxel coordinates with the batch index   |
| `cls_score0`    | Output    | Per-proposal class heatmap logits        |
| `bbox_pred0`    | Output    | Per-proposal raw box regression channels |
| `dir_cls_pred0` | Output    | Per-proposal direction classification    |

The `height` channel of `bbox_pred0` is the z of the box gravity center, which is how `autoware_lidar_transfusion` reads it.

Exports land in `mlruns/<config_name>/<run_id>/artifacts/exports/` of a dedicated deploy run linked to the training run. Every module is stamped with its provenance and the `--release` version; omit `--release` only for throwaway exports, which are stamped `unversioned`. The current verification scope covers ONNX export. TensorRT engine generation has not been validated yet.

## Implementation

| Path                                                                    | Description                                      |
| ----------------------------------------------------------------------- | ------------------------------------------------ |
| `autoware_ml/models/detection3d/main_modules/transfusion.py`            | TransFusion model wrapper                        |
| `autoware_ml/models/detection3d/encoders/voxel.py`                      | Hard voxelization feature encoder                |
| `autoware_ml/models/detection3d/encoders/sparse/sparse_encoder.py`      | Sparse 3D convolution encoder                    |
| `autoware_ml/models/detection3d/backbones/second.py`                    | SECOND backbone                                  |
| `autoware_ml/models/detection3d/necks/second_fpn.py`                    | SECONDFPN neck                                   |
| `autoware_ml/models/detection3d/heads/transfusions/transfusion_head.py` | TransFusion detection head                       |
| `autoware_ml/models/detection3d/task_modules/`                          | Shared assigners, costs, coders                  |
| `autoware_ml/datamodule/t4dataset/dataset.py`                           | T4Dataset records as a dataset                   |
| `autoware_ml/datamodule/nuscenes/dataset.py`                            | nuScenes dataset                                 |
| `autoware_ml/datamodule/common/detection3d.py`                          | 3D boxes of a record                             |
| `autoware_ml/configs/datamodule/`                                       | Datamodule, dataset and transform configurations |
| `autoware_ml/preprocessing/detection3d/point_pillar_preprocessor.py`    | Pillar preprocessing                             |
| `autoware_ml/configs/tasks/detection3d/transfusion/`                    | Experiment configurations                        |

## Acknowledgment

The Autoware-ML TransFusion implementation was ported from the official mmdetection3d
project by OpenMMLab.

<!-- cspell:ignore Xuyang -->
- Repository: <https://github.com/open-mmlab/mmdetection3d>
- License: Apache License 2.0
- Paper: Bai, Xuyang, et al. "TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers" CVPR, 2022.
