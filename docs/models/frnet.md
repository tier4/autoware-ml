---
icon: lucide/dice-5
---

# FRNet

FRNet is a LiDAR-based 3D semantic segmentation model built on frustum and range-view feature fusion. It is integrated under the `segmentation3d` task namespace for NuScenes and T4Dataset.

## Summary

| Property     | Value                                               |
| ------------ | --------------------------------------------------- |
| Task         | 3D semantic segmentation                            |
| Modality     | LiDAR                                               |
| Input        | Point cloud                                         |
| Output       | Point-wise semantic class labels                    |
| Architecture | Frustum encoder, FRNet backbone, segmentation heads |
| Datasets     | NuScenes, T4Dataset                                 |

## Available Configurations

| Config Name                                   | Dataset   | Purpose                         |
| --------------------------------------------- | --------- | ------------------------------- |
| `segmentation3d/frnet/hdl32e_nuscenes`        | NuScenes  | Standard NuScenes configuration |
| `segmentation3d/frnet/ot128_t4dataset_j6gen2` | T4Dataset | T4Dataset OT128 configuration   |
| `segmentation3d/frnet/qt128_t4dataset_j6gen2` | T4Dataset | T4Dataset QT128 configuration   |

## Training

```bash
autoware-ml train --config-name segmentation3d/frnet/hdl32e_nuscenes
autoware-ml train --config-name segmentation3d/frnet/ot128_t4dataset_j6gen2
autoware-ml train --config-name segmentation3d/frnet/qt128_t4dataset_j6gen2
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name segmentation3d/frnet/hdl32e_nuscenes \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name segmentation3d/frnet/hdl32e_nuscenes \
    +checkpoint=mlruns/segmentation3d/frnet/hdl32e_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

## Deployment

```bash
autoware-ml deploy \
    --config-name segmentation3d/frnet/hdl32e_nuscenes \
    +checkpoint=mlruns/segmentation3d/frnet/hdl32e_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

To validate ONNX export without building a TensorRT engine:

```bash
autoware-ml deploy \
    --config-name segmentation3d/frnet/hdl32e_nuscenes \
    +checkpoint=mlruns/segmentation3d/frnet/hdl32e_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

The exported ONNX model returns point-wise semantic probabilities through a
final softmax layer. Training and evaluation continue to use logits inside the
Lightning model.

Deployment uses an explicit FRNet export wrapper with copied model
submodules and explicit single-sample export metadata, so export-specific
behavior does not mutate the training model.

## Data Pipeline

The FRNet preprocessing path converts raw point clouds into frustum and range-view representations and prepares both point-level and range-view supervision targets. The training pipeline includes dataset-specific augmentations and FRNet-specific transforms such as frustum mixing, instance copy, and range interpolation.

The standard FRNet training configs follow the AWML experiment contract:

- `AdamW` with `OneCycleLR`
- step-based validation every `1500` training steps
- best-checkpoint selection by validation loss
- mix augmentations that apply a secondary-sample transform pipeline before frustum mixing and instance copy

## Implementation

| Path                                                        | Description                                |
| ----------------------------------------------------------- | ------------------------------------------ |
| `autoware_ml/models/segmentation3d/frnet.py`                | FRNet Lightning model wrapper              |
| `autoware_ml/models/segmentation3d/encoders/frnet.py`       | Frustum feature encoder                    |
| `autoware_ml/models/segmentation3d/backbones/frnet.py`      | FRNet backbone                             |
| `autoware_ml/models/segmentation3d/heads/frnet.py`          | FRNet segmentation heads                   |
| `autoware_ml/losses/segmentation3d/`                        | Segmentation losses used by FRNet          |
| `autoware_ml/datamodule/nuscenes/segmentation3d.py`         | NuScenes segmentation datamodule           |
| `autoware_ml/datamodule/t4dataset/segmentation3d.py`        | T4Dataset segmentation datamodule          |
| `autoware_ml/transforms/segmentation3d/`                    | Segmentation task transforms used by FRNet |
| `autoware_ml/preprocessing/segmentation3d/frustum_range.py` | Frustum and range preprocessing            |
| `autoware_ml/configs/tasks/segmentation3d/frnet/`           | Task configurations                        |

## Acknowledgment

The Autoware-ML FRNet implementation was ported from the official FRNet project.

<!-- cspell:ignore Xiang -->
- Repository: <https://github.com/Xiangxu-0103/FRNet>
- License: Apache 2.0
- Paper: Xu, Xiang, et al. "FRNet: Frustum-Range Networks for Scalable LiDAR Segmentation." IEEE Transactions on Image Processing, vol. 34, pp. 2173-2186, 2025.
