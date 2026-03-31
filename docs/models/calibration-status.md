---
icon: lucide/camera
---

# Calibration Status Classifier

The Calibration Status Classifier is a camera-LiDAR binary classification model for calibration health monitoring. It predicts whether the current sensor calibration is valid or whether recalibration is required.

## Summary

| Property     | Value                                                  |
| ------------ | ------------------------------------------------------ |
| Task         | Calibration status classification                      |
| Modality     | Camera and LiDAR                                       |
| Input        | Fused five-channel BGRDI image                         |
| Output       | Binary calibration status                              |
| Architecture | ResNet18 backbone with global average pooling and head |
| Datasets     | NuScenes, T4Dataset                                    |

## Available Configurations

| Config Name                                                                  | Dataset   | Purpose                         |
| ---------------------------------------------------------------------------- | --------- | ------------------------------- |
| `calibration_status/calibration_status_classifier/resnet18_nuscenes`         | NuScenes  | Standard training configuration |
| `calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2` | T4Dataset | Standard training configuration |

## Input Representation

The model consumes a fused image with five channels.

| Channel | Content                    |
| ------- | -------------------------- |
| `0-2`   | BGR image                  |
| `3`     | LiDAR depth projection     |
| `4`     | LiDAR intensity projection |

## Training

```bash
autoware-ml train --config-name calibration_status/calibration_status_classifier/resnet18_nuscenes
autoware-ml train --config-name calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name calibration_status/calibration_status_classifier/resnet18_nuscenes \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name calibration_status/calibration_status_classifier/resnet18_nuscenes \
    +checkpoint=mlruns/calibration_status/calibration_status_classifier/resnet18_nuscenes/<date>/<time>/checkpoints/best.ckpt
```

## Deployment

```bash
autoware-ml deploy \
    --config-name calibration_status/calibration_status_classifier/resnet18_nuscenes \
    +checkpoint=mlruns/calibration_status/calibration_status_classifier/resnet18_nuscenes/<date>/<time>/checkpoints/best.ckpt
```

## Data Pipeline

The training pipeline includes image undistortion, synthetic calibration perturbation, LiDAR-camera fusion, and channel-first tensor conversion. Calibration perturbation is used to generate positive training samples representing miscalibrated sensor pairs.

## Implementation

| Path                                                                          | Description          |
| ----------------------------------------------------------------------------- | -------------------- |
| `autoware_ml/models/calibration_status/`                                      | Model implementation |
| `autoware_ml/datamodule/nuscenes/calibration_status.py`                       | NuScenes datamodule  |
| `autoware_ml/datamodule/t4dataset/calibration_status.py`                      | T4Dataset datamodule |
| `autoware_ml/configs/tasks/calibration_status/calibration_status_classifier/` | Task configurations  |
