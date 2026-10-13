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
    --weights mlruns/calibration_status/calibration_status_classifier/resnet18_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

## Deployment

```bash
autoware-ml deploy \
    --config-name calibration_status/calibration_status_classifier/resnet18_nuscenes \
    --weights mlruns/calibration_status/calibration_status_classifier/resnet18_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

The exported model keeps the original calibration-status behavior and emits a
single probability tensor. Class labels can be derived downstream with
`argmax` when needed.

## Data Pipeline

The training pipeline includes image undistortion, synthetic calibration perturbation, LiDAR-camera fusion, and channel-first tensor conversion. Calibration perturbation is used to generate positive training samples representing miscalibrated sensor pairs.

## Implementation

| Path                                                                          | Description                                      |
| ----------------------------------------------------------------------------- | ------------------------------------------------ |
| `autoware_ml/models/calibration_status/`                                      | Model implementation                             |
| `autoware_ml/datamodule/t4dataset/dataset.py`                                 | T4Dataset records as a dataset                   |
| `autoware_ml/datamodule/nuscenes/dataset.py`                                  | nuScenes dataset                                 |
| `autoware_ml/transforms/camera_lidar/camera_lidar.py`                         | Calibration perturbation and LiDAR-camera fusion |
| `autoware_ml/configs/datamodule/`                                             | Datamodule, dataset and transform configurations |
| `autoware_ml/configs/tasks/calibration_status/calibration_status_classifier/` | Task configurations                              |
