---
icon: lucide/cctv
---

# StreamPETR

StreamPETR is a camera-based 3D object detection model integrated under the `detection3d` task namespace. It uses a multiview image backbone, a feature pyramid neck, and a native query-based detection head.

## Summary

| Property     | Value                                     |
| ------------ | ----------------------------------------- |
| Task         | 3D object detection                       |
| Modality     | Camera                                    |
| Input        | Synchronized multiview images             |
| Output       | 3D bounding boxes and class scores        |
| Architecture | Multiview VoVNet/FPN + query decoder head |
| Datasets     | NuScenes, T4Dataset                       |

## Available Configurations

| Config Name                                           | Dataset   | Purpose                          |
| ----------------------------------------------------- | --------- | -------------------------------- |
| `detection3d/streampetr/vov_320x800_nuscenes`         | NuScenes  | Standard NuScenes configuration  |
| `detection3d/streampetr/vov_480x640_t4dataset_j6gen2` | T4Dataset | Standard T4Dataset configuration |

## Training

```bash
autoware-ml train --config-name detection3d/streampetr/vov_320x800_nuscenes
autoware-ml train --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_320x800_nuscenes \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/streampetr/vov_320x800_nuscenes \
    --weights mlruns/detection3d/streampetr/vov_320x800_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/streampetr/vov_320x800_nuscenes \
    --weights mlruns/detection3d/streampetr/vov_320x800_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

The current verification scope covers ONNX export. TensorRT engine generation has not been validated yet.

## Implementation

| Path                                                        | Description                        |
| ----------------------------------------------------------- | ---------------------------------- |
| `autoware_ml/models/detection3d/streampetr.py`              | StreamPETR model wrapper           |
| `autoware_ml/models/detection3d/heads/streampetr.py`        | Query-based detection head         |
| `autoware_ml/models/common/backbones/vovnet.py`             | Multiview image backbone           |
| `autoware_ml/models/common/necks/lss_fpn.py`                | Multiview feature pyramid neck     |
| `autoware_ml/models/detection3d/task_modules/`              | Shared assigners, costs, coders    |
| `autoware_ml/datamodule/common/multiview_detection3d.py`    | Shared multiview detection dataset |
| `autoware_ml/datamodule/nuscenes/multiview_detection3d.py`  | NuScenes multiview datamodule      |
| `autoware_ml/datamodule/t4dataset/multiview_detection3d.py` | T4Dataset multiview datamodule     |
| `autoware_ml/configs/tasks/detection3d/streampetr/`         | Task configurations                |

## Acknowledgment

<!-- cspell:ignore exiawsh -->
The Autoware-ML StreamPETR implementation was ported from the official streampetr
project by exiawsh.

<!-- cspell:ignore Shihao -->
- Repository: <https://github.com/exiawsh/streampetr>
- License: Apache License 2.0
- Paper: Wang, Shihao, et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection" ICCV, 2023.
