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
| Datasets     | NuScenes, T4Dataset                                          |

## Available Configurations

| Config Name                                                             | Dataset   | Purpose                                     |
| ----------------------------------------------------------------------- | --------- | ------------------------------------------- |
| `detection3d/transfusion/voxel0075_second_secfpn_54m_nuscenes`          | NuScenes  | Official sparse NuScenes 54 m configuration |
| `detection3d/transfusion/voxel0170_second_secfpn_120m_t4dataset_j6gen2` | T4Dataset | Wide-range sparse T4Dataset configuration   |

## Training

```bash
autoware-ml train --config-name detection3d/transfusion/voxel0075_second_secfpn_54m_nuscenes
autoware-ml train --config-name detection3d/transfusion/voxel0170_second_secfpn_120m_t4dataset_j6gen2
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection3d/transfusion/voxel0075_second_secfpn_54m_nuscenes \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/transfusion/voxel0075_second_secfpn_54m_nuscenes \
    --weights mlruns/detection3d/transfusion/voxel0075_second_secfpn_54m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/transfusion/voxel0075_second_secfpn_54m_nuscenes \
    --weights mlruns/detection3d/transfusion/voxel0075_second_secfpn_54m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

The current verification scope covers ONNX export. TensorRT engine generation has not been validated yet.

## Implementation

| Path                                                    | Description                       |
| ------------------------------------------------------- | --------------------------------- |
| `autoware_ml/models/detection3d/transfusion.py`         | TransFusion model wrapper         |
| `autoware_ml/models/detection3d/encoders/voxel.py`      | Hard voxelization feature encoder |
| `autoware_ml/models/detection3d/encoders/sparse.py`     | Sparse 3D convolution encoder     |
| `autoware_ml/models/detection3d/backbones/second.py`    | SECOND backbone                   |
| `autoware_ml/models/detection3d/necks/second_fpn.py`    | SECONDFPN neck                    |
| `autoware_ml/models/detection3d/heads/transfusion.py`   | TransFusion detection head        |
| `autoware_ml/models/detection3d/task_modules/`          | Shared assigners, costs, coders   |
| `autoware_ml/datamodule/nuscenes/detection3d.py`        | NuScenes detection datamodule     |
| `autoware_ml/datamodule/t4dataset/detection3d.py`       | T4Dataset detection datamodule    |
| `autoware_ml/preprocessing/detection3d/point_pillar.py` | Pillar preprocessing              |
| `autoware_ml/configs/tasks/detection3d/transfusion/`    | Task configurations               |
