---
icon: lucide/dice-5
---

# PointTransformerV3

PointTransformerV3 is a LiDAR-based 3D semantic segmentation model that combines serialized point attention with sparse convolution blocks. It is integrated under the `segmentation3d` task namespace for NuScenes and T4Dataset.

The training path uses `flash-attn` for serialized attention. The export path
automatically disables flash attention and falls back to the standard attention
implementation so ONNX export remains supported.

## Summary

| Property     | Value                                                               |
| ------------ | ------------------------------------------------------------------- |
| Task         | 3D semantic segmentation                                            |
| Modality     | LiDAR                                                               |
| Input        | Point cloud                                                         |
| Output       | Point-wise semantic class labels                                    |
| Architecture | PointTransformerV3 with sparse convolution stem and flash attention |
| Datasets     | NuScenes, T4Dataset                                                 |

## Available Configurations

| Config Name                                          | Dataset   | Purpose                          |
| ---------------------------------------------------- | --------- | -------------------------------- |
| `segmentation3d/ptv3/voxel005_102m_nuscenes`         | NuScenes  | Standard NuScenes configuration  |
| `segmentation3d/ptv3/voxel005_102m_t4dataset_j6gen2` | T4Dataset | Standard T4Dataset configuration |

## Training

```bash
autoware-ml train --config-name segmentation3d/ptv3/voxel005_102m_nuscenes
autoware-ml train --config-name segmentation3d/ptv3/voxel005_102m_t4dataset_j6gen2
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name segmentation3d/ptv3/voxel005_102m_nuscenes \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name segmentation3d/ptv3/voxel005_102m_nuscenes \
    +checkpoint=mlruns/segmentation3d/ptv3/voxel005_102m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

## Deployment

PointTransformerV3 ONNX export is available. The generic TensorRT stage remains
disabled in Autoware-ML because PTv3 requires a runtime with matching sparse
convolution plugins.

```bash
autoware-ml deploy \
    --config-name segmentation3d/ptv3/voxel005_102m_nuscenes \
    +checkpoint=mlruns/segmentation3d/ptv3/voxel005_102m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

The deployment command switches PTv3 attention blocks into non-flash export
mode automatically.

The exported ONNX model returns both `pred_labels` and `pred_probs`. The
probability output is produced by a final softmax layer, while training and
evaluation continue to use logits inside the Lightning model.

Deployment uses an explicit PTv3 export wrapper and a copied backbone, so the
training model is not mutated when export-only sparse-convolution and
serialization settings are applied.

## Implementation

| Path                                                  | Description                                |
| ----------------------------------------------------- | ------------------------------------------ |
| `autoware_ml/models/segmentation3d/ptv3.py`           | PTv3 Lightning model wrapper               |
| `autoware_ml/models/segmentation3d/backbones/ptv3.py` | Reusable PTv3 backbone components          |
| `autoware_ml/utils/point_cloud/serialization/`        | Shared point-cloud serialization           |
| `autoware_ml/utils/point_cloud/`                      | Shared point-cloud utilities               |
| `autoware_ml/ops/segment/segment_csr.py`              | Segment reduction export operator          |
| `autoware_ml/losses/segmentation3d/`                  | Segmentation losses used by PTv3           |
| `autoware_ml/datamodule/nuscenes/segmentation3d.py`   | NuScenes datamodule                        |
| `autoware_ml/datamodule/t4dataset/segmentation3d.py`  | T4Dataset datamodule                       |
| `autoware_ml/transforms/point_cloud/`                 | Shared point-cloud transforms used by PTv3 |
| `autoware_ml/configs/tasks/segmentation3d/ptv3/`      | Task configurations                        |

## Acknowledgment

The Autoware-ML PointTransformerV3 implementation was ported from the official PointTransformerV3
project by Pointcept.

<!-- cspell:ignore Xiaoyang -->
- Repository: <https://github.com/Pointcept/PointTransformerV3>
- License: MIT
- Paper: Wu, Xiaoyang, et al. "Point Transformer V3: Simpler, Faster, Stronger." CVPR, 2024.
