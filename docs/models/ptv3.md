---
icon: lucide/dice-5
---

# PointTransformerV3

PointTransformerV3 is a LiDAR encoder for LiDAR 3D semantic segmentation and
3D object detection. The shared component is the hierarchical PTv3 **encoder**
(`PointTransformerV3Encoder`); each task owns its head:

- Segmentation: `PTv3SegDecoderHead` unpools the deepest encoder stage back to
  full resolution through the encoder skip chain and classifies each point.
- Detection: `PTv3DetBEVNeck` fuses the two coarsest encoder stages
  (`PTv3DetFeatureFusion`), scatters them onto a dense BEV canvas at the
  detection-head resolution, and feeds a `TransFusionHead`.

Because the detection branch taps the encoder only, finetuning the
segmentation head with a frozen encoder cannot change detection outputs - the
segmentation and detection checkpoints of a joint model stay independently
finetunable and stackable at deploy time.

The training path uses `flash-attn` for serialized attention. The export path
automatically disables flash attention and falls back to the standard attention
implementation so ONNX export remains supported.

## Summary

| Property     | Value                                                          |
| ------------ | -------------------------------------------------------------- |
| Task         | 3D semantic segmentation, 3D object detection                  |
| Modality     | LiDAR                                                          |
| Input        | Point cloud                                                    |
| Output       | Point-wise semantic labels or 3D boxes/scores/classes          |
| Architecture | PointTransformerV3 with sparse convolution stem and task heads |
| Datasets     | NuScenes, T4Dataset                                            |

## Available Configurations

| Config Name                                          | Task                         | Dataset   | Head        | Range | Purpose                        |
| ---------------------------------------------------- | ---------------------------- | --------- | ----------- | ----- | ------------------------------ |
| `segmentation3d/ptv3/voxel005_51m_nuscenes`          | segmentation3d               | NuScenes  | Seg decoder | 51 m  | NuScenes segmentation          |
| `segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2` | segmentation3d               | T4Dataset | Seg decoder | 122 m | T4Dataset segmentation         |
| `detection3d/ptv3/voxel005_51m_nuscenes`             | detection3d                  | NuScenes  | TransFusion | 51 m  | NuScenes detection             |
| `detection3d/ptv3/voxel012_122m_t4dataset_j6gen2`    | detection3d                  | T4Dataset | TransFusion | 122 m | T4Dataset detection            |
| `multi/ptv3/voxel005_51m_nuscenes`                   | segmentation3d + detection3d | NuScenes  | TransFusion | 51 m  | Joint segmentation + detection |
| `multi/ptv3/voxel012_122m_t4dataset_j6gen2`          | segmentation3d + detection3d | T4Dataset | TransFusion | 122 m | Joint segmentation + detection |

## Training

```bash
autoware-ml train --config-name segmentation3d/ptv3/voxel005_51m_nuscenes
autoware-ml train --config-name segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2
autoware-ml train --config-name detection3d/ptv3/voxel005_51m_nuscenes
autoware-ml train --config-name detection3d/ptv3/voxel012_122m_t4dataset_j6gen2
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name segmentation3d/ptv3/voxel005_51m_nuscenes \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name segmentation3d/ptv3/voxel005_51m_nuscenes \
    --weights mlruns/segmentation3d/ptv3/voxel005_51m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt

autoware-ml test \
    --config-name detection3d/ptv3/voxel012_122m_t4dataset_j6gen2 \
    --weights mlruns/detection3d/ptv3/voxel012_122m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt
```

## Deployment

PointTransformerV3 ONNX export is available. The generic TensorRT stage remains
disabled in Autoware-ML because PTv3 requires a runtime with matching sparse
convolution plugins.

```bash
autoware-ml deploy \
    --config-name segmentation3d/ptv3/voxel005_51m_nuscenes \
    --weights mlruns/segmentation3d/ptv3/voxel005_51m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false

autoware-ml deploy \
    --config-name detection3d/ptv3/voxel012_122m_t4dataset_j6gen2 \
    --weights mlruns/detection3d/ptv3/voxel012_122m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

The deployment command switches PTv3 attention blocks into non-flash export
mode automatically.

The exported ONNX model returns both `pred_labels` and `pred_probs`. The
probability output is produced by a final softmax layer, while training and
evaluation continue to use logits inside the Lightning model.

Deployment uses an explicit PTv3 export wrapper and a copied encoder, so the
training model is not mutated when export-only sparse-convolution and
serialization settings are applied.

### ONNX Preprocessing Contract

The exported PTv3 encoder ONNX expects all pooling-shape metadata to be
generated by preprocessing outside the engine. `N_0` is the input voxel count,
and `O` is the number of serialization orders.

| Base input        | Shape      | Meaning                       |
| ----------------- | ---------- | ----------------------------- |
| `grid_coord`      | `[N_0, 3]` | Integer voxel coordinates.    |
| `feat`            | `[N_0, 4]` | Per-voxel input features.     |
| `serialized_code` | `[O, N_0]` | Serialization code per order. |

For every encoder pooling stage `i`, with input voxel count `N_i` and pooled
output voxel count `M_i`, preprocessing also provides:

| Pooling metadata                          | Shape       | Meaning                                                       |
| ----------------------------------------- | ----------- | ------------------------------------------------------------- |
| `serialized_pooling_i_indices`            | `[N_i]`     | ONNX `Gather` indices grouping features before CSR reduction. |
| `serialized_pooling_i_indptr`             | `[M_i + 1]` | CSR row pointer consumed by `autoware::SegmentCSR`.           |
| `serialized_pooling_i_cluster`            | `[N_i]`     | Input voxel to pooled voxel id mapping for unpooling.         |
| `serialized_pooling_i_head_indices`       | `[M_i]`     | Representative input voxel for each pooled voxel.             |
| `serialized_pooling_i_grid_coord`         | `[M_i, 3]`  | Integer coordinates of pooled voxels.                         |
| `serialized_pooling_i_serialized_order`   | `[O, M_i]`  | Serialization order for pooled voxels.                        |
| `serialized_pooling_i_serialized_inverse` | `[O, M_i]`  | Inverse serialization order for pooled voxels.                |

Because preprocessing resolves every pooling shape ahead of time, the exported
graph contains no data-dependent pooling shape discovery. Pooled feature
reduction is implemented with native ONNX `Gather` and the
`autoware::SegmentCSR` plugin.

### Split-export module contract

The split export produces one graph per `deploy.onnx.modules` entry:

- `encoder` - the encoder; outputs per-stage features `point_feat_0` …
  `point_feat_{S-1}` (finest to deepest, `S` encoder stages). It consumes the
  per-stage pooling metadata **except** `serialized_pooling_i_cluster`, which
  only drives head-side unpooling and enters the head graphs as
  `pooling_cluster_i` instead.
- `seg3d_head` - consumes all per-stage features plus the per-pooling
  `pooling_cluster_i` tensors (the `serialized_pooling_i_cluster` metadata)
  and outputs `pred_labels`/`pred_probs`. For every decoder stage `i` with
  attention blocks (`dec_depths[i] > 0`) the graph additionally consumes that
  stage's serialization metadata, under the same names as the encoder
  inputs: `serialized_pooling_{i-1}_serialized_order`,
  `serialized_pooling_{i-1}_serialized_inverse`, and
  `serialized_pooling_{i-1}_grid_coord` (stage 0 instead takes the base
  `serialized_code` and `grid_coord`). The rule is implemented once in
  `seg_head_export_input_names` and must be mirrored by deployment consumers
  from the artifact's `dec_depths`.
- `det3d_head` - consumes `point_feat_{S-2}`, `point_feat_{S-1}`,
  `pooling_cluster_{S-2}`, and `point_grid_coord_{S-2}` and outputs the
  detection head tensors.

## Implementation

| Path                                                  | Description                                    |
| ----------------------------------------------------- | ---------------------------------------------- |
| `autoware_ml/models/segmentation3d/ptv3.py`           | PTv3 Lightning model wrapper                   |
| `autoware_ml/models/segmentation3d/heads/ptv3.py`     | PTv3 segmentation decoder head                 |
| `autoware_ml/models/detection3d/ptv3.py`              | PTv3 BEV neck and detection model wrapper      |
| `autoware_ml/models/detection3d/heads/transfusion.py` | TransFusion detection head                     |
| `autoware_ml/models/segmentation3d/encoders/ptv3.py`  | Reusable PTv3 encoder components               |
| `autoware_ml/utils/point_cloud/`                      | Shared point-cloud utilities and serialization |
| `autoware_ml/ops/segment/segment_csr.py`              | Segment reduction export operator              |
| `autoware_ml/losses/segmentation3d/`                  | Segmentation losses used by PTv3               |
| `autoware_ml/datamodule/nuscenes/segmentation3d.py`   | NuScenes datamodule                            |
| `autoware_ml/datamodule/t4dataset/segmentation3d.py`  | T4Dataset datamodule                           |
| `autoware_ml/datamodule/t4dataset/detection3d.py`     | T4Dataset 3D detection datamodule              |
| `autoware_ml/transforms/point_cloud/`                 | Shared point-cloud transforms used by PTv3     |
| `autoware_ml/configs/tasks/segmentation3d/ptv3/`      | Task configurations                            |
| `autoware_ml/configs/tasks/detection3d/ptv3/`         | Detection task configurations                  |

## Acknowledgment

The Autoware-ML PointTransformerV3 implementation was ported from the official PointTransformerV3
project by Pointcept.

<!-- cspell:ignore Xiaoyang -->
- Repository: <https://github.com/Pointcept/PointTransformerV3>
- License: MIT
- Paper: Wu, Xiaoyang, et al. "Point Transformer V3: Simpler, Faster, Stronger." CVPR, 2024.
