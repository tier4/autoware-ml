---
icon: lucide/feather
---

# LitePT

LitePT is a PTv3-compatible model that reduces latency by turning off early
attention and late convolution blocks. It is fully compatible with PTv3's
export contract, task models and heads.

It is implemented as `LitePTEncoder`, a subclass of `PointTransformerV3Encoder`:
the gating and the rotary embedding are options on the shared components, and
PTv3's defaults leave both off. Everything not listed below is identical to
[PointTransformerV3](ptv3.md).

## Summary

| Property     | Value                                                           |
| ------------ | --------------------------------------------------------------- |
| Task         | 3D semantic segmentation, 3D object detection                   |
| Modality     | LiDAR                                                           |
| Input        | Point cloud                                                     |
| Output       | Point-wise semantic labels or 3D boxes/scores/classes           |
| Architecture | PTv3 hierarchy with per-stage conv/attention gating and 3D RoPE |
| Datasets     | T4Dataset                                                       |

## What differs from PTv3

| Aspect              | PTv3                                | LitePT                                              |
| ------------------- | ----------------------------------- | --------------------------------------------------- |
| Block composition   | Every block: conv + attention + MLP | Stages 0-2 conv only, stages 3-4 attention only     |
| Positional encoding | Submanifold conv per block          | Conv early; axis-split 3D RoPE on `grid_coord` late |
| Decoder             | `dec_depths` blocks per stage       | `dec_depths` all zero - unpooling only              |
| Encoder blocks      | 14 conv + 14 attention              | 6 conv + 8 attention                                |
| Serialization       | Every stage sorts for attention     | Only the two coarsest stages read an order          |

## Rotary position embedding

`Point3DRoPE` splits the head dimension into three equal per-axis chunks and
rotates each with a shared frequency ladder over integer `grid_coord` positions.
Each chunk pairs dimension `i` with `i + chunk // 2` as the two components one
shared `(cos, sin)` rotates together, so the rotated span must be a multiple of
six: `rope_span` returns the largest one that fits and any trailing dimensions
pass through unrotated. At `head_dim = 32` that is 30 rotated dimensions and a
2-dimension NoPE tail.

The default `enc_rope_base` of 100.0 mirrors what LitePT's reference CUDA
operator calls `rope_freq`. This is a hyperparameter and needs tuning.

## Available Configurations

| Config Name                                            | Task           | Dataset   | Range | Purpose                |
| ------------------------------------------------------ | -------------- | --------- | ----- | ---------------------- |
| `segmentation3d/litept/voxel012_122m_t4dataset_j6gen2` | segmentation3d | T4Dataset | 122 m | T4Dataset segmentation |

The config inherits dataset, transforms, optimizer and deploy settings from the
PTv3 config of the same name and swaps only the encoder and the decoder head.

```bash
autoware-ml train --config-name segmentation3d/litept/voxel012_122m_t4dataset_j6gen2
```

## Export contract

Declared identically to [PTv3](ptv3.md#onnx-preprocessing-contract), but the
gated graph reads fewer of the inputs - nothing consumes the base serialization
order, nor the `head_indices` of the stages that carry no convolution - and
`torch.onnx.export` drops whatever the traced graph never consumes. A LitePT
artifact therefore exposes a subset of the PTv3 inputs.

The deployed node binds the inputs the artifact declares rather than the full
PTv3 list, so a subset is a drop-in engine. Narrowing the declaration to match
is future work.

## Implementation

| Path                                                 | Description                                   |
| ---------------------------------------------------- | --------------------------------------------- |
| `autoware_ml/models/segmentation3d/encoders/ptv3.py` | `LitePTEncoder`, `Point3DRoPE`, block gating  |
| `autoware_ml/models/segmentation3d/ptv3_base.py`     | `PTv3EncoderExportBase`, shared export inputs |
| `autoware_ml/models/segmentation3d/heads/ptv3.py`    | Decoder gating (`dec_conv`, `dec_attn`)       |
| `autoware_ml/configs/tasks/segmentation3d/litept/`   | Task configurations                           |
| `autoware_ml/tests/models/test_litept.py`            | Rotary, gating, and export tests              |

## Acknowledgment

LitePT follows the published topology of the LitePT point-transformer variant,
reimplemented on top of the Autoware-ML PTv3 components. The reference
implementation's kernel-5 submanifold stem is not adopted: it exports poorly
and showed no accuracy advantage over the linear stem.
