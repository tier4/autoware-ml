---
icon: lucide/cctv
---

# StreamPETR

<!-- cspell:ignore CPFPN kokseang fcos3d imgbackbone -->

StreamPETR is a camera-based 3D object detection model integrated under the `detection3d` task namespace. It uses a multiview image backbone, a feature pyramid neck, and a native query-based detection head with a streaming memory queue for temporal modeling.

## Summary

| Property     | Value                                       |
|--------------|---------------------------------------------|
| Task         | 3D object detection                         |
| Modality     | Camera                                      |
| Input        | Synchronized multiview images               |
| Output       | 3D bounding boxes and class scores          |
| Architecture | Multiview VoVNet/CPFPN + query decoder head |
| Datasets     | NuScenes, T4Dataset                         |

## Available Configurations

| Config Name                                             | Dataset   | Purpose                                                   |
|----------------------------------------------------------|-----------|-----------------------------------------------------------|
| `detection3d/streampetr/vov_320x800_nuscenes`            | NuScenes  | Plain nuScenes reference configuration                    |
| `detection3d/streampetr/vov_320x800_nuscenes_pretrain`   | NuScenes  | Stage 1: nuScenes pretrain (CPFPN + auxiliary 2D head)    |
| `detection3d/streampetr/vov_480x640_t4dataset_base`      | T4Dataset | Stage 2: full T4 base DB                                  |
| `detection3d/streampetr/vov_480x640_t4dataset_j6gen2`    | T4Dataset | Stage 3: j6gen2 fine-tune — the production configuration  |

## Three-Stage Training Flow

The production model is trained natively in three stages; each stage's config
pins the exact recipe (GPU count, per-GPU batch, LRs) of the accepted run.
There is no `auto_scale_lr`: for a different total batch size rescale both
LRs linearly (each config's header states its base).

| Stage | Config                                                  | Data                                              | Epochs | Total batch / LR   | Init from                        |
|-------|----------------------------------------------------------|---------------------------------------------------|--------|--------------------|-----------------------------------|
| 1     | `detection3d/streampetr/vov_320x800_nuscenes_pretrain`   | nuScenes                                          | 30     | 32 / 8e-4          | RGB-stem DD3D/FCOS3D VoVNet-99    |
| 2     | `detection3d/streampetr/vov_480x640_t4dataset_base`      | `info/kokseang_2_8_1/t4dataset_base_infos_*`      | 35     | 32 / 2e-4          | stage-1 `best.ckpt`               |
| 3     | `detection3d/streampetr/vov_480x640_t4dataset_j6gen2`    | `info/kokseang_2_8_1/t4dataset_j6gen2_base_infos_*` | 35   | 64 / 4e-4          | stage-2 `best.ckpt`               |

`--weights` accepts the previous stage's Lightning checkpoint directly — no
conversion between stages. Both T4 stages train all 7 classes with
`traffic_cone`/`barrier` partial-ignore: frames whose scene lacks those
annotations do not punish their background predictions, annotated frames
train them normally. The 2.8.1 info generation carries both classes in the
base DB too (train: ~272k cone / ~36k barrier boxes on 60k annotated frames
of 151k), so stage 2 supervises them as well.

The `img_backbone` LR is always `lr * 0.1`, set via
`model.optimizer_group_overrides.img_backbone.lr`. Validation runs every 5
epochs; `best.ckpt` is selected by `val/det3d/mAP`.

### Stage 1 — nuScenes pretrain

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_320x800_nuscenes_pretrain \
    --weights <rgb_stem_vovnet_backbone.pth> \
    datamodule.data_root=<nuscenes_root>
```

The backbone init is the upstream StreamPETR DD3D/FCOS3D-pretrained
VoVNet-99 (`fcos3d_vovnet_imgbackbone-remapped.pth`, BGR) with its stem conv
flipped to RGB; without it the recipe underperforms. The training log should
report `Loaded matching weight tensors: 626/1526 (+626 shared-tensor
aliases)` for this init. Watch the first few hundred iterations: at lr 8e-4
with grad-clip 35 a loss spike/NaN means both LRs should back off (e.g. to
6e-4 / 6e-5).

Accepted run: val mAP **0.5031** at epoch 29 (7-class; the upstream 10-class
published result is 0.4697).

### Stage 2 — T4 base DB

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_base \
    --weights mlruns/detection3d/streampetr/vov_320x800_nuscenes_pretrain/<run_id>/artifacts/checkpoints/best.ckpt \
    datamodule.data_root=<t4_data_root>
```

Cross-stage checkpoint loads must report `Loaded matching weight tensors:
1526/1526 (+0 shared-tensor aliases)` — full coverage, nothing dropped (the
nuScenes and T4 stages share the same 7-class head).

Accepted run: val mAP **0.4420** at epoch 34. That run used the older
5-class `info/detection3d` base infos (no cone/barrier at all — a 35-epoch
supervision gap for those classes); the config now points at the 2.8.1
infos, which close that gap, so a re-run's numbers are not directly
comparable.

### Stage 3 — j6gen2 fine-tune

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/detection3d/streampetr/vov_480x640_t4dataset_base/<run_id>/artifacts/checkpoints/best.ckpt \
    datamodule.data_root=<t4_data_root>
```

For a pipeline validation run add `+trainer.fast_dev_run=true`. For a short
real run instead (e.g. `trainer.max_epochs=1`) also set
`trainer.check_val_every_n_epoch=1`: validation defaults to every 5 epochs,
and a run that finishes without ever validating fails at teardown. Full-length
training is unaffected.

Resuming an interrupted run: `--resume-checkpoint <last.ckpt>` (mutually
exclusive with `--weights`; repeat every other CLI override, they are not
restored from the checkpoint; add `--new-run` only to fork a new MLflow run).

### Evaluate

```bash
autoware-ml test \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/detection3d/streampetr/vov_480x640_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt \
    trainer.devices=1
```

The headline number is the `0-121m` bucket mAP, directly comparable to AWML
T4MetricV2's `mAP_center_distance_bev` in the `bev_center_0.0-121.0`
evaluator (see the alignment notes below).

### Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/.../checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

The current verification scope covers ONNX export. TensorRT engine
generation has not been validated yet.

## Results

Final accepted stage-3 run (batch 64 / lr 4e-4, trained on the 2.8 info
generation; an earlier batch-32 / lr 2e-4 run matched within ±0.5 pp on
every headline number). The three stage `best.ckpt` files are archived on
the training host under `~/ckpt_backup/`.

| Split | mAP (0-121m) | 0-50m  | 50-90m |
|-------|--------------|--------|--------|
| val   | 0.5160       | —      | —      |
| test  | **0.5025**   | 0.5336 | 0.1730 |

Test per-class mAP (0-121m): car 0.677, bus 0.651, truck 0.585, bicycle
0.456, pedestrian 0.464, traffic_cone 0.344, barrier 0.340. The 90-121m
bucket is 0 by construction: `point_cloud_range` is ±51.2 m and
`post_center_range` ±61.2 m, so no predictions exist beyond ~86 m.

The two weakest classes trace to the data, not the recipe: the accepted
run's stage-2 infos lacked cone/barrier entirely (a 35-epoch supervision
gap) and only part of the j6gen2 scenes carry them. The configs have since
moved to the 2.8.1 infos, whose base DB includes both classes with per-frame
annotation status — the expected fix for these two classes on the next
training round.

For reference, the earlier AWML-checkpoint-conversion route on the same split
scored val 0.3913 / test 0.3661, and AWML itself 0.3752 / 0.3552 — the
native three-stage flow exceeds both by a wide margin. Cross-checks backing
comparability: the metric stacks agree to ≤ 2.5e-8 per class on identical
predictions, and running one AWML checkpoint through both inference stacks
differs by −0.80 test mAP (camera-pipeline numerics, consistent in sign
across all 7 classes).

## Evaluation Alignment Notes

Two evaluation-side fixes were required to make mAP comparable with AWML
T4MetricV2; both are part of the default config:

1. **`gt_num_points` collation** — without it the evaluation-time min-points
   GT filter silently never engages.
2. **No radial `eval_class_range` cap inside the pc_range square** — the GT
   is already limited to the ±51.2 m *square* by the pipeline
   `ObjectRangeFilter`. An additional radial cap at 51.2/54 m removed the
   square's corners from the GT while corner *predictions* stayed and became
   guaranteed false positives — a semantics T4MetricV2 (radial on both GT
   and predictions) cannot reproduce. The config therefore restates the
   121 m dataset default (a no-op inside the square) and mirrors
   T4MetricV2's distance buckets (0-50 / 50-90 / 90-121 / 0-121 m).

## Implementation

| Path                                                        | Description                                          |
|-------------------------------------------------------------|------------------------------------------------------|
| `autoware_ml/models/detection3d/streampetr.py`              | StreamPETR model wrapper                             |
| `autoware_ml/models/detection3d/heads/streampetr.py`        | Query-based detection head                           |
| `autoware_ml/models/detection3d/heads/focal2d.py`           | Auxiliary 2D head (FocalHead2D)                      |
| `autoware_ml/models/detection3d/partial_ignore.py`          | Partial-ignore label handling                        |
| `autoware_ml/models/common/backbones/vovnet.py`             | Multiview image backbone                             |
| `autoware_ml/models/common/necks/cp_fpn.py`                 | CPFPN neck (reference StreamPETR neck)               |
| `autoware_ml/models/detection3d/task_modules/`              | Shared assigners, costs, coders, streaming memory    |
| `autoware_ml/datamodule/common/multiview_detection3d.py`    | Shared multiview detection dataset                   |
| `autoware_ml/datamodule/nuscenes/multiview_detection3d.py`  | NuScenes multiview datamodule                        |
| `autoware_ml/datamodule/t4dataset/multiview_detection3d.py` | T4Dataset multiview datamodule                       |
| `autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py`  | Warmup + epoch-cosine LR schedule                    |
| `autoware_ml/configs/tasks/detection3d/streampetr/`         | Task configurations                                  |

An AWML/mmdet3d checkpoint converter
(`autoware_ml/tools/convert_streampetr_checkpoint.py`) existed during the
migration and was removed once the native training flow replaced converted
checkpoints; recover it from git history if an mm-style checkpoint ever needs
importing again.

## Acknowledgment

<!-- cspell:ignore exiawsh -->
The Autoware-ML StreamPETR implementation was ported from the official streampetr
project by exiawsh.

<!-- cspell:ignore Shihao -->
- Repository: <https://github.com/exiawsh/streampetr>
- License: Apache License 2.0
- Paper: Wang, Shihao, et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection" ICCV, 2023.
