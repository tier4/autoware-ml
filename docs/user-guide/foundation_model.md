# Offline foundation model

The offline foundation model is a 6 cm PT-v3m3 network with a segmentation decoder and a
TransFusion detection head, trained to label data offline (pseudo labels for the deployed
models). It is never deployed, so it may read the frame after the current one and has no
latency budget. This page lists the training stages, the data they read and how to run them.

Everything specific to it lives in its own config namespace and is opt-in, so the PTv3
tasks are unchanged:

- `configs/tasks/foundation/ptv3/`: `common.yaml` (what the stages share) and one file per stage;
- `configs/database/t4dataset/foundation/` and `configs/database/t4dataset/scenarios/foundation/`:
  the three corpora as the foundation level reads them;
- `configs/database/t4dataset/taxonomy/**/foundation*.yaml`: the foundation level.

The generic features it uses (rulebook backend, bf16, preemption-safe checkpoints, label
sets, the future sweep window) are switched on by these configs and available to any task.

## Stages

| stage | config | data | init |
|---|---|---|---|
| 1 | `foundation/ptv3/stage1_voxel006_122m_t4dataset_j6gen2` | J6 gen2 semantic segmentation v1 (old specification, about 3.2k training frames) | scratch |
| 2 | `foundation/ptv3/stage2_voxel006_122m_t4dataset_pseudo_10hz` | 10 Hz pseudo label corpus (about 181k frames, 15 databases) and the stage 1 data rehearsed | stage 1 |
| 3 | `foundation/ptv3/stage3_voxel006_122m_t4dataset_kognic` | Kognic corpus of the new specification (about 4.4k training frames, repeated 4 times) and the pseudo corpus at 1 Hz | stage 2 |

Every stage is bound to the `foundation` taxonomy level
(`configs/database/t4dataset/taxonomy/foundation.yaml`): the 33 classes of the fine level,
`traffic_sign` and `traffic_light` appended (35 segmentation classes, 20 detection classes),
and label sets for the corpora that are annotated at a coarser grain. A point of a label set
trains the summed probability of the set's classes and is not scored. See
[Foundation level mapping](../databases/foundation_level_mapping.md) for what each corpus
resolves to; only the Kognic corpus supervises `vertical_thin`, the curbs,
`manmade.solid_stuff`, `noise.ghost_point`, `ego_vehicle` and the two sign classes one by one.

Shared by the three stages (`foundation/ptv3/common.yaml`):

- Input: the current frame, one past and one future sweep (the default pipelines read the
  future window from `t4dataset.lidar.{train,test}_future_sweeps`, null by default),
  described per voxel by the sweep-split voxel encoder (17 features).
- Model: PT-v3m3 depths with hardware-aligned widths (encoder
  64/128/256/448/576, heads of 32 channels), 171M parameters with the segmentation head, 180M
  with the detection branch.
- Precision: `bf16-mixed` (`/defaults/amp_rulebook`) with the feed-forward layers in fp32
  (`model.encoder.fp32_sublayers: [mlp]`). In an ablation on stage 1 (same frames, 12 x 100 batches)
  plain bf16 lost 10 % relative mIoU against fp32 and fp32 feed-forward layers recovered it
  (mIoU 0.225 vs 0.226) at 1.7 times the step rate of the fp32 spconv run. fp16 overflowed in the deep encoder
  stages during stage 2.
- Sparse convolutions: `rulebook_torch`, the spconv replacement developed by Max Schmeller
  (repository `spconv-replacement`), selected with `sparse_conv_backend: rulebook`. spconv
  cannot train under autocast. Its kernels exist for channel widths that are multiples of 32;
  `PaddedSubMConv3d` pads other widths, which the aligned widths avoid altogether.
- Activation checkpointing in every block, one frame per GPU, global batch 32 (eight GPUs,
  four accumulation steps).
- Preemption-safe checkpoints (`/defaults/preemption`, see
  [Preemptible training](preemption_resume.md)), `best.ckpt` by the grouped segmentation mIoU
  of the validation split.

## Data

| corpus | root (env var) | layout |
|---|---|---|
| J6 gen2 semantic segmentation v1 | `${data_root_path}/t4dataset` (`AUTOWARE_ML_DATA_PATH`) | `t4dataset/db_j6gen2_semaseg_v1`, perception-devops lists |
| 10 Hz pseudo labels | `AUTOWARE_ML_PSEUDO_10HZ_PATH`, default `${data_root_path}/pseudo_10hz` | `<db>/<scene>/<version>/{annotation,lidarseg,data}` with `data` linking to the source scene, `scenario_lists/<db>.yaml` |
| Kognic (new specification) | `AUTOWARE_ML_KOGNIC_FUSED_PATH`, default `${data_root_path}/t4dataset` | `db_j6gen2_semseg_kognic_fused_v1/<scene>/<version>`, `scenario_lists/` |

The pseudo corpus scenario lists are written by
`scripts/datasets/write_pseudo_10hz_scenario_lists.py` (one validation and one test scene in
40 by a hash of the scene id). The corpus labels every frame, and for a scene with human boxes
those boxes are its keyframe labels, so the validation and test scenes of every benchmark must
be held out of training:

```bash
python scripts/datasets/write_pseudo_10hz_scenario_lists.py $AUTOWARE_ML_PSEUDO_10HZ_PATH \
    --holdout external/perception-devops/dataset_annotation/t4dataset/detection3d \
              external/perception-devops/dataset_annotation/t4dataset/segmentation3d
```

Lists written without `--holdout` put 33 test scenes of det3d-j6gen2 into the training split. The Kognic point clouds are the export after the sensing
filter of the vehicle, at the density of every other J6 gen2 database; only labelled frames
are kept (`semantic_masks: true`).

### Compressed point clouds

A host short of storage may replace the loose frames of a scene,
`data/LIDAR_CONCAT/*.pcd.bin`, by one `data/LIDAR_CONCAT.pack` file (t4pack: every frame
zstd-compressed over byte-shuffled columns). The loader reads the
loose file when it exists and the frame from the pack otherwise
(`autoware_ml/utils/point_cloud/t4pack.py`), so record tables and configs are the same for both.

## Running

```bash
# Stage 1
autoware-ml train --config-name foundation/ptv3/stage1_voxel006_122m_t4dataset_j6gen2

# Stage 2, from the stage 1 checkpoint
autoware-ml train --config-name foundation/ptv3/stage2_voxel006_122m_t4dataset_pseudo_10hz \
    --weights <stage 1 run>/artifacts/checkpoints/best.ckpt

# Stage 3, from the last stage 2 checkpoint (fully annealed)
autoware-ml train --config-name foundation/ptv3/stage3_voxel006_122m_t4dataset_kognic \
    --weights <stage 2 run>/artifacts/checkpoints/last.ckpt
```

A relaunch with `--resume-latest <run>/artifacts/checkpoints` continues mid-epoch; see
[Preemptible training](preemption_resume.md) for the hooks a scheduler wrapper uses.

Measured on eight H100: stage 1 takes about 6 minutes per epoch (12 epochs), stage 2 about
6.4 hours per epoch (30 epochs, so the 7-day limit needs one resume). Stage 3 took 5.4 hours per
epoch on two RTX PRO 6000 (10 epochs).
