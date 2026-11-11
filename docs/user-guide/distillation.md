---
icon: lucide/git-merge
---

# Knowledge Distillation

Train a student on the ground truth and on the outputs of a frozen, trained teacher at the same
time. The teacher runs on every training batch the student sees, augmentation included, so any
two models that read the same batch can be paired: a PTv3 teacher and a LitePT student, a large
and a small PTv3, or a joint det3d+seg3d teacher and a student of either task.

## Running it

A distillation is a config group an experiment composes through its defaults list. The group
holds the terms; the experiment stays the student's own and names the teacher checkpoint when
it runs. `multi/litept/nuscenes/distill_ptv3_voxel005_b16_100e_nuscenes_51m` distills the PTv3
nuScenes experiment of the same name into LitePT:

```yaml
# @package _global_
defaults:
  - /tasks/multi/ptv3/nuscenes/voxel005_b16_100e_nuscenes_51m
  - /tasks/segmentation3d/litept/encoder
  - /tasks/segmentation3d/litept/seg3d_head
  - /distillation: ptv3_det3d_seg3d
  - _self_

experiment_group_name: litept_det3d_seg3d/${database.version}/nuscenes
experiment_name: distill_ptv3_voxel005_b16_100e_nuscenes_51m
```

```bash
autoware-ml train --config-name multi/litept/nuscenes/distill_ptv3_voxel005_b16_100e_nuscenes_51m \
    teacher_weights=[mlruns/<group>/<experiment>/<run_id>/artifacts/checkpoints/best.ckpt]
```

A `+distillation=<group>` override on the command line does not reach the group: Hydra resolves
such an override relative to the directory of the experiment config. The bundled groups live
in `autoware_ml/configs/distillation/`:

| Group              | Student     | Terms                                                                        |
| ------------------ | ----------- | ---------------------------------------------------------------------------- |
| `ptv3_det3d_seg3d` | det3d+seg3d | segmentation logits, dense detection heatmap, shared-query detection outputs |
| `ptv3_seg3d`       | seg3d       | segmentation logits, deepest encoder features                                |

The teacher is rebuilt from the `config/resolved.yaml` every training run writes next to its
`checkpoints` directory, so a checkpoint path is all it needs. Several `teacher_weights` stack
the way `--weights` does, later checkpoints overriding earlier ones; every teacher parameter
must be covered. Left unset, the experiment still composes but the run fails at fit setup.

The checkpoints of the student hold the student and the adapters of the terms, never the
teacher. `test` and `deploy` therefore run with the experiment config alone, without the group.

## What a term compares

Every term names a student tensor, a teacher tensor, a loss and a weight:

```yaml
model:
  distillation:
    _target_: autoware_ml.distillation.distillation.Distillation
    stages: [train]
    teacher:
      _target_: autoware_ml.distillation.teacher.Teacher
      weights: ${teacher_weights}
      autocast_dtype: bfloat16
      share_student_inputs: true
    terms:
      - _target_: autoware_ml.distillation.term.DistillationTerm
        name: seg3d_logits
        student: segmentation3d_head_outputs.logits
        teacher: segmentation3d_head_outputs.logits
        weight: 1.0
        loss:
          _target_: autoware_ml.losses.distillation.soft_target.SoftTargetKLDivLoss
          temperature: 2.0
      - _target_: autoware_ml.distillation.term.DistillationTerm
        name: bev_features
        student: module:bev_neck
        teacher: module:bev_neck
        weight: 0.1
        adapter:
          _target_: torch.nn.Conv2d
          in_channels: 96
          out_channels: 128
          kernel_size: 1
        loss:
          _target_: autoware_ml.losses.distillation.feature.FeatureMSELoss
          normalize: true
```

A tap is either an attribute path into the `ModelOutputs` the model returns, such as
`segmentation3d_head_outputs.logits` or
`detection3d_head_outputs.transfusion_head_outputs.dense_heatmap`, or `module:<path>` naming a
submodule whose forward output a hook captures, such as `module:bev_neck` or
`module:encoder.enc.enc3`. A module that returns a PTv3 `Point` yields its `feat`; `@<key>`
picks another entry and `@<index>` an element of a tuple output.

Both tensors must have the same shape. A student whose channels differ from the teacher's
gets an `adapter`, any module, applied to the student tensor; it trains with the student and is
saved in its checkpoint. Position-wise tensors align because the teacher reads the student's
voxels (`share_student_inputs`).

Per-query detection outputs do not align on their own, each head picks its own proposals, so
the detection group adds a term of its own kind, `TransFusionSharedQueriesTerm`: it runs the
student's head once more at the proposals the teacher picked, on the student's own BEV
features, and matches the paired queries, class logits with a soft binary cross entropy and
box regressions with an L1 weighted by the teacher's confidence in each query. It is logged in
two parts, `distill/det3d_queries/cls` and `distill/det3d_queries/reg`. The dense heatmap term
stays next to it: at inference the student still picks its own proposals, and the heatmap is
what makes it look where the teacher looks. A term is any module with a name, a weight and the
two taps that returns a scalar or a mapping of parts; one that implements `bind(student)` is
handed the student at fit setup.

| Loss                                                            | For                                      | Notes                                                                          |
| --------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------ |
| `losses.distillation.soft_target.SoftTargetKLDivLoss`           | class logits, `(N, C)` or `(B, C, H, W)` | `temperature` softens both distributions, divergence scaled by its square      |
| `losses.distillation.sigmoid_soft_target.SigmoidSoftTargetLoss` | sigmoid maps such as heatmaps            | normalized by the teacher's probability mass, `focal_gamma` focuses on the gap |
| `losses.distillation.feature.FeatureMSELoss`                    | features of the same shape               | `normalize` matches directions rather than magnitudes                          |

Any module called as `loss(student, teacher)` and returning a scalar works as a loss.

## What is logged

On the stages the distillation runs at, by default training only:

| Key                    | Value                                           |
| ---------------------- | ----------------------------------------------- |
| `train/gt_loss`        | the student's own loss against the ground truth |
| `train/distill/<name>` | each term, unweighted                           |
| `train/distill_loss`   | the weighted sum of the terms                   |
| `train/loss`           | `gt_loss + distill_loss`, what trains           |

`val/loss` stays the ground-truth loss, so the checkpoint and early-stopping callbacks keep
measuring the student alone. Add `val` to `stages` to log the terms there too; `val/loss` then
includes them.

## The teacher at runtime

The teacher is built at fit setup and is not a submodule of the student: it stays out of the
state dict, the optimizer, the distributed wrapping and the model summary, and never leaves eval
mode. Its forward runs without gradients under `autocast_dtype`, bf16 by default. Frames that
carry no ground truth for a task, such as detection frames without boxes, still get the
teacher's supervision.

The adapters are the only trainable parameters the distillation adds. A model that groups its
parameters structurally in `build_optimizer_groups()` gets them in a `distillation` group, which
`optimizer_group_overrides` can address like any other.
