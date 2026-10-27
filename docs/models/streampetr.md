---
icon: lucide/cctv
---

# StreamPETR

<!-- cspell:ignore fcos3d imgbackbone -->

StreamPETR is a camera-based 3D object detection model integrated under the `detection3d` task namespace. It uses a multiview image backbone, a feature pyramid neck, and a native query-based detection head with a streaming memory queue for temporal modeling.

## Summary

| Property     | Value                                       |
| ------------ | ------------------------------------------- |
| Task         | 3D object detection                         |
| Modality     | Camera                                      |
| Input        | Synchronized multiview images               |
| Output       | 3D bounding boxes and class scores          |
| Architecture | Multiview VoVNet/CPFPN + query decoder head |
| Datasets     | NuScenes                                    |

## Available Configurations

Experiment configs live under `autoware_ml/configs/tasks/detection3d/streampetr/<dataset>/` and are passed to the CLI as the path relative to `tasks/`.

| Leaf                                                        | Database | Weights                                               |
| ----------------------------------------------------------- | -------- | ----------------------------------------------------- |
| `nuscenes/vov99_cpfpn_320x800_b8_30e_nuscenes_51m_pretrain` | nuScenes | RGB-stem DD3D/FCOS3D VoVNet-99 given with `--weights` |

The leaf name spells out its variant: the backbone and neck, the image size, the batch size, the epoch count and the detection range. The configs are layered so that each file owns one concern:

| File                          | Owns                                                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `base.yaml`                   | Model architecture and head recipe, image normalization and grid mask, optimizer, LR schedule, trainer settings, ONNX modules |
| `focal2d_head.yaml`           | The auxiliary 2D head block (`model.img_roi_head`), composed by the leaves that train it                                      |
| `<dataset>/base_<range>.yaml` | Detection range, image size, camera order, metrics, partial annotations, transforms and the streaming datamodule              |
| `<dataset>/<leaf>.yaml`       | Database selection and taxonomy, batch size, experiment name, epochs, learning rates and the pretrained weights               |

## Training

Every leaf pins the recipe of its accepted run: the GPU count, the per-GPU batch and the learning rates. There is no `auto_scale_lr`, so for another total batch size rescale both learning rates linearly (the header of every leaf states its base). The `img_backbone` learning rate is always a tenth of the base one, set via `model.optimizer_group_overrides.img_backbone.lr`, and the LR schedule is bound to `trainer.max_epochs`. Validation runs every 5 epochs and then every epoch over the last 10 (`callbacks.dynamic_validation.epoch_intervals`); `best.ckpt` is selected by `val/det3d/mAP`.

Every dataloader streams the frames of its scenes in the order of their sample time: the range base composes `/datamodule/streaming@datamodule: default_streaming_datamodule` after the database datamodule, which makes it a `StreamingDataModule`. Its sampler joins the scenes and cuts their frames into one lane per batch position of every rank, and a lane cut inside a scene starts a new stream there. A shorter lane ends with a padding slot that the model leaves out of the loss and the metrics, so every split serves every frame exactly once on any number of GPUs. The training split reshuffles the order of the scenes every epoch from the experiment `seed`.

During training the auxiliary 2D head (`FocalHead2D`, `focal2d_head.yaml`) supervises the image features with the 2D boxes that `LoadAnnotations2DFromBoxes3D` projects from the augmented 3D boxes. The projection runs before the range filter, so the 2D targets keep the objects outside the 3D range, as those of the reference do. Evaluation and the exported graphs never run it.

### Stage 1 — nuScenes pretrain

```bash
autoware-ml train \
    --config-name detection3d/streampetr/nuscenes/vov99_cpfpn_320x800_b8_30e_nuscenes_51m_pretrain \
    --weights <rgb_stem_vovnet_backbone.pth>
```

Stage 1 trains at the ±51.2 m range of the reference recipe and of the T4 stages: the head normalizes its learned reference points by the range, so they carry over to stage 2.

The backbone init is the upstream StreamPETR DD3D/FCOS3D-pretrained VoVNet-99 (`fcos3d_vovnet_imgbackbone-remapped.pth`, BGR); without it the recipe underperforms. The detector keeps its backbone inside its image feature extractor, so the file's `img_backbone.*` tensors are renamed to `image_feature_extractor.img_backbone.*` (under the original names no tensor matches and `--weights` fails) and the stem conv is flipped to RGB:

```python
import torch

state = torch.load("fcos3d_vovnet_imgbackbone-remapped.pth", map_location="cpu")
state = state.get("state_dict", state)
backbone = {
    f"image_feature_extractor.{key}": value
    for key, value in state.items()
    if key.startswith("img_backbone.")
}
stem = "image_feature_extractor.img_backbone.stem.stem_1/conv.weight"
backbone[stem] = backbone[stem][:, [2, 1, 0]].contiguous()  # BGR -> RGB
torch.save({"state_dict": backbone}, "rgb_stem_vovnet_backbone.pth")
```

The training log should report `Loaded matching weight tensors: 626/894` for this init. Watch the first few hundred iterations: at lr 8e-4 with grad-clip 35 a loss spike or NaN means both learning rates should back off (for example to 6e-4 / 6e-5).

Checkpoints are written to `mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/` as `best.ckpt` and `last.ckpt`. For a pipeline validation run add `+trainer.fast_dev_run=true`. For a short real run instead (for example `trainer.max_epochs=1`) also set `callbacks.dynamic_validation.epoch_intervals=[[0,1]]`: validation defaults to every 5 epochs, and a run that finishes without ever validating fails at teardown.

Resuming an interrupted run: `--resume-checkpoint <last.ckpt>` (mutually exclusive with `--weights`; repeat every other CLI override, they are not restored from the checkpoint; add `--new-run` only to fork a new MLflow run).

## Implementation

| Path                                                                 | Description                                       |
| -------------------------------------------------------------------- | ------------------------------------------------- |
| `autoware_ml/models/detection3d/streampetr.py`                       | StreamPETR model wrapper                          |
| `autoware_ml/models/detection3d/heads/streampetr.py`                 | Query-based detection head                        |
| `autoware_ml/models/detection3d/heads/focal2d.py`                    | Auxiliary 2D head (FocalHead2D)                   |
| `autoware_ml/models/detection3d/task_modules/partial_annotations.py` | Annotated class masks of the partial annotations  |
| `autoware_ml/models/common/backbones/vovnet.py`                      | Multiview image backbone                          |
| `autoware_ml/models/common/multiview_image_encoder.py`               | Per-camera image encoder shared with BEVFusion    |
| `autoware_ml/models/common/necks/cp_fpn.py`                          | CPFPN neck (reference StreamPETR neck)            |
| `autoware_ml/models/detection3d/task_modules/`                       | Shared assigners, costs, coders, streaming memory |
| `autoware_ml/datamodule/streaming_data_module.py`                    | `StreamingDataModule`, the streaming datamodule   |
| `autoware_ml/datamodule/samplers.py`                                 | `GroupStreamingSampler`, the scene lanes          |
| `autoware_ml/datamodule/base_dataset.py`                             | Scene grouping of the record table                |
| `autoware_ml/dataclasses/batch/frame_meta.py`                        | Per-frame pose, scene and sample time             |
| `autoware_ml/dataclasses/batch/detection2d.py`                       | 2D boxes padded to `max_num_2d_gt_bboxes`         |
| `autoware_ml/transforms/camera/annotations2d.py`                     | 2D targets projected from the 3D boxes            |
| `autoware_ml/transforms/camera/resize.py`                            | `ResizeCropFlipRotImage`, the image augmentation  |
| `autoware_ml/transforms/camera_lidar/geometry.py`                    | `GlobalRotScaleTrans`, camera-only here           |
| `autoware_ml/preprocessing/camera/normalize.py`                      | Batched image normalization                       |
| `autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py`           | Warmup + epoch-cosine LR schedule                 |
| `autoware_ml/configs/tasks/detection3d/streampetr/`                  | Task configurations                               |

## Acknowledgment

<!-- cspell:ignore exiawsh -->
The Autoware-ML StreamPETR implementation was ported from the official streampetr
project by exiawsh.

<!-- cspell:ignore Shihao -->
- Repository: <https://github.com/exiawsh/streampetr>
- License: Apache License 2.0
- Paper: Wang, Shihao, et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection" ICCV, 2023.
