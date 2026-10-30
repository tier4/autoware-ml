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
| Datasets     | NuScenes, T4Dataset                         |

## Available Configurations

Experiment configs live under `autoware_ml/configs/tasks/detection3d/streampetr/<dataset>/` and are passed to the CLI as the path relative to `tasks/`. The three leaves are the three stages of the production training flow.

| Leaf                                                                | Database                   | Weights                                               |
| ------------------------------------------------------------------- | -------------------------- | ----------------------------------------------------- |
| `nuscenes/vov99_cpfpn_320x800_b8_30e_nuscenes_51m_pretrain`         | nuScenes                   | RGB-stem DD3D/FCOS3D VoVNet-99 given with `--weights` |
| `t4dataset/vov99_cpfpn_480x640_b8_35e_t4dataset_51m_gen1_gen2_base` | T4Dataset gen1 + gen2 base | Fine-tuned from the stage-1 checkpoint (`--weights`)  |
| `t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base`   | T4Dataset j6gen2           | Fine-tuned from the stage-2 checkpoint (`--weights`)  |

The leaf name spells out its variant: the backbone and neck, the image size, the batch size, the epoch count and the detection range. The configs are layered so that each file owns one concern:

| File                          | Owns                                                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `base.yaml`                   | Model architecture and head recipe, image normalization and grid mask, optimizer, LR schedule, trainer settings, ONNX modules |
| `focal2d_head.yaml`           | The auxiliary 2D head block (`model.img_roi_head`), composed by the leaves that train it                                      |
| `<dataset>/base_<range>.yaml` | Detection range, image size, camera order, metrics, partial annotations, transforms and the streaming datamodule              |
| `<dataset>/<leaf>.yaml`       | Database selection and taxonomy, batch size, experiment name, epochs, learning rates and the pretrained weights               |

## Training

The production model is trained in three stages, each one initialized from the `best.ckpt` of the previous one:

| Stage | Leaf                                                                | Database                                      | Epochs | Total batch / LR | Init from                      |
| ----- | ------------------------------------------------------------------- | --------------------------------------------- | ------ | ---------------- | ------------------------------ |
| 1     | `nuscenes/vov99_cpfpn_320x800_b8_30e_nuscenes_51m_pretrain`         | `nuscenes_base`                               | 30     | 32 / 8e-4        | RGB-stem DD3D/FCOS3D VoVNet-99 |
| 2     | `t4dataset/vov99_cpfpn_480x640_b8_35e_t4dataset_51m_gen1_gen2_base` | `t4dataset_gen1_base` + `t4dataset_gen2_base` | 35     | 32 / 2e-4        | stage-1 `best.ckpt`            |
| 3     | `t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base`   | `t4dataset_j6gen2_base`                       | 35     | 64 / 4e-4        | stage-2 `best.ckpt`            |

Every leaf pins the recipe of its accepted run: the GPU count, the per-GPU batch and the learning rates. There is no `auto_scale_lr`, so for another total batch size rescale both learning rates linearly (the header of every leaf states its base). The `img_backbone` learning rate is always a tenth of the base one, set via `model.optimizer_group_overrides.img_backbone.lr`, and the LR schedule is bound to `trainer.max_epochs`. Validation runs every 5 epochs and then every epoch over the last 10 (`callbacks.dynamic_validation.epoch_intervals`); `best.ckpt` is selected by `val/det3d/mAP`.

Every dataloader streams the frames of its scenes in the order of their sample time: the range base composes `/datamodule/streaming@datamodule: default_streaming_datamodule` after the database datamodule, which makes it a `StreamingDataModule`. Its sampler joins the scenes and cuts their frames into one lane per batch position of every rank, and a lane cut inside a scene starts a new stream there. A shorter lane ends with a padding slot that the model leaves out of the loss and the metrics, so every split serves every frame exactly once on any number of GPUs. The training split reshuffles the order of the scenes every epoch from the experiment `seed`.

During training the auxiliary 2D head (`FocalHead2D`, `focal2d_head.yaml`) supervises the image features with the 2D boxes that `LoadAnnotations2DFromBoxes3D` projects from the augmented 3D boxes. The projection runs before the range filter, so the 2D targets keep the objects outside the 3D range, as those of the reference do. The T4 stages also project before they drop the boxes recorded with fewer than two lidar points, so the 2D targets keep those boxes too. Evaluation and the exported graphs never run it.

`--weights` accepts the Lightning checkpoint of the previous stage directly, and stage 2 -> 3 needs nothing else. Every stage trains the same 7 classes, but stage 1 -> 2 crosses taxonomies whose last two classes are swapped: the nuScenes taxonomy lists `car, truck, bus, bicycle, pedestrian, traffic_cone, barrier`, and the T4 stages select the `default_dynamic` taxonomy of the database (the one the other T4 detectors train), which lists `barrier, traffic_cone`. `--weights` loads by shape, so a stage-1 checkpoint loads into stage 2 without a warning, with the rows of class 5 and 6 of the `bbox_head.cls_branches.*` final Linear layers (one per decoder layer) and of the `img_roi_head.cls` conv swapped: the decoder would start stage 2 scoring barriers as traffic cones. Permute those rows of the stage-1 checkpoint into the T4 order first (no other tensor depends on the class order):

```python
import torch

NUSCENES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
T4 = ["car", "truck", "bus", "bicycle", "pedestrian", "barrier", "traffic_cone"]
rows = [NUSCENES.index(name) for name in T4]  # [0, 1, 2, 3, 4, 6, 5]

checkpoint = torch.load("stage1_best.ckpt", map_location="cpu", weights_only=False)
state = checkpoint["state_dict"]
for key, value in list(state.items()):
    is_decoder_cls = key.startswith("bbox_head.cls_branches.") and value.shape[0] == len(T4)
    if is_decoder_cls or key.startswith("img_roi_head.cls."):
        state[key] = value[rows].clone()
torch.save(checkpoint, "stage1_best_t4_order.ckpt")
```

`autoware_ml/models/detection3d/tests/test_streampetr_class_permutation.py` checks this permutation: the per-class logits of the permuted model are those of the source model with the columns reordered. The model package of a T4-trained graph lists the classes in the T4 order too (see Deployment).

The T4 stages train `traffic_cone` and `barrier` with partial annotations (`model.partial_annotations`, as the other T4 detectors): a frame whose scenario is flagged as annotated without them does not supervise those two classes in either head, so the detector is not taught they are absent. The flag comes from the scenario lists through the record table (`traffic_cone_barrier_bbox_status`) and the batch (`gt_traffic_cone_barrier_bbox_status`); the model refuses a batch without it, so every scenario of the T4 stages has to carry the status. The nuScenes stage supervises every class (`partial_annotations: null`). Stage 2 reads the gen1 and the gen2 base corpora as two dataset sources of one datamodule, both under the same taxonomy.

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

### Stage 2 — T4 gen1 + gen2 base

```bash
autoware-ml train \
    --config-name detection3d/streampetr/t4dataset/vov99_cpfpn_480x640_b8_35e_t4dataset_51m_gen1_gen2_base \
    --weights stage1_best_t4_order.ckpt
```

`stage1_best_t4_order.ckpt` is the stage-1 `best.ckpt` with its classification rows permuted into the T4 class order (see the snippet above). Cross-stage checkpoint loads must report `Loaded matching weight tensors: 894/894`: the nuScenes and T4 stages share the same 7-class head shapes, so nothing is dropped.

### Stage 3 — j6gen2 fine-tune

```bash
autoware-ml train \
    --config-name detection3d/streampetr/t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base \
    --weights <stage-2 best.ckpt>
```

Checkpoints are written to `mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/` as `best.ckpt` and `last.ckpt`. For a pipeline validation run add `+trainer.fast_dev_run=true`. For a short real run instead (for example `trainer.max_epochs=1`) also set `callbacks.dynamic_validation.epoch_intervals=[[0,1]]`: validation defaults to every 5 epochs, and a run that finishes without ever validating fails at teardown.

Resuming an interrupted run: `--resume-checkpoint <last.ckpt>` (mutually exclusive with `--weights`; repeat every other CLI override, they are not restored from the checkpoint; add `--new-run` only to fork a new MLflow run).

## Evaluate

```bash
autoware-ml test \
    --config-name detection3d/streampetr/t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base \
    --weights <stage-3 best.ckpt>
```

The headline number is the `0-121m` bucket mAP, directly comparable to AWML T4MetricV2's `mAP_center_distance_bev` in the `bev_center_0.0-121.0` evaluator:

1. **`gt_num_points` reaches the metric**, so the evaluation drops the boxes with fewer than `min_num_points` lidar points as T4MetricV2 does; the training pipelines drop them from the 3D targets with `BBoxesMinRecordedPointsFilter`. The filter runs after the 2D projection, so these boxes stay targets of the auxiliary 2D head.
2. **No radial `eval_class_range` cap inside the pc_range square.** The GT is already limited to the ±51.2 m square by the pipeline `BBoxesRangeFilter`. A radial cap at 51.2 m would remove the corners of the square from the GT while the corner predictions stayed and became false positives, which T4MetricV2 (radial on both GT and predictions) cannot reproduce. The `default_dynamic` taxonomy therefore evaluates every class up to 121 m, a no-op inside the square, and the T4 distance buckets mirror those of T4MetricV2 (0-50 / 50-90 / 90-121 / 0-121 m).

Validation during training and `autoware-ml test` stream one lane per rank, so every frame is scored exactly once on any number of GPUs.

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/streampetr/t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base \
    --weights <stage-3 best.ckpt> \
    --release v1.0.0
```

The export traces the three graphs the runtime node binds (`extract_img_feat`, `position_embedding`, `pts_head_memory`) from the first sample of the predict split. `extract_img_feat` takes images normalized with the ImageNet statistics: the node normalizes the camera images itself, so the `BatchNormalizeImages` layer of the training preprocessing is not part of the graph. TensorRT engine generation has not been validated yet.

`pts_head_memory` emits the predictions of the last decoder layer only: `all_cls_scores` is `(1, 7, 900)` and `all_bbox_preds` is `(1, 10, 900)`, one column per query (644 learned and 256 propagated). The model package that ships a graph of this export to the autoware.universe node must therefore set `num_proposals: 900`, and list `model_params.class_names` in the order of the trained taxonomy: `BARRIER` before `TRAFFIC_CONE` for the T4 stages. A stream starts from an all-zero memory bank, so the node must clear the memory tensors when it allocates and resets them (autoware.universe #13267), and it should bind the motion since the previous frame as the ego pose (autoware.universe #13549).

## Results

The three stages are being retrained on this code; their results will be added here once measured.

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
| `autoware_ml/transforms/camera/permutation.py`                       | `ShuffleCameraOrder`, T4 training only            |
| `autoware_ml/transforms/camera/resize.py`                            | `ResizeCropFlipRotImage`, the image augmentation  |
| `autoware_ml/transforms/camera_lidar/geometry.py`                    | `GlobalRotScaleTrans`, camera-only here           |
| `autoware_ml/preprocessing/camera/normalize.py`                      | Batched image normalization                       |
| `autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py`           | Warmup + epoch-cosine LR schedule                 |
| `autoware_ml/configs/tasks/detection3d/streampetr/`                  | Task configurations                               |
| `autoware_ml/configs/database/t4dataset/taxonomy/`                   | `default_dynamic` taxonomy of the T4 stages       |

## Acknowledgment

<!-- cspell:ignore exiawsh -->
The Autoware-ML StreamPETR implementation was ported from the official streampetr
project by exiawsh.

<!-- cspell:ignore Shihao -->
- Repository: <https://github.com/exiawsh/streampetr>
- License: Apache License 2.0
- Paper: Wang, Shihao, et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection" ICCV, 2023.
