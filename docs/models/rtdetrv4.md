---
icon: lucide/camera
---

# RT-DETRv4

RT-DETRv4 is a camera-based 2D object detector built around an HGNetv2
backbone, a hybrid encoder, and a D-FINE-style transformer decoder. It is
integrated under the `detection2d` task namespace for COCO-style datasets,
with ready-made configs for MS COCO and the Mapillary Vistas simple-driving
subset.

## Summary

| Property     | Value                                                       |
| ------------ | ----------------------------------------------------------- |
| Task         | 2D object detection                                         |
| Modality     | Camera                                                      |
| Input        | RGB image                                                   |
| Output       | Bounding boxes, class labels, confidence scores             |
| Architecture | HGNetv2 backbone, HybridEncoder, D-FINE transformer decoder |
| Datasets     | MS COCO and Mapillary Vistas simple-driving subset         |

## Available Configurations

| Config Name                                                       | Dataset             | Purpose                                    |
| ----------------------------------------------------------------- | ------------------- | ------------------------------------------ |
| `detection2d/rtdetrv4/hgnetv2_s`                                  | MS COCO             | Small HGNetv2 RT-DETRv4 baseline           |
| `detection2d/rtdetrv4/hgnetv2_m`                                  | MS COCO             | Medium HGNetv2 RT-DETRv4 baseline          |
| `detection2d/rtdetrv4/hgnetv2_l`                                  | MS COCO             | Large HGNetv2 RT-DETRv4 baseline           |
| `detection2d/rtdetrv4/hgnetv2_x`                                  | MS COCO             | Extra-large HGNetv2 RT-DETRv4 baseline     |
| `detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_simple_driving`  | Mapillary Vistas    | Ten-class simple-driving training preset   |
| `detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer`   | Mapillary Vistas    | COCO checkpoint fine-tuning preset         |

## Training

```bash
autoware-ml train --config-name detection2d/rtdetrv4/hgnetv2_s
autoware-ml train --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_simple_driving
```

To fine-tune the Mapillary preset from an upstream RT-DETRv4-S COCO
checkpoint:

```bash
autoware-ml train \
    --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
    model.init_checkpoint_path=/workspace/weights/rtv4_hgnetv2_s_coco.pth
```

The transfer preset is configured for raw upstream checkpoints that store model
weights under `ema.module`. It also filters mismatched shapes so the detector
head can be reinitialized safely when the dataset class count changes.

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection2d/rtdetrv4/hgnetv2_s \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
    +checkpoint=mlruns/detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer/<date>/<time>/checkpoints/best.ckpt
```

Validation and test runs log standard COCO bbox metrics such as `mAP`, `AP50`,
and `AP75`. COCO evaluation uses `pycocotools` and is executed in a spawned
subprocess so repeated evaluation does not destabilize the long-running trainer
process.

## Deployment

```bash
autoware-ml deploy \
    --config-name detection2d/rtdetrv4/hgnetv2_s \
    +checkpoint=mlruns/detection2d/rtdetrv4/hgnetv2_s/<date>/<time>/checkpoints/best.ckpt
```

The export wrapper emits postprocessed `pred_labels`, `pred_boxes`, and
`pred_scores`. Deployment clones the detector and postprocessor before applying
deploy-only conversions, so export does not mutate the training model state.

## Data Pipeline

RT-DETRv4 uses the shared COCO-style detection datamodule. Training samples are
loaded from standard COCO annotations and pass through image decode, bounding
box conversion, and strong detection augmentations including mosaic,
photometric distortion, zoom-out, random IoU crop, horizontal flip, and final
resize to the configured square input size.

Validation, test, and prediction use the deterministic resize path without the
strong train-time augmentations. The datamodule also supports split-specific
sample selection, max-sample limits, and scheduled multiscale and mixup
behavior for larger training runs.

## Implementation

| Path                                                     | Description                                      |
| -------------------------------------------------------- | ------------------------------------------------ |
| `autoware_ml/models/detection2d/rtdetrv4/model.py`      | Lightning wrapper and export contract            |
| `autoware_ml/models/detection2d/rtdetrv4/backbone/`     | HGNetv2 backbone and shared backbone utilities   |
| `autoware_ml/models/detection2d/rtdetrv4/hybrid_encoder.py` | Feature encoder and deploy conversion hooks  |
| `autoware_ml/models/detection2d/rtdetrv4/dfine_decoder.py`  | Transformer decoder and box refinement logic |
| `autoware_ml/models/detection2d/rtdetrv4/rtv4_criterion.py` | Matching and training loss computation        |
| `autoware_ml/models/detection2d/rtdetrv4/postprocessor.py`  | Top-k selection and box rescaling             |
| `autoware_ml/models/detection2d/base.py`                | Shared detection2d Lightning evaluation flow     |
| `autoware_ml/datamodule/coco/detection2d.py`            | COCO-style 2D detection datamodule               |
| `autoware_ml/datamodule/detection2d/base.py`            | Shared dataset, batching, and multiscale logic   |
| `autoware_ml/transforms/detection2d/`                   | Detection image loading and augmentation stack   |
| `autoware_ml/metrics/detection2d.py`                    | COCO and localization metric helpers             |
| `autoware_ml/configs/tasks/detection2d/rtdetrv4/`       | RT-DETRv4 task configurations                    |
| `autoware_ml/scripts/visualize_detection2d.py`          | Offline prediction visualization utility         |
