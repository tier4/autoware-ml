---
icon: lucide/plus-circle
---

# Adding Models

This guide walks you through adding a new model to Autoware-ML. You'll implement a model class, pick the data it trains on, and wire everything together with a config.

## The BaseModel Interface

New models should inherit from `BaseModel`. The minimal contract is three abstract methods:

```python
from autoware_ml.models.base import BaseModel

class MyModel(BaseModel):
    def forward(self, **kwargs: Any) -> Any:
        ...

    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        ...

    def compute_metrics(
        self, batch_inputs: ModelBatchInputs, outputs: Any
    ) -> dict[str, torch.Tensor]:
        ...
```

The base class handles training/validation/test/predict steps, optimizer
configuration, metric logging, prediction output conversion, runtime
preprocessing, and deployment export integration. The `forward()` method
takes the tensors the network reads, and `forward_inputs()` picks them from
the typed model inputs.

!!! note "Extending `BaseModel`"
    Specialized models should still use `BaseModel`. When the default
    shared step path is not enough, prefer overriding hooks such as
    `set_data_preprocessing()`, `predict_outputs()`, `get_log_batch_size()`,
    or `build_export_spec()` instead of introducing a standalone
    `LightningModule`. Output decoding (for example, voxel-to-point scatter
    for segmentation) belongs inside the model, typically in `forward()`,
    `compute_metrics()`, and `predict_outputs()` - not in a separate
    framework pipeline.

## Step 1: Implement the Model

Create a new file in `autoware_ml/models/`:

```python title="autoware_ml/models/my_task/my_model.py"
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.models.base import BaseModel


class MyModel(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        ignore_index: int,
        **kwargs: Any,  # Pass optimizer, scheduler and metrics to BaseModel
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, feat: torch.Tensor) -> ModelOutputs:
        features = self.encoder(feat)
        logits = self.decoder(features)
        return ModelOutputs(segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=logits))

    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        return {"feat": batch_inputs.multi_task_gt_batch.point_cloud_gt_batch.points}

    def compute_metrics(
        self,
        batch_inputs: ModelBatchInputs,
        outputs: ModelOutputs,
    ) -> dict[str, torch.Tensor]:
        labels = batch_inputs.multi_task_gt_batch.segmentation3d_gt_batch.gt_semantic_masks
        loss = self.loss_fn(outputs.segmentation3d().logits, labels)
        return {"loss": loss}
```

### Key Points

1. **`forward_inputs()` picks the forward arguments** - It reads the typed model inputs, for
   example the points of `multi_task_gt_batch.point_cloud_gt_batch` or the `voxels_data` a
   preprocessing layer computed, and returns them keyed by the `forward()` parameter names.

2. **`compute_metrics()` receives the model inputs and outputs** - The first argument is the
   `ModelBatchInputs` of the batch, and the second is `outputs` from `forward()`. Read the
   targets from its ground truth batches, for example the point labels of
   `segmentation3d_gt_batch`.

3. **Return `'loss'`** - The metrics dict must include a `'loss'` key for backpropagation.

4. **Typed outputs** - `forward()` returns a `ModelOutputs` with the head outputs of every task
   the model predicts, and `predict_outputs()` returns a `ModelPredictions`. A task the model does
   not predict stays unset.

5. **Optimizer and scheduler** - Passed as callables to `BaseModel.__init__()`. Need to be marked as `_partial_: true` in YAML configs.

6. **Use hooks when needed** - If your model needs custom batch unpacking,
   prediction formatting, or an explicit deployment wrapper, override the
   appropriate `BaseModel` hook instead of bypassing the shared training and
   deployment flow.

## Step 2: Choose the Data

Models do not bring their own data module. Every task config uses the shared `DataModule`
(`autoware_ml.datamodule.data_module.DataModule`), which reads the record table of a database,
splits it by scenario lists and builds one dataset per dataset source of a split. A model picks:

- a database config from `configs/database/`, which names the record table and its taxonomy
- a datamodule config from `configs/datamodule/<database>/`, which selects the task datasets of
  every split, for example `t4dataset/default_segmentation3d_datamodule`
- the transforms of every split, written in its own task config

The dataset of a split is `T4Dataset` (or `NuScenesDataset`) with one task dataset per task,
such as `Detection3DTask` or `T4Segmentation3DTask`. A task dataset reads the annotations of a
record and returns them as a `ModelGTSample`.

### Data Flow

```text
get_data_sample() -> transforms -> collate_fn() -> BaseModel.on_after_batch_transfer() -> forward() -> compute_metrics()/predict_outputs()
```

1. `get_data_sample()`: return the sensor file paths, calibration and annotations of a record
   as a `ModelGTSample`
2. `transforms`: load files and apply per sample augmentations (in the dataset)
3. `collate_fn()`: stack the samples into a typed `ModelGTBatch`
4. `BaseModel.on_after_batch_transfer()`: run the model's runtime preprocessing, which turns
   the batch into the typed `ModelBatchInputs`
5. `forward()`: model inference/training forward pass
6. `compute_metrics()` / `predict_outputs()`: model owns any output shaping
   (e.g., voxel-to-point scatter for segmentation) directly inside these
   methods

### Supporting a New Database or Annotation

A new database family subclasses `BaseDatabase` to write its record table, see
[Database Design](../databases/design.md). Its dataset subclasses `BaseDataset`, or `T4Dataset`
when the record table follows the T4 layout. A new kind of annotation gets a task dataset
deriving `BaseDatasetTask`, listed under `dataset_tasks` of the dataset config.

## Step 3: Create Config

Create a task config:

```yaml title="configs/tasks/my_task/my_model/base.yaml"
# @package _global_
defaults:
  - /defaults/default_runtime
  - _self_

model:
  _target_: autoware_ml.models.my_task.my_model.MyModel
  num_classes: ${dataset.segmentation3d.num_classes}
  ignore_index: ${dataset.segmentation3d.ignore_index}
  metrics: ${dataset.segmentation3d.metrics}

  encoder:
    _target_: torch.nn.Linear
    in_features: 5
    out_features: 64

  decoder:
    _target_: torch.nn.Linear
    in_features: 64
    out_features: ${model.num_classes}

  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true
    lr: 0.001
    weight_decay: 0.01

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    _partial_: true
    T_max: ${trainer.max_epochs}

trainer:
  max_epochs: 50

data_preprocessing:
  _target_: autoware_ml.preprocessing.base.DataPreprocessing
  pipeline: []
```

Create a dataset-specific config that selects the database, the datamodule and the dataloader
settings:

```yaml title="configs/tasks/my_task/my_model/my_config.yaml"
# @package _global_
defaults:
  - /tasks/my_task/my_model/base
  - /datasets/t4dataset/segmentation3d
  - /datasets/t4dataset/lidar
  - /database@database: t4dataset/t4dataset_j6gen2_semaseg
  - /datamodule@datamodule: t4dataset/default_segmentation3d_datamodule
  - _self_

dataset: ${t4dataset}
point_cloud_range: [-122.88, -122.88, -3.0, 122.88, 122.88, 5.0]

datamodule:
  train_dataset:
    transforms:
      _target_: autoware_ml.transforms.base.TransformsCompose
      _convert_: all
      pipeline:
        - _target_: autoware_ml.transforms.point_cloud.loading.LoadPointsFromFile
          use_dim: [0, 1, 2, 3]
        - _target_: autoware_ml.transforms.point_cloud.geometry.GlobalRotScaleTrans
          yaw_rot_range: [-3.14159265, 3.14159265]
          scale_ratio_range: [0.9, 1.1]
          translation_std: [0.5, 0.5, 0.2]
        - _target_: autoware_ml.transforms.point_cloud.geometry.PointsRangeFilter
          points_range: ${point_cloud_range}
  test_dataset:
    transforms:
      _target_: autoware_ml.transforms.base.TransformsCompose
      _convert_: all
      pipeline:
        - _target_: autoware_ml.transforms.point_cloud.loading.LoadPointsFromFile
          use_dim: [0, 1, 2, 3]
        - _target_: autoware_ml.transforms.point_cloud.geometry.PointsRangeFilter
          points_range: ${point_cloud_range}
  train_dataloader:
    batch_size: 8
    num_workers: 4
    shuffle: true
  validation_dataloader:
    batch_size: 8
    num_workers: 4
```

The dataset configs leave `transforms` required, so every task config writes the pipeline of the
training and test datasets. Validation and predict use the test pipeline. Models never share a
pipeline, a model that needs the same steps copies them.

A split can mix several corpora. Each entry of `train_sources`, `validation_sources`,
`test_sources` or `predict_sources` names a database, whether its boxes (`det3d`) and semantic
masks (`seg3d`) supervise the run, and how many times its frames appear in one epoch (`repeat`):

```yaml
datamodule:
  train_sources:
    - database: ${database}
    # A second database config composed into the task
    - database: ${extra_database}
      det3d: false
      repeat: 2
```

!!! note
    Some parameters are inherited from the default runtime config. Take a look at `configs/defaults/default_runtime.yaml` for more details.

Runtime preprocessing lives at the top level of the composed config and is
attached to the model by the entrypoints.

## Step 4: Add Transforms (Optional)

Transforms take a `ModelGTSample` and return the updated sample. The sample is a named tuple,
so a transform returns a copy with `_replace()`:

```python title="autoware_ml/transforms/point_cloud/my_transform.py"
import numpy as np

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.transforms.base import BaseTransform


class MyAugmentation(BaseTransform):
    _required_keys = ["point_cloud_data"]

    def __init__(self, probability: float = 0.5, intensity: float = 0.1):
        # BaseTransform skips the transform with the remaining probability
        super().__init__(probability=probability)
        self.intensity = intensity

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        point_cloud_data = model_gt_sample.point_cloud_data
        # Your augmentation logic on point_cloud_data
        return model_gt_sample._replace(point_cloud_data=point_cloud_data)
```

Add the transform to the training pipeline in your task config:

```yaml title="configs/tasks/my_task/my_model/my_config.yaml"
datamodule:
  train_dataset:
    transforms:
      _target_: autoware_ml.transforms.base.TransformsCompose
      _convert_: all
      pipeline:
        - _target_: autoware_ml.transforms.point_cloud.loading.LoadPointsFromFile
          use_dim: [0, 1, 2, 3]
        - _target_: autoware_ml.transforms.point_cloud.my_transform.MyAugmentation
          probability: 0.5
          intensity: 0.1
```

## Step 5: Add Runtime Data Preprocessing (Optional)

Runtime preprocessing runs on the target device after batch transfer and
before the forward pass. It is configured at the top level and attached to
the model by the entrypoint scripts.

If your task needs custom preprocessing:

```python title="autoware_ml/preprocessing/my_preprocessing/my_preprocessing.py"
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs


class MyPreprocessingLayer:
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        batch = batch_inputs.multi_task_gt_batch
        points = batch.point_cloud_gt_batch
        scaled = points._replace(points=points.points * self.scale)
        return batch_inputs.replace(multi_task_gt_batch=batch._replace(point_cloud_gt_batch=scaled))
```

Add to config:

```yaml
data_preprocessing:
  _target_: autoware_ml.preprocessing.base.DataPreprocessing
  pipeline:
    - _target_: autoware_ml.preprocessing.my_preprocessing.my_preprocessing.MyPreprocessingLayer
      scale: 1.0
```

!!! warning
    Preprocessing layers must be callable objects that accept `ModelBatchInputs` and the keyword
    `is_training`, and return `ModelBatchInputs` with the features they compute added or the
    data they change replaced.

Output-side shaping (logits -> probabilities, decoder scatter, voxel-to-point mapping, etc.) belongs
**inside the model** - in `forward()`, `compute_metrics()`, or `predict_outputs()`.

## Step 6: Train and Deploy

### Config Naming Convention

Task configs should follow:

```text
<task>/<model>/<variant>_<dataset>
```

Use these rules when creating `<variant>`:

- include only future-distinguishing choices such as backbone, modality, voxel size, or range
- do not encode properties that are inherent to the model family
- normalize voxel sizes as `voxel020`, `voxel005`
- encode ranges as human-readable suffixes such as `50m`, `90m`, `102m`, `121m`
- keep dataset names explicit and stable, for example `nuscenes` and `t4dataset_j6gen2`

Examples:

```text
segmentation3d/ptv3/voxel005_51m_nuscenes
segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2
my_task/my_model/my_variant_my_dataset
```

```bash
# Train
autoware-ml train --config-name my_task/my_model/my_config

# Deploy
autoware-ml deploy \
    --config-name my_task/my_model/my_config \
    --weights mlruns/my_task/my_model/my_config/<run_id>/artifacts/checkpoints/last.ckpt
```

## Common Patterns

### Multiple Inputs

```python
def forward(self, image: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
    img_features = self.image_encoder(image)
    lidar_features = self.lidar_encoder(lidar)
    fused = torch.cat([img_features, lidar_features], dim=1)
    return self.head(fused)
```

`forward_inputs()` returns both tensors, keyed `image` and `lidar`.

### Multiple Tasks

```python
def forward(self, x: torch.Tensor) -> ModelOutputs:
    features = self.backbone(x)
    return ModelOutputs(
        detection3d_head_outputs=Detection3DHeadOutputs(
            center_head_outputs=None, transfusion_head_outputs=self.bbox_head(features)
        ),
        segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=self.seg_head(features)),
    )

def compute_metrics(self, batch_inputs: ModelBatchInputs, outputs: ModelOutputs):
    gt_batch = batch_inputs.multi_task_gt_batch
    gt_detections = gt_batch.detection3d_gt_batch
    box_losses = self.bbox_head.loss(
        outputs.detection3d().transfusion_head(),
        gt_detections.valid_bboxes_3d(),
        gt_detections.valid_labels_3d(),
    )
    seg_loss = self.seg_loss(
        outputs.segmentation3d().logits, gt_batch.segmentation3d_gt_batch.gt_semantic_masks
    )
    return {"loss": box_losses["loss"] + seg_loss, "seg_loss": seg_loss}
```
