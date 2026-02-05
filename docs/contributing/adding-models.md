---
icon: lucide/plus-circle
---

# Adding Models

This guide walks you through adding a new model to Autoware-ML. You'll implement a model class, create a DataModule, and wire everything together with a config.

## The BaseModel Interface

All models inherit from `BaseModel` and implement two abstract methods:

```python
from autoware_ml.models.base import BaseModel

class MyModel(BaseModel):
    def forward(self, **kwargs: Any) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        ...

    def compute_metrics(
        self, outputs: Union[torch.Tensor, Sequence[torch.Tensor]], **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        ...
```

The base class handles training/validation/test steps, optimizer configuration, and metric logging automatically. The `forward()` method can have any signature - the base class automatically filters batch inputs to match the method signature.

## Step 1: Implement the Model

Create a new file in `autoware_ml/models/`:

```python title="autoware_ml/models/my_task/my_model.py"
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from autoware_ml.models.base import BaseModel


class MyModel(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        **kwargs: Any,  # Pass optimizer, scheduler to BaseModel
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.encoder(input_tensor)
        logits = self.decoder(features)
        return logits

    def compute_metrics(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        gt_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        loss = self.loss_fn(logits, gt_labels)

        # Optional: compute accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == gt_labels).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
        }
```

### Key Points

1. **`forward()` signature matters** - Parameter names must match keys in your batch dictionary. The base class automatically extracts matching keys using signature inspection.

2. **`compute_metrics()` receives outputs** - The first argument is always `outputs` from `forward()` (as a `Union[torch.Tensor, Sequence[torch.Tensor]]`). Additional parameters are matched from the batch.

3. **Return `'loss'`** - The metrics dict must include a `'loss'` key for backpropagation.

4. **Optimizer and scheduler** - Passed as callables to `BaseModel.__init__()`. Need to be marked as `_partial_: true` in YAML configs.

## Step 2: Create a DataModule

Create a DataModule that provides data for your model:

```python title="autoware_ml/datamodule/my_dataset/my_task.py"
from typing import Any, Dict, Optional
import pickle

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.transforms import TransformsCompose


class MyDataset(Dataset):
    def __init__(
        self,
        ann_file: str,
        data_root: str,
        dataset_transforms: Optional[TransformsCompose] = None,
    ):
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root

        # Load annotations
        with open(ann_file, "rb") as f:
            self.annotations = pickle.load(f)

    def __len__(self) -> int:
        return len(self.annotations)

    def _get_input_dict(self, index: int) -> Dict[str, Any]:
        ann = self.annotations[index]

        # Load your data
        input_tensor = self._load_input(ann)
        gt_labels = ann["label"]

        return {
            "input_tensor": input_tensor,
            "gt_labels": gt_labels,
        }

    def _load_input(self, ann):
        # Your data loading logic
        ...


class MyDataModule(DataModule):
    def __init__(
        self,
        data_root: str,
        train_ann_file: str,
        val_ann_file: str,
        test_ann_file: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.train_ann_file = train_ann_file
        self.val_ann_file = val_ann_file
        self.test_ann_file = test_ann_file or val_ann_file

    def _create_dataset(
        self,
        split: str,
        transforms: Optional[TransformsCompose] = None,
    ) -> Dataset:
        ann_file = {
            "train": self.train_ann_file,
            "val": self.val_ann_file,
            "test": self.test_ann_file,
            "predict": self.test_ann_file,
        }[split]

        return MyDataset(
            ann_file=ann_file,
            data_root=self.data_root,
            dataset_transforms=transforms,
        )
```

### Data Flow

```text
_get_input_dict() → transforms → collate_fn() → on_after_batch_transfer() → model
```

1. `_get_input_dict()`: Load raw sample as dict
2. `transforms`: Apply per-sample augmentations (in Dataset)
3. `collate_fn()`: Batch samples, convert to tensors
4. `on_after_batch_transfer()`: GPU preprocessing (optional)
5. Model receives the batch dict

## Step 3: Register Components

Add `__init__.py` exports:

```python title="autoware_ml/models/my_task/__init__.py"
from autoware_ml.models.my_task.my_model import MyModel

__all__ = ["MyModel"]
```

```python title="autoware_ml/datamodule/my_dataset/__init__.py"
from autoware_ml.datamodule.my_dataset.my_task import MyDataModule, MyDataset

__all__ = ["MyDataModule", "MyDataset"]
```

## Step 4: Create Config

Create a task config:

```yaml title="configs/my_task/my_model_base.yaml"
# @package _global_
defaults:
  - /defaults/default_runtime
  - _self_

datamodule:
  _target_: autoware_ml.datamodule.my_dataset.my_task.MyDataModule
  stack_keys: [input_tensor, gt_labels]  # Keys to stack into tensors

  train_dataloader_cfg:
    batch_size: 8
    num_workers: 4
    shuffle: true

  val_dataloader_cfg:
    batch_size: 8
    num_workers: 4

  # GPU preprocessing (optional)
  data_preprocessing:
    _target_: autoware_ml.preprocessing.base.DataPreprocessing
    pipeline: []

model:
  _target_: autoware_ml.models.my_task.MyModel
  num_classes: 10

  encoder:
    _target_: autoware_ml.models.common.backbones.resnet.ResNet18
    in_channels: 3

  decoder:
    _target_: torch.nn.Linear
    in_features: 512
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
```

Create a dataset-specific config:

```yaml title="configs/my_task/my_model_my_dataset.yaml"
# @package _global_
defaults:
  - /my_task/my_model_base
  - _self_

data_root: /autoware-ml-data/my_dataset

datamodule:
  data_root: ${data_root}
  train_ann_file: ${data_root}/info/train.pkl
  val_ann_file: ${data_root}/info/val.pkl
```

!!! note
    Some of parameters are inherited from the default runtime config. Take a look on `configs/defaults/default_runtime.yaml` for more details.

## Step 5: Add Transforms (Optional)

If your task needs custom transforms:

```python title="autoware_ml/transforms/my_transforms/my_transform.py"
from typing import Any, Dict
import numpy as np

from autoware_ml.transforms.base import BaseTransform


class MyAugmentation(BaseTransform):
    def __init__(self, p: float = 0.5, intensity: float = 0.1):
        self.p = p
        self.intensity = intensity

    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.p:
            return {}  # No changes

        # Your augmentation logic
        input_tensor = input_dict["input_tensor"]
        augmented = input_tensor + np.random.randn(*input_tensor.shape) * self.intensity

        return {"input_tensor": augmented}
```

Add to config:

```yaml
datamodule:
  train_transforms:
    pipeline:
      - _target_: autoware_ml.transforms.my_transforms.MyAugmentation
        p: 0.5
        intensity: 0.1
```

## Step 6: Add Preprocessing (Optional)

Preprocessing runs on GPU after batch transfer, enabling hardware-accelerated operations. Unlike transforms (CPU-side, per-sample), preprocessing operates on entire batches already on the target device.

If your task needs custom preprocessing:

```python title="autoware_ml/preprocessing/my_preprocessing/my_preprocessing.py"
from typing import Any, Dict

import torch
import torch.nn as nn


class MyPreprocessingLayer(nn.Module):
    def __init__(self, input_key: str = "input_tensor", scale: float = 1.0):
        super().__init__()
        self.input_key = input_key
        self.scale = scale

    def forward(self, batch_inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        processed = batch_inputs_dict[self.input_key] * self.scale
        return {self.input_key: processed}
```

Add to config:

```yaml
datamodule:
  data_preprocessing:
    _target_: autoware_ml.preprocessing.base.DataPreprocessing
    pipeline:
      - _target_: autoware_ml.preprocessing.my_preprocessing.MyPreprocessingLayer
        input_key: input_tensor
        scale: 1.0
```

!!! warning
    Preprocessing layers must be `nn.Module` subclasses that accept `Dict[str, Any]` and return `Dict[str, Any]`.

## Step 7: Train and Deploy

```bash
# Train
autoware-ml train --config-name my_task/my_model_my_dataset

# Deploy
autoware-ml deploy \
    --config-name my_task/my_model_my_dataset \
    +checkpoint=mlruns/<date>/<time>/checkpoints/last.ckpt
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

Batch dict must have `image` and `lidar` keys.

### Multiple Outputs

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    features = self.backbone(x)
    boxes = self.box_head(features)
    scores = self.score_head(features)
    return boxes, scores

def compute_metrics(self, outputs: Tuple[torch.Tensor, torch.Tensor], gt_boxes: torch.Tensor, gt_scores: torch.Tensor):
    boxes, scores = outputs
    box_loss = self.box_loss(boxes, gt_boxes)
    score_loss = self.score_loss(scores, gt_scores)
    return {"loss": box_loss + score_loss, "box_loss": box_loss, "score_loss": score_loss}
```
