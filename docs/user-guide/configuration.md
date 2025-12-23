---
icon: lucide/settings
---

# Configuration

Autoware-ML uses [Hydra](https://hydra.cc/) for configuration management. This gives you hierarchical YAML configs with powerful runtime overrides-no code changes needed to experiment with different settings.

## Config Structure

All configs live in `autoware_ml/configs/`:

```text
configs/
├── defaults/          # Base settings and module defaults
└── tasks/             # Task-specific configs
```

## Hydra Syntax

### `# @package _global_`

Always include this directive at the top of task configs to merge contents at the root level:

```yaml
# @package _global_
defaults:
  - /defaults/default_runtime
  - _self_
```

### `_target_`

Specifies the Python class or function to instantiate:

```yaml
model:
  _target_: autoware_ml.models.my_task.MyModel
  num_classes: 10
```

Nested `_target_` keys are recursively instantiated by default.

### `_partial_`

Use `_partial_: true` when you want Hydra to create a `functools.partial` instead of calling the function immediately:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 0.001
  weight_decay: 0.01
```

This creates `functools.partial(AdamW, lr=0.001, weight_decay=0.01)` which can later be called with additional arguments (like `params`).

### `_recursive_`

Controls whether nested `_target_` keys are instantiated (default: `true`). Set to `false` to receive raw configs.

## Top-Level Config Keys

A complete task config includes these sections:

### `datamodule`

Controls data loading and preprocessing:

```yaml
datamodule:
  _target_: autoware_ml.datamodule.my_dataset.my_task.MyDataModule
  data_root: ${data_root}
  train_ann_file: ${data_root}/info/train.pkl
  val_ann_file: ${data_root}/info/val.pkl

  stack_keys: [input_tensor, gt_labels]

  train_dataloader_cfg:
    batch_size: 8
    num_workers: 4
    shuffle: true

  train_transforms:
    pipeline:
      - _target_: autoware_ml.transforms.my_transforms.MyTransform
        param: value

  data_preprocessing:
    _target_: autoware_ml.preprocessing.DataPreprocessing
    pipeline:
      - _target_: autoware_ml.preprocessing.my_preprocessing.MyPreprocessing
        param: value
```

### `model`

Defines model architecture and optimization:

```yaml
model:
  _target_: autoware_ml.models.my_task.MyModel

  backbone:
    _target_: autoware_ml.models.common.backbones.my_backbone.MyBackbone
    in_channels: 3

  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true
    lr: 0.001
    weight_decay: 0.01

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    _partial_: true
    T_max: ${trainer.max_epochs}
```

### `trainer`

PyTorch Lightning Trainer settings:

```yaml
trainer:
  _target_: lightning.Trainer
  max_epochs: 30
  accelerator: auto
  devices: auto
  precision: 16-mixed
  gradient_clip_val: 10.0
  gradient_clip_algorithm: norm
  accumulate_grad_batches: 1
  check_val_every_n_epoch: 1
  log_every_n_steps: 10
```

### `callbacks`

Lightning callbacks for checkpointing, early stopping, etc.:

```yaml
callbacks:
  model_checkpoint:
    _target_: lightning.pytorch.callbacks.ModelCheckpoint
    dirpath: ${hydra:run.dir}/checkpoints
    filename: "epoch={epoch}-step={step}"
    save_top_k: 3
    monitor: val/loss
    mode: min
    save_last: true

  early_stopping:
    _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: val/loss
    patience: 10
    mode: min
```

### `logger`

MLflow experiment tracking:

```yaml
logger:
  _target_: lightning.pytorch.loggers.MLFlowLogger
  tracking_uri: sqlite:///mlruns/mlflow.db
```

### `deploy`

ONNX and TensorRT export settings:

```yaml
deploy:
  onnx:
    opset_version: 21
    input_names: [input]
    output_names: [output]
    dynamic_shapes:
      fused_img: { 2: height, 3: width }
    modify_graph: null  # Optional graph modifier

  tensorrt:
    workspace_size: 1073741824  # 1GB
    input_shapes:
      input:
        min_shape: [1, 5, 1080, 1920]
        opt_shape: [1, 5, 1440, 2560]
        max_shape: [1, 5, 2160, 3840]
```

## Config Inheritance

Configs inherit using the `defaults` key:

```yaml
# @package _global_
defaults:
  - /my_task/my_model_base  # Inherit base config
  - _self_                        # Apply this file's overrides

# Override specific values
data_root: /path/to/dataset

datamodule:
  data_root: ${data_root}
```

## Variable Interpolation

Reference other config values with `${...}`:

```yaml
data_root: /path/to/dataset

datamodule:
  data_root: ${data_root}
  train_ann_file: ${data_root}/info/train.pkl
```

Hydra resolvers:

```yaml
output_dir: ${hydra:run.dir}          # Hydra's output directory
experiment: ${hydra:job.config_name}  # Config name
```

## Runtime Overrides

Override any value from the command line:

```bash
# Override existing parameter (no + prefix)
autoware-ml train --config-name my_task/my_model \
    trainer.max_epochs=100

# Nested override
autoware-ml train --config-name my_task/my_model \
    model.optimizer.lr=0.0005

# Add new parameter (use + prefix)
autoware-ml train --config-name my_task/my_model \
    +callbacks.my_callback._target_=lightning.pytorch.callbacks.MyCallback
```

!!! warning "Override vs Add"
    Use `+` prefix only when adding a **new** parameter that doesn't exist in the config. For overriding existing parameters, use the path directly without `+`.

## Creating Custom Configs

Create a new YAML file inheriting from a base config:

```yaml title="configs/my_task/my_experiment.yaml"
# @package _global_
defaults:
  - /my_task/my_model_base
  - _self_

trainer:
  max_epochs: 50

model:
  optimizer:
    lr: 0.0001
```

Run with your config:

```bash
autoware-ml train --config-name my_task/my_experiment
```

## Debugging Configs

Print the resolved config without running:

```bash
autoware-ml train --config-name my_task/my_model --cfg job
```

Print a specific section:

```bash
autoware-ml train --config-name my_task/my_model \
    --cfg job --package model
```

## Multi-Run (Sweeps)

Run parameter sweeps with `--multirun`:

```bash
autoware-ml train --config-name my_task/my_model \
    --multirun \
    model.optimizer.lr=0.001,0.0005,0.0001
```

For intelligent hyperparameter search, see [Optuna](optuna.md).

## Learn More

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Reference](https://omegaconf.readthedocs.io/)
