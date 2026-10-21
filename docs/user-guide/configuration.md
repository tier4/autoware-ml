---
icon: lucide/settings
---

# Configuration

Autoware-ML uses [Hydra](https://hydra.cc/) for configuration management. This gives you hierarchical YAML configs with powerful runtime overrides without code changes.

## Config Structure

All configs live in `autoware_ml/configs/`:

```text
configs/
├── defaults/          # Base settings and module defaults
└── tasks/       # Experiment configs, one tree per task and model
```

## Hydra Syntax

### `# @package _global_`

Always include this directive at the top of experiment configs to merge contents at the root
level. A model base composes the experiment root, which brings in the runtime modules:

```yaml
# @package _global_
defaults:
  - /tasks/default_task
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

Controls the dataset sources, datasets, transforms and dataloaders of every split. Task configs
select a datamodule config from `configs/datamodule/` and override what they need:

```yaml
defaults:
  - /database@database: t4dataset/t4dataset_j6gen2_semaseg
  - /datamodule@datamodule: t4dataset/default_segmentation3d_datamodule
  - _self_

# Datasets of the training and test splits, the validation and predict splits read the test one
datasets:
  train:
    max_num_3d_gt_bboxes: 0
    # Only the semantic masks of the corpus supervise the run
    det3d_supervised: false
    # Transform pipeline of the training split
    transforms:
      _target_: autoware_ml.transforms.base.TransformsCompose
      _convert_: all
      pipeline:
        - _target_: autoware_ml.transforms.point_cloud.loading.LoadPointsFromFile
          use_dim: [0, 1, 2, 3]

datamodule:
  # Dataset sources by name, several sources mix their corpora
  sources:
    main:
      repeat: 1

  train_dataloader:
    batch_size: 8
    num_workers: 4
    shuffle: true
```

A split (`train`, `validation`, `test`, `predict`) is served by the sources declaring a dataset
for it:

- `datasets.train` and `datasets.test` are the dataset configs the datamodule config composes at
  the top level, instantiated without records. They name the database they read
  (`database_root_path`, `lidar_intensity_scale`, both from `${database}`), whether its boxes
  (`det3d_supervised`) and semantic masks (`seg3d_supervised`) supervise the run, and carry the
  task datasets (`dataset_tasks`) and the transforms of the split. The validation and predict
  splits read the test one. A T4 dataset also carries the lanelet `map_provider` of its
  database, resolved against its own `database_root_path`. The dataset writes every frame's
  lanelet map path into the frame metadata, so the map-backed evaluation filters read each
  frame's map from the corpus that served it, whichever sources a run evaluates.
- `datamodule.sources.<name>` is a `DatasetSource` that Hydra instantiates. It names a database,
  the dataset of every split it serves (`train_dataset`, `validation_dataset`, `test_dataset`,
  `predict_dataset`, `null` for a split it stays out of), and how many times its frames appear
  in one training epoch (`repeat`). Every dataset of a source must read the database of that
  source, so a source over a second database composes its own copy of the split dataset config.
  The default source `main` reads `${database}` with `${datasets.train}` for training and
  `${datasets.test}` for the other splits. In `setup()` the datamodule binds a copy of each
  dataset to the records of its database and concatenates the sources of a split in declaration
  order.
- `<split>_dataloader` holds the dataloader settings (batch size, workers, shuffling, pin_memory).

The dataset configs leave `transforms` required. Every model writes the pipeline of the training
and test datasets in its own task config, so two models never share a pipeline and a fine tuning
config changes only the values it needs.

For custom components, point `_target_` at the concrete implementation module,
for example `autoware_ml.transforms.my_transforms.my_transform.MyTransform` or
`autoware_ml.models.common.backbones.my_backbone.MyBackbone`.

### `data_preprocessor`

Defines runtime preprocessing that runs after batch transfer and before
`forward()`. The experiment root declares an empty `DataPreprocessor`, a model base lists the
modules it needs, and the builders hand the instance to the model:

```yaml
data_preprocessor:
  preprocessor_modules:
    - _target_: autoware_ml.preprocessing.my_preprocessing.my_preprocessing.MyPreprocessingLayer
      param: value
```

Output-side shaping (logits -> probabilities, voxel-to-point scatter, etc.)
is **not** a configurable framework pipeline - it lives inside the model's
own `forward()`, `compute_metrics()`, `build_eval_output()` and `decode_outputs()` methods.

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

  dynamic_validation:
    _target_: autoware_ml.callbacks.dynamic_validation.DynamicValidation
    # [start_epoch, check_val_every_n_epoch] pairs: validate every 5 epochs, then every epoch
    # from epoch 20 onwards. The root leaves this at ??? and every leaf sets it next to its
    # trainer.max_epochs, since the schedule follows the epoch count.
    epoch_intervals: [[0, 5], [20, 1]]
```

`DynamicValidation` rewrites `trainer.check_val_every_n_epoch` at the start of every epoch
according to its schedule, so the validation frequency can change as training progresses. It
requires `trainer.val_check_interval: 1.0` and an integer `trainer.check_val_every_n_epoch`,
which is the initial value before the first epoch. It never modifies `trainer.log_every_n_steps`.

Every leaf follows one rule: validate every 5 epochs, then every epoch over the last 10 epochs
of its schedule, so the best checkpoint is picked at epoch resolution where it matters. A
30-epoch leaf sets `[[0, 5], [20, 1]]`, a 50-epoch leaf `[[0, 5], [40, 1]]`, a 100-epoch leaf
`[[0, 5], [90, 1]]`, and a run of 10 epochs or fewer validates every epoch with `[[0, 1]]`.
Tasks validated by steps rather than epochs, or whose loss is the only training signal, set
`dynamic_validation: null` in their base config and keep Lightning's own cadence.

The root also turns on the `IterationTimer` and `DataProcessingTimer` callbacks, which log the
step and data loading times on every batch, written out at `trainer.log_every_n_steps` like
every other step metric, and the `RichProgressBar`, for every task. The `GradNormMonitor`
callback logs `train/grad_norm_l2`, the norm of the gradients on every optimizer step, taken after
AMP unscaling and before `trainer.gradient_clip_val` clips them, so the curve shows how often
clipping is active; `grad_norm_monitor.norm_type` picks the order, `l1`, `l2` or `inf`, and names
the logged key. The `ModelSummaryExampleInput` callback sets the first training sample as the
module's example input at fit start, so the model summary reports the FLOPs of one sample. The
summary runs that forward pass once on every rank. Set `model_summary_example_input: null` to
skip it. The callback fails fit when the datamodule serves no training sample.

### `logger`

MLflow experiment tracking:

```yaml
logger:
  _target_: lightning.pytorch.loggers.MLFlowLogger
  tracking_uri: sqlite:///mlruns/mlflow.db
```

Autoware-ML populates `experiment_name`, `run_name`, `run_id`, and default tags automatically at runtime.

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
    workspace_size: 8589934592  # 8 GiB
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
  - /tasks/my_task/my_model/base  # Inherit base config
  - _self_                        # Apply this file's overrides

# Override specific values
datamodule:
  train_dataloader:
    batch_size: 4
```

## Variable Interpolation

Reference other config values with `${...}`:

```yaml
batch_size: 8

datamodule:
  train_dataloader:
    batch_size: ${batch_size}
  validation_dataloader:
    batch_size: ${batch_size}
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
autoware-ml train --config-name <task>/<model>/<config> \
    trainer.max_epochs=100

# Nested override
autoware-ml train --config-name <task>/<model>/<config> \
    model.optimizer.lr=0.0005

# Add new parameter (use + prefix)
autoware-ml train --config-name <task>/<model>/<config> \
    +callbacks.my_callback._target_=lightning.pytorch.callbacks.MyCallback
```

!!! warning "Override vs Add"
    Use `+` prefix only when adding a **new** parameter that doesn't exist in the config. For overriding existing parameters, use the path directly without `+`.

## Creating Custom Configs

Create a new YAML file inheriting from a base config:

```yaml title="configs/tasks/my_task/my_model/my_experiment.yaml"
# @package _global_
defaults:
  - /tasks/my_task/my_model/base
  - _self_

trainer:
  max_epochs: 50

model:
  optimizer:
    lr: 0.0001
```

Run with your config:

```bash
autoware-ml train --config-name my_task/my_model/my_experiment
```

## Debugging Configs

Print the resolved config without running:

```bash
autoware-ml train --config-name <task>/<model>/<config> --cfg job
```

Print a specific section:

```bash
autoware-ml train --config-name <task>/<model>/<config> \
    --cfg job --package model
```

## Multi-Run (Sweeps)

Run parameter sweeps with `--multirun`:

```bash
autoware-ml train --config-name <task>/<model>/<config> \
    --multirun \
    model.optimizer.lr=0.001,0.0005,0.0001
```

For intelligent hyperparameter search, see [Optuna](optuna.md).

## Learn More

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Reference](https://omegaconf.readthedocs.io/)
