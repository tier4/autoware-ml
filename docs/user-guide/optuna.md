---
icon: lucide/sliders-horizontal
---

# Optuna

Autoware-ML integrates [Optuna](https://optuna.org/) for automated hyperparameter optimization. Optuna intelligently searches the hyperparameter space, finding optimal configurations faster than grid search or random search.

## How It Works

Optuna sits above Hydra in the configuration hierarchy:

```text
Optuna (suggests hyperparameters)
    ↓
Hydra (configures training)
    ↓
Lightning (runs training)
    ↓
MLflow (logs results)
```

Each trial:

1. Optuna suggests hyperparameter values
2. Hydra builds the config with those values
3. Training runs and reports the objective metric
4. Optuna uses the result to inform the next trial

## Running a Hyperparameter Search

```bash
autoware-ml train --config-name my_task/my_model \
    --multirun \
    hydra/sweeper=optuna \
    hydra.sweeper.n_trials=50 \
    hydra.sweeper.direction=minimize
```

This launches 50 trials, minimizing the objective metric (default: validation loss).

## Defining Search Spaces

Define hyperparameter ranges in your config:

```yaml title="configs/my_task/my_model_optuna.yaml"
# @package _global_
defaults:
  - /my_task/my_model_base
  - _self_

hydra:
  sweeper:
    params:
      model.optimizer.lr: interval(0.0001, 0.01)
      model.optimizer.weight_decay: interval(0.001, 0.1)
      datamodule.train_dataloader_cfg.batch_size: choice(2, 4, 8, 16)
      trainer.max_epochs: range(10, 50, step=10)
```

### Search Space Types

| Type          | Syntax                          | Example                          |
| ------------- | ------------------------------- | -------------------------------- |
| Continuous    | `interval(low, high)`           | `interval(0.0001, 0.01)`         |
| Log-scale     | `interval(low, high, log=true)` | `interval(1e-5, 1e-2, log=true)` |
| Integer range | `range(start, end, step)`       | `range(10, 100, step=10)`        |
| Categorical   | `choice(a, b, c)`               | `choice(adam, sgd, adamw)`       |

### Log-Scale for Learning Rates

Learning rates are typically searched on a log scale:

```yaml
hydra:
  sweeper:
    params:
      model.optimizer.lr: interval(1e-5, 1e-2, log=true)
```

## Configuring the Sweeper

Configure the Optuna sweeper in your config:

```yaml
hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.OptunaSweeper
    study_name: my_optimization
    direction: minimize
    n_trials: 100

    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
```

Common samplers: `TPESampler` (default), `CmaEsSampler`, `RandomSampler`, `GridSampler`.

## Objective Metrics

By default, Optuna optimizes `val/loss`. To optimize a different metric:

```yaml
hydra:
  sweeper:
    direction: maximize
```

## Parallel Trials

Run multiple trials in parallel:

```bash
autoware-ml train --config-name my_task/my_model_optuna \
    --multirun \
    hydra.sweeper.n_jobs=4
```

!!! warning "GPU Considerations"
    Each parallel trial needs GPU memory. Ensure sufficient VRAM or use multiple GPUs.

Multiple workers can connect to the same study.

## Viewing Results

### MLflow Integration

All trials are logged to MLflow. Use the MLflow UI to compare trials as regular training runs.

## Best Practices

- **Start with wide ranges**, then narrow based on results
- **Use pruning** to avoid wasting compute on bad configurations
- **Fix unimportant parameters** to reduce search space

## Example: Full Optimization Config

```yaml title="configs/my_task/my_model_optuna.yaml"
# @package _global_
defaults:
  - /my_task/my_model_base
  - override /hydra/sweeper: optuna
  - _self_

hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.OptunaSweeper
    study_name: my_optimization
    direction: minimize
    n_trials: 100

    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42

    params:
      model.optimizer.lr: interval(1e-5, 1e-2, log=true)
      model.optimizer.weight_decay: interval(1e-4, 1e-1, log=true)
      datamodule.train_dataloader_cfg.batch_size: choice(2, 4, 8)

trainer:
  max_epochs: 20  # Shorter for faster trials
```

Run it:

```bash
autoware-ml train --config-name my_task/my_model_optuna --multirun
```

## Learn More

For detailed Optuna documentation, see the [official Optuna documentation](https://optuna.org/).
