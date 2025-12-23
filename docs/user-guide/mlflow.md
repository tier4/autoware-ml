---
icon: lucide/line-chart
---

# MLflow

Autoware-ML uses [MLflow](https://mlflow.org/) for experiment tracking. Every training run automatically logs metrics, hyperparameters, and artifacts to a local MLflow backend.

## Launching the UI

```bash
autoware-ml mlflow-ui
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

**Options:**

- `--port`: Custom port (default: 5000)
- `--db-path`: Custom database path (default: `mlruns/mlflow.db`)

For remote access, run MLflow directly with `--host 0.0.0.0`.

## What Gets Logged

Each training run automatically logs:

- **Metrics**: Loss curves, task-specific metrics, learning rate
- **Hyperparameters**: Complete Hydra configuration

## Using the UI

Experiments are named after config paths (e.g., `tasks_my_task_my_model`). Each run represents one training execution.

Click a run to view:

- **Parameters**: All hyperparameters
- **Metrics**: Interactive training curves

## Comparing Runs

Select multiple runs and click **Compare** to view:

- Parallel coordinates plots for hyperparameter relationships
- Scatter plots comparing metrics
- Overlaid training curves

## Organizing Experiments

Experiments are named from config paths. Use meaningful config names for clarity.

Add tags for organization:

```yaml
logger:
  tags:
    experiment: baseline
    dataset: my_dataset
```

Or via CLI:

```bash
autoware-ml train --config-name my_task/my_model \
    +logger.tags.experiment=ablation_study
```

## Programmatic Access

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
experiment = mlflow.get_experiment_by_name("tasks_my_task_my_model")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.`val/loss` < 0.5",
    order_by=["metrics.`val/loss` ASC"]
)
```

## Storage Location

MLflow data is stored locally in `mlruns/`:

- `mlflow.db`: SQLite database (metrics, params)
- Date-organized directories: Run artifacts

To backup, copy the entire `mlruns/` directory.
