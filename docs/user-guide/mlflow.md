---
icon: lucide/line-chart
---

# MLflow

Autoware-ML uses [MLflow](https://mlflow.org/) for experiment tracking. Training, testing, and deployment all log to a shared local MLflow backend.

## Launching the UI

```bash
autoware-ml mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

**Options:**

- `--port`: Custom port (default: 5000)
- `--db-path`: SQLite database path (default: `mlruns/mlflow.db`)
- `--experiment-name`: Export only the specified experiment into an isolated store before launching the UI
- `--config-name`: Shorthand for `--experiment-name` using a task config path
- `--export-dir`: Directory for the extracted experiment store

By default, the UI opens the shared tracking DB directly. To isolate a single experiment for sharing or inspection, export it first:

```bash
autoware-ml mlflow export --config-name calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2 --export-dir /tmp/calibration_status_mlflow
autoware-ml mlflow ui --db-path /tmp/calibration_status_mlflow/mlflow.db
```

This creates an extracted MLflow store for that experiment and lets you open the UI against the exported DB.

For remote access, run MLflow directly with `--host 0.0.0.0`.

If MLflow is running on a remote machine, `0.0.0.0` only makes it listen on that machine. You still need to forward the port to your local machine. A common option is SSH port forwarding:

```bash
ssh -L 5000:localhost:5000 <remote-machine>
```

Then open:

```text
http://localhost:5000
```

## What Gets Logged

Training runs automatically log:

- **Metrics**: Loss curves, task-specific metrics, learning rate
- **Hyperparameters**: Complete Hydra configuration
- **Artifacts**: Hydra configs, run metadata, and saved checkpoints

Testing and deployment create separate runs in the same experiment and keep lineage to the source training run. Deployment runs also log exported ONNX and TensorRT artifacts.

## Using the UI

Experiments are named after config paths (e.g., `<task>/<model>/<config>`). Runs are named with stage and timestamp, and tagged with metadata such as:

- `config_name`
- `task`
- `model`
- `config_variant`
- `stage`
- `run_dir`
- `checkpoint_path` for test/deploy
- `git_sha`

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

Add custom tags for organization:

```yaml
logger:
  tags:
    experiment: baseline
    dataset: <dataset>
```

Or via CLI:

```bash
autoware-ml train --config-name <task>/<model>/<config> \
    +logger.tags.experiment=ablation_study
```

## Programmatic Access

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
experiment = mlflow.get_experiment_by_name("<task>/<model>/<config>")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.`val/loss` < 0.5",
    order_by=["metrics.`val/loss` ASC"]
)
```

## Storage Location

MLflow data is stored locally in `mlruns/`:

- `mlruns/mlflow.db`: Shared SQLite backend store
- `mlruns/<task>/<model>/<config>/<date>/<time>/`: Run-specific artifacts and Hydra outputs
- `mlruns/<task>/<model>/<config>/<date>/<time>/run_metadata.json`: Run metadata used to preserve lineage across train, test, and deploy

To backup, copy the entire `mlruns/` directory.
