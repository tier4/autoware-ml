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

- `--port`, `-p`: Custom port (default: 5000)
- `--db-path`: SQLite database path (default: `mlruns/mlflow.db`)

By default, the UI opens the shared tracking DB directly. To isolate a single experiment for sharing or inspection, export it first:

```bash
autoware-ml mlflow export --config-name segmentation3d/ptv3/nuscenes/voxel005_b16_50e_nuscenes_51m --export-dir /tmp/ptv3_mlflow
autoware-ml mlflow ui --db-path /tmp/ptv3_mlflow/mlflow.db
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
- **Artifacts**: Resolved config snapshots, run metadata, saved checkpoints, and deploy exports

Testing and deployment create separate runs in the same experiment and keep lineage to the source training run. Deployment runs also log exported ONNX and TensorRT artifacts.

## Using the UI

Experiments are named `<experiment_group_name>/<experiment_name>` from the two keys
of the task config, so the UI groups them by model, database version and vehicle rather
than by config path. Runs are named `<stage>_<experiment_name>_<experiment_uid>`
and tagged with metadata such as:

- `config_name`, with `task`, `model` and `config_variant` taken from its path
- `stage`
- `hydra_dir` and `artifact_dir`
- `checkpoint_path` for test and deploy, and `param_drift` for a resumed training run
- `git_sha` and `hostname`

Click a run to view:

- **Parameters**: All hyperparameters
- **Metrics**: Interactive training curves

## Comparing Runs

Select multiple runs and click **Compare** to view:

- Parallel coordinates plots for hyperparameter relationships
- Scatter plots comparing metrics
- Overlaid training curves

## Organizing Experiments

Experiments are named by the `experiment_group_name` and `experiment_name` keys of the task
config, not by its path. Leaves that should sit side by side in the UI share one
`experiment_group_name`, for example `bevfusion/${database.version}/j6gen2_base` for every
BEVFusion variant trained on that database, and differ in `experiment_name`.

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
experiment = mlflow.get_experiment_by_name("<experiment_group_name>/<experiment_name>")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.`val/loss` < 0.5",
    order_by=["metrics.`val/loss` ASC"]
)
```

## Storage Location

Every task config names its MLflow experiment through two keys,
`experiment_group_name` and `experiment_name`, and the experiment is
`<experiment_group_name>/<experiment_name>`. That one name keys the experiment, its
artifact root and the Hydra directories alike, for `train`, `test` and `deploy` and
whether a stage is launched through the CLI or its entrypoint directly. Two configs
that set the same two keys share one experiment. Runs are named
`<stage>_<experiment_name>_<experiment_uid>`.

Autoware-ML keeps both Hydra scratch outputs and MLflow-owned artifacts
under the same `mlruns/` root:

- `mlruns/mlflow.db`: Shared SQLite backend store
- `mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/hydra/`: Hydra scratch directory and local command outputs
- `mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/`: Run-owned MLflow artifacts
- `mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/run_metadata.json`: Run metadata used to preserve lineage across train, test, and deploy

To back up experiments completely, copy `mlruns/`.
