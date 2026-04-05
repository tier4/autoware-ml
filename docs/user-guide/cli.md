---
icon: lucide/terminal
---

# CLI Reference

Autoware-ML provides a unified command-line interface for all major workflows.
Run the commands below either inside the Docker container or from a local
`pixi shell --environment default` / `pixi shell --environment dev`.

## Commands

| Command          | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| `train`          | Train models using PyTorch Lightning                 |
| `test`           | Evaluate models from a checkpoint                    |
| `deploy`         | Export models to ONNX and TensorRT                   |
| `mlflow ui`      | Launch the MLflow tracking UI                        |
| `mlflow export`  | Export one experiment into its own MLflow store      |
| `session start`  | Start a tmux-backed background session               |
| `session attach` | Attach to a tmux-backed background session           |
| `session detach` | Detach clients from a tmux-backed background session |
| `session ls`     | List tmux-backed background sessions                 |
| `session stop`   | Stop a tmux-backed background session                |
| `create-dataset` | Generate dataset info files                          |

## train

Train a model using the specified Hydra configuration.

```bash
autoware-ml train --config-name <config_path> [hydra_overrides...]
```

All arguments after `--config-name` are passed to Hydra as overrides. See [Configuration](configuration.md) for details.

**Examples:**

```bash
# Basic training
autoware-ml train --config-name <task>/<model>/<config>

# With overrides
autoware-ml train --config-name <task>/<model>/<config> \
    trainer.max_epochs=100 \
    model.optimizer.lr=0.0001
```

## deploy

Export a trained model to ONNX and TensorRT.

```bash
autoware-ml deploy --config-name <config_path> +checkpoint=<path> [options...]
```

**Arguments:**

- `--config-name`: Path to config (same as used for training)
- `+checkpoint`: Path to `.ckpt` checkpoint file

**Options:**

- `output_name=<name>`: Base name for output files
- `output_dir=<path>`: Output directory

**Example:**

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    +checkpoint=mlruns/<task>/<model>/<config>/<date>/<time>/checkpoints/best.ckpt
```

## test

Evaluate a trained model from a checkpoint.

```bash
autoware-ml test --config-name <config_path> +checkpoint=<path> [hydra_overrides...]
```

**Arguments:**

- `--config-name`: Path to config (same as used for training)
- `+checkpoint`: Path to `.ckpt` checkpoint file

**Example:**

```bash
autoware-ml test \
    --config-name <task>/<model>/<config> \
    +checkpoint=mlruns/<task>/<model>/<config>/<date>/<time>/checkpoints/best.ckpt
```

## mlflow ui

Launch the MLflow tracking UI.

```bash
autoware-ml mlflow ui [--port PORT] [--db-path PATH]
```

**Options:**

- `--port`, `-p`: Port for the UI (default: 5000)
- `--db-path`: SQLite database path (default: `mlruns/mlflow.db`)

## mlflow export

Export one experiment from the global MLflow store into an isolated store.

```bash
autoware-ml mlflow export [--db-path PATH] [--experiment-name NAME | --config-name CONFIG] [--export-dir PATH]
```

**Options:**

- `--db-path`: SQLite database path (default: `mlruns/mlflow.db`)
- `--experiment-name`: Exact MLflow experiment name to export
- `--config-name`: Export the experiment matching this task config path
- `--export-dir`: Directory for the extracted experiment store

## session start

Start a detached tmux session running an `autoware-ml` command.

```bash
autoware-ml session start --name <session_name> [--cwd PATH] [--attach] -- <autoware-ml command...>
```

**Example:**

```bash
autoware-ml session start --name calibration-status-train --cwd /workspace -- \
    train --config-name calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2
```

Use `--raw` to run a non-`autoware-ml` command in the managed session:

```bash
autoware-ml session start --name docs --raw --cwd /workspace -- zensical serve
```

When attached to a managed session, `Ctrl+C` detaches the client without stopping the training job. To stop the job, use `autoware-ml session stop`.

## session attach

Attach to an existing tmux session.

```bash
autoware-ml session attach --name <session_name>
```

## session detach

Detach clients from an existing tmux session.

```bash
autoware-ml session detach --name <session_name>
```

## session ls

List tmux sessions.

```bash
autoware-ml session ls
```

## session stop

Stop a tmux session.

```bash
autoware-ml session stop --name <session_name>
```

## create-dataset

Generate preprocessed info files for a dataset.

```bash
autoware-ml create-dataset \
    --dataset <name> \
    --task <task> \
    --root-path <path> \
    --out-dir <path> \
    [options...]
```

**Arguments:**

- `--dataset`: Dataset name
- `--task`: Task name (can be repeated for multiple tasks)
- `--root-path`: Dataset root directory
- `--out-dir`: Output directory for info files

**Options:**

- `--version`: Dataset version
- `--max-sweeps`: Max LiDAR sweeps to include
- `--info-prefix`: Prefix for output files

**Example:**

```bash
autoware-ml create-dataset \
    --dataset nuscenes \
    --task my_task \
    --root-path /path/to/dataset \
    --out-dir /path/to/output
```
