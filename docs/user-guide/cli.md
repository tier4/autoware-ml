---
icon: lucide/terminal
---

# CLI Reference

Autoware-ML provides a unified command-line interface for all major workflows.
Run the commands below either inside the Docker container or from a local
`pixi shell --environment default` / `pixi shell --environment dev`.
Bash completion is installed automatically by the Docker image build and by
`pixi run --environment <default|dev> setup-project` for local installs.

## Commands

| Command          | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `train`          | Train models using PyTorch Lightning                |
| `test`           | Evaluate models from a checkpoint                   |
| `deploy`         | Export models to ONNX and TensorRT                  |
| `mlflow ui`      | Launch the MLflow tracking UI                       |
| `mlflow export`  | Export one experiment into its own MLflow store     |
| `session start`  | Start a managed background task                     |
| `session attach` | View live terminal output from a background task    |
| `session detach` | Disconnect raw tmux clients from a managed session  |
| `session ls`     | List managed background tasks                       |
| `session stop`   | Stop a managed background task                      |
| `create-dataset` | Generate dataset info files                         |

## train

Train a model using the specified Hydra configuration.

```bash
autoware-ml train --config-name <config_path> [--weights <path> ...] [--resume-checkpoint <path>] [hydra_overrides...]
```

**Arguments:**

- `--config-name`: Path to config
- `--weights`: One or more `.ckpt` paths for pretrained weight initialization (repeatable; later checkpoints overwrite earlier ones on overlapping keys). Use this for transfer learning, e.g. initializing a det3d backbone from a seg3d checkpoint. Mutually exclusive with `--resume-checkpoint`.
- `--resume-checkpoint`: Full Lightning checkpoint path to resume an interrupted training run from (restores model weights, optimizer state, and epoch). Mutually exclusive with `--weights`.

All remaining arguments are passed to Hydra as overrides. See [Configuration](configuration.md) for details.

**Examples:**

```bash
# Basic training
autoware-ml train --config-name <task>/<model>/<config>

# Initialize det3d backbone from a seg3d checkpoint
autoware-ml train --config-name <task>/<model>/<config> \
    --weights mlruns/segmentation3d/<model>/<config>/<run_id>/artifacts/checkpoints/best.ckpt

# Resume an interrupted run
autoware-ml train --config-name <task>/<model>/<config> \
    --resume-checkpoint mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/last.ckpt

# With Hydra overrides
autoware-ml train --config-name <task>/<model>/<config> \
    trainer.max_epochs=100 \
    model.optimizer.lr=0.0001
```

## deploy

Export a trained model to ONNX and TensorRT.

```bash
autoware-ml deploy --config-name <config_path> --weights <path> [--weights <path> ...] [options...]
```

**Arguments:**

- `--config-name`: Path to config (same as used for training)
- `--weights`: One or more `.ckpt` paths whose parameters are merged into the
  export model. Pass once per checkpoint. Later checkpoints overwrite earlier
  ones on overlapping keys. Every parameter in the export model must be
  covered by at least one `--weights`; missing keys raise a runtime error
  listing what is uncovered.

**Options:**

- `output_name=<name>`: Base name for output files
- `output_dir=<path>`: Output directory

**Single-task example:**

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/best.ckpt
```

**Multi-head example:**

```bash
autoware-ml deploy \
    --config-name detection3d/ptv3/voxel012_122m_t4dataset_j6gen2 \
    --weights mlruns/segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt \
    --weights mlruns/detection3d/ptv3/voxel012_122m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt
```

## test

Evaluate a trained model from one or more checkpoints.

```bash
autoware-ml test --config-name <config_path> --weights <path> [--weights <path> ...] [hydra_overrides...]
```

**Arguments:**

- `--config-name`: Path to config (same as used for training)
- `--weights`: One or more `.ckpt` paths whose parameters are merged into the model for evaluation (repeatable; later checkpoints overwrite earlier ones). Every parameter must be covered by at least one checkpoint.

**Single-task example:**

```bash
autoware-ml test \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/best.ckpt
```

**Multi-head PTv3 detection example** (merge a pretrained PTv3 backbone
checkpoint with a detection checkpoint):

```bash
autoware-ml test \
    --config-name detection3d/ptv3/voxel012_122m_t4dataset_j6gen2 \
    --weights mlruns/segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt \
    --weights mlruns/detection3d/ptv3/voxel012_122m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt
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

Start a detached managed session running an `autoware-ml` command.

```bash
autoware-ml session start --name <session_name> [--cwd PATH] [--attach] -- <autoware-ml command...>
```

Managed sessions use a private tmux server internally, but the public workflow
is intentionally narrow: start a background task, view its live output, list
running sessions, and stop the task.

**Example:**

```bash
autoware-ml session start --name ptv3-train --cwd /workspace -- \
    train --config-name segmentation3d/ptv3/voxel005_51m_nuscenes
```

Use `--raw` to run a non-`autoware-ml` command in the managed session:

```bash
autoware-ml session start --name docs --raw --cwd /workspace -- zensical serve
```

Use `--attach` with `session start` to open the live viewer immediately after
startup. Use `session attach` later to view an already running task. In the
viewer, `Ctrl+C` returns to your shell without stopping the task. To terminate
the task, use `autoware-ml session stop`.

## session attach

Render a live terminal view of an existing managed session.

```bash
autoware-ml session attach --name <session_name>
```

This is a read-only viewer, not a tmux client. Press `Ctrl+C` to exit the
viewer while keeping the task running.

## session detach

Disconnect raw tmux clients from an existing managed session.

```bash
autoware-ml session detach --name <session_name>
```

Most users do not need this command because `autoware-ml session attach` does
not create a tmux client.

## session ls

List managed background sessions.

```bash
autoware-ml session ls
```

## session stop

Stop the tracked task and close its managed session.

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
