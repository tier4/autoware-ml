---
icon: lucide/terminal
---

# CLI Reference

Autoware-ML provides a unified command-line interface for all major workflows.

## Commands

| Command          | Purpose                              |
| ---------------- | ------------------------------------ |
| `train`          | Train models using PyTorch Lightning |
| `deploy`         | Export models to ONNX and TensorRT   |
| `mlflow-ui`      | Launch the MLflow tracking UI        |
| `create-dataset` | Generate dataset info files          |

## train

Train a model using the specified Hydra configuration.

```bash
autoware-ml train --config-name <config_path> [hydra_overrides...]
```

All arguments after `--config-name` are passed to Hydra as overrides. See [Configuration](configuration.md) for details.

**Examples:**

```bash
# Basic training
autoware-ml train --config-name my_task/my_model

# With overrides
autoware-ml train --config-name my_task/my_model \
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
    --config-name my_task/my_model \
    +checkpoint=path/to/checkpoint.ckpt
```

## mlflow-ui

Launch the MLflow tracking UI.

```bash
autoware-ml mlflow-ui [--port PORT] [--db-path PATH]
```

**Options:**

- `--port`, `-p`: Port for the UI (default: 5000)
- `--db-path`: SQLite database path (default: `mlruns/mlflow.db`)

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
