---
icon: lucide/chart-no-axes-combined
---

# Weights & Biases

Autoware-ML can use [Weights & Biases](https://wandb.ai/) as an experiment
tracking backend through PyTorch Lightning's `WandbLogger`.

Use W&B when you want managed experiment dashboards, run comparison, system
metrics, artifact lineage, and offline-to-online sync from GPU servers.

## Installation

W&B is included in the Autoware-ML Pixi/Docker environment:

```toml
wandb>=0.23.0,<0.26.0
```

If you are using an older container that was built before the dependency was
added, refresh the Pixi environment or install W&B into the active Pixi Python:

```bash
/opt/pixi/envs/autoware-ml-3316095427567709615/envs/dev/bin/python -m pip install "wandb>=0.23.0,<0.26.0"
```

## Hydra Configuration

MLflow remains the default logger. Switch one run to W&B with the Hydra logger
group override:

```bash
logger=wandb
```

The default W&B logger config is:

```yaml
logger:
  _target_: lightning.pytorch.loggers.WandbLogger
  project: ${oc.env:WANDB_PROJECT,autoware-ml}
  entity: ${oc.env:WANDB_ENTITY,null}
  save_dir: ${oc.env:WANDB_DIR,wandb_runs}
  offline: false
  log_model: false
  tags: []
```

Autoware-ML sets the run name, run ID, group, job type, and default tags at
runtime. `logger.log_model` is intentionally `false` because Autoware-ML logs
checkpoints as explicit W&B artifacts. This also keeps offline runs valid.

## Offline Runs

Offline mode records runs locally without an API key. This is useful on GPU
servers, Slurm jobs, and air-gapped development machines.

```bash
export WANDB_MODE=offline
export WANDB_PROJECT=autoware-ml-local
export WANDB_DIR=/workspace/wandb_runs
export WANDB_CACHE_DIR=/workspace/.wandb-cache
export WANDB_CONFIG_DIR=/workspace/.wandb-config
export WANDB_DATA_DIR=/workspace/.wandb-cache
```

Run training:

```bash
autoware-ml train \
  --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
  logger=wandb \
  model.init_checkpoint_path=/workspace/weights/rtv4_hgnetv2_s_coco.pth \
  trainer.max_epochs=3 \
  trainer.log_every_n_steps=5 \
  datamodule.max_train_samples=512 \
  datamodule.max_val_samples=128
```

At the end of the run, W&B prints a sync command:

```bash
wandb sync /workspace/wandb_runs/wandb/offline-run-<timestamp>-<run_id>
```

## Syncing Offline Runs

Login and sync to a specific team/project:

```bash
export WANDB_API_KEY=<your-api-key>
export WANDB_ENTITY=<team-or-user>
export WANDB_PROJECT=<project>
export WANDB_DIR=/workspace/wandb_runs
export WANDB_CACHE_DIR=/workspace/.wandb-cache
export WANDB_CONFIG_DIR=/workspace/.wandb-config
export WANDB_DATA_DIR=/workspace/.wandb-cache

wandb login "$WANDB_API_KEY"

wandb sync \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  /workspace/wandb_runs/wandb/offline-run-<timestamp>-<run_id>
```

Set the W&B cache/config/data directories before training and before sync. This
prevents W&B from staging artifacts under a user-specific path such as
`/root/.local/share/wandb`.

## Online Runs

Online runs update the W&B web UI live:

```bash
export WANDB_API_KEY=<your-api-key>
export WANDB_ENTITY=<team-or-user>
export WANDB_PROJECT=<project>
unset WANDB_MODE

wandb login "$WANDB_API_KEY"

autoware-ml train \
  --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
  logger=wandb \
  model.init_checkpoint_path=/workspace/weights/rtv4_hgnetv2_s_coco.pth
```

Use a service-account API key for servers, CI jobs, Launch agents, or Slurm
jobs. Human users should normally use their own W&B accounts for UI access.

## What Gets Logged

Training, testing, and deployment use the same tracking abstraction. For W&B,
Autoware-ML logs:

- **Metrics**: Lightning metrics such as train/validation loss, task metrics, and learning rate.
- **System metrics**: W&B records GPU, CPU, memory, disk, and network metrics.
- **Config artifacts**: Resolved Hydra config and CLI overrides.
- **Run metadata**: Stage, config name, run ID, source checkpoint, and local artifact paths.
- **Dataset metadata**: Lightweight datamodule identity fields such as data root, annotation files, image roots, and sample limits.
- **Model artifacts**: Checkpoints from training and deployment export outputs.

Offline mode does not provide a local web UI. Use `wandb sync` to view the run in
the hosted or self-hosted W&B app.

## Code Path

The Hydra logger group selects the backend:

```bash
logger=wandb
```

Autoware-ML then detects the logger target:

```python
from autoware_ml.utils.tracking_helpers import get_tracking_backend

backend = get_tracking_backend(cfg)
assert backend == "wandb"
```

Run context and logger fields are populated before Lightning instantiates the
logger:

```python
from autoware_ml.utils.tracking_helpers import (
    configure_logger,
    prepare_run_context,
    write_run_config_artifacts,
)

run_context = prepare_run_context(
    cfg,
    config_name,
    hydra_dir=work_dir,
    stage="train",
)
write_run_config_artifacts(cfg, run_context)
configure_logger(cfg, run_context)
trainer_logger = hydra.utils.instantiate(cfg.logger)
```

After the stage finishes, W&B artifacts are uploaded through the live W&B run:

```python
from autoware_ml.utils.tracking_helpers import log_stage_artifacts

trainer.fit(model, datamodule=datamodule)
log_stage_artifacts(cfg, trainer_logger, run_context, "train")
```

The W&B-specific artifact helper logs staged paths:

```python
import wandb

artifact = wandb.Artifact(name="model-<run_id>", type="model")
artifact.add_dir(".../artifacts/checkpoints")
run.log_artifact(artifact, aliases=["latest", "best"])
```

## Comparing Runs

In the W&B project UI:

1. Open the project.
2. Select two or more runs.
3. Compare metrics such as `val/loss`, `train/loss_epoch`, and task-specific metrics.
4. Inspect the artifacts tab for config, dataset metadata, run metadata, and checkpoint artifacts.

Example comparison from RT-DETRv4 Mapillary smoke runs:

| Run | Variant | Epoch 0 val/loss | Epoch 1 val/loss | Epoch 2 val/loss |
| --- | --- | ---: | ---: | ---: |
| `3d5ae611` | default augmentation | 33.01274 | 32.77001 | 32.11499 |
| `2fb24955` | no augmentation | 38.02945 | 37.29772 | 35.83192 |

## Storage Layout

Autoware-ML stages W&B artifacts under `wandb_runs/`:

- `wandb_runs/wandb/offline-run-*/`: W&B offline run data for later sync.
- `wandb_runs/<task>/<model>/<config>/<run_id>/hydra/`: Hydra output directory.
- `wandb_runs/<task>/<model>/<config>/<run_id>/artifacts/config/`: Resolved config and overrides.
- `wandb_runs/<task>/<model>/<config>/<run_id>/artifacts/datasets/`: Dataset metadata.
- `wandb_runs/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/`: Training checkpoints.
- `wandb_runs/<task>/<model>/<config>/<run_id>/artifacts/run_metadata.json`: Run metadata used for lineage.

