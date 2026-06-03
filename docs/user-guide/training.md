---
icon: lucide/graduation-cap
---

# Training

This guide covers training models with Autoware-ML.

## Basic Training

```bash
autoware-ml train --config-name <task>/<model>/<config>
```

Hydra scratch outputs are saved automatically under
`mlruns/<task>/<model>/<config>/<run_id>/hydra/`.
MLflow-owned artifacts are saved under
`mlruns/<task>/<model>/<config>/<run_id>/artifacts/`, including checkpoints,
resolved config snapshots, and run metadata.

Example:

```bash
autoware-ml train --config-name segmentation3d/ptv3/voxel005_102m_nuscenes
```

## Long-Running Training

For long jobs inside Docker or remote environments, prefer managed background sessions over `nohup`:

```bash
autoware-ml session start --name ptv3-train --cwd /workspace -- \
    train --config-name segmentation3d/ptv3/voxel005_102m_nuscenes
```

Later you can open the live viewer with `session attach`:

```bash
autoware-ml session attach --name ptv3-train
```

If you want to open the viewer immediately at startup, add `--attach` to `session start`. The viewer
is read-only. Press `Ctrl+C` to return to your shell while keeping the training process running.
To terminate the training job, use:

```bash
autoware-ml session stop --name ptv3-train
```

## Resuming Training

Continue from a checkpoint:

```bash
autoware-ml train --config-name <task>/<model>/<config> \
    --resume-checkpoint mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/last.ckpt
```

## Testing

Evaluate a trained checkpoint with the same task config:

```bash
autoware-ml test --config-name <task>/<model>/<config> \
    --weights mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/best.ckpt
```

The test command creates a dedicated MLflow run linked to the source training run.

## Common Overrides

Override any config value via command line:

```bash
# Training duration
autoware-ml train --config-name <task>/<model>/<config> \
    trainer.max_epochs=100

# Batch size and workers
autoware-ml train --config-name <task>/<model>/<config> \
    datamodule.train_dataloader_cfg.batch_size=16 \
    datamodule.train_dataloader_cfg.num_workers=8

# Learning rate
autoware-ml train --config-name <task>/<model>/<config> \
    model.optimizer.lr=0.0005

# Mixed precision
autoware-ml train --config-name <task>/<model>/<config> \
    trainer.precision=16-mixed
```

## Multi-GPU Training

By default `devices=auto` and `strategy=auto` are set. Lightning uses all available GPUs and selects DDP automatically when more than one GPU is detected. Single-GPU runs stay in the main process (no subprocess overhead).

```bash
# Use a subset of GPUs
autoware-ml train --config-name <task>/<model>/<config> \
    trainer.devices=[0,1]
```

To override the strategy, see [Lightning strategy documentation](https://lightning.ai/docs/pytorch/stable/extensions/strategy.html).

## Debugging

```bash
# Fast dev run (one batch)
autoware-ml train --config-name <task>/<model>/<config> \
    +trainer.fast_dev_run=true

# Limit batches
autoware-ml train --config-name <task>/<model>/<config> \
    +trainer.limit_train_batches=10 +trainer.limit_val_batches=5

# Anomaly detection
autoware-ml train --config-name <task>/<model>/<config> \
    +trainer.detect_anomaly=true
```

## Performance Tips

- Use mixed precision (`trainer.precision=16-mixed`) for speed
- Enable `pin_memory=true` for faster CPU->GPU transfer
- Use `persistent_workers=true` to avoid worker restart overhead
- Increase `num_workers` if many CPU cores are available
- Use SSD storage for large datasets
