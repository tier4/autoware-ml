---
icon: lucide/graduation-cap
---

# Training

This guide covers training models with Autoware-ML.

## Basic Training

```bash
autoware-ml train --config-name my_task/my_model
```

Checkpoints and logs are saved automatically to timestamped directories under `mlruns/`.

## Resuming Training

Continue from a checkpoint:

```bash
autoware-ml train --config-name my_task/my_model \
    +checkpoint=path/to/checkpoint.ckpt
```

## Common Overrides

Override any config value via command line:

```bash
# Training duration
autoware-ml train --config-name my_task/my_model \
    trainer.max_epochs=100

# Batch size and workers
autoware-ml train --config-name my_task/my_model \
    datamodule.train_dataloader_cfg.batch_size=16 \
    datamodule.train_dataloader_cfg.num_workers=8

# Learning rate
autoware-ml train --config-name my_task/my_model \
    model.optimizer.lr=0.0005

# Mixed precision
autoware-ml train --config-name my_task/my_model \
    trainer.precision=16-mixed
```

## Multi-GPU Training

```bash
# All available GPUs
autoware-ml train --config-name my_task/my_model \
    trainer.devices=auto +trainer.strategy=ddp

# Specific GPUs
autoware-ml train --config-name my_task/my_model \
    trainer.devices=[0,1] +trainer.strategy=ddp
```

**Strategy options:**

- `ddp`: Default distributed training
- `ddp_find_unused_parameters_true`: For models with unused parameters
- `fsdp`: For very large models (parameter sharding)

## Debugging

```bash
# Fast dev run (one batch)
autoware-ml train --config-name my_task/my_model \
    +trainer.fast_dev_run=true

# Limit batches
autoware-ml train --config-name my_task/my_model \
    +trainer.limit_train_batches=10 +trainer.limit_val_batches=5

# Anomaly detection
autoware-ml train --config-name my_task/my_model \
    +trainer.detect_anomaly=true
```

## Performance Tips

- Use mixed precision (`trainer.precision=16-mixed`) for speed
- Enable `pin_memory=true` for faster CPU→GPU transfer
- Use `persistent_workers=true` to avoid worker restart overhead
- Increase `num_workers` if many CPU cores are available
- Use SSD storage for large datasets
