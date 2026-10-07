# Preemptible training and mid-epoch resume

Shared slurm clusters requeue a job whenever a higher priority job needs its
GPUs, often without any grace period. This page describes the pieces that make
a long training run survive that: rotating intra-epoch checkpoints, atomic
checkpoint writes, a training dataloader that continues where it stopped, and a
launch script that resumes automatically.

## What a checkpoint restores

A full Lightning checkpoint carries the model, the optimizer and scheduler
states, the gradient scaler of mixed precision, the loop counters (epoch, global
step, batches of the current epoch) and the callbacks. On its own, Lightning
resumes a mid-epoch checkpoint by running the *remaining number* of batches from
a fresh iterator, so the head of the epoch is trained twice and its tail never.

The framework's training dataloader closes that gap. Its sampler derives the
order of every epoch from the seed and the epoch index alone, the loader counts
the batches it handed out, and the datamodule stores that count in the
checkpoint. On resume the loader skips exactly the batches already trained, so
the interrupted epoch finishes with the batches it still owed, in the order it
would have used. This holds for uniform shuffling, sequential order and the
repeat factor frame sampling, on one GPU or under DDP, as long as the batch size
and the number of GPUs are unchanged (otherwise the position is converted to the
nearest sample offset and a warning is logged).

Not restored: the running validation of an interrupted validation pass (it
restarts), and schedulers sized from the estimated number of steps
(`OneCycleLR`) recompute their horizon from the resumed run's devices, batch size
and accumulation, so keep those identical across relaunches.

## Checkpoint cadence

`/defaults/preemption` adds two components to a task config:

```yaml
defaults:
  - /tasks/multi/ptv3/voxel006_122m_t4dataset_base_pseudo
  - /defaults/amp_rulebook
  - /defaults/preemption
  - _self_
```

- `PeriodicCheckpoint` saves the full state `saves_per_epoch` times per epoch
  (default 4, the period is derived from the epoch length in optimizer steps)
  and at the end of every epoch, keeps the newest `keep_last` files (default 2)
  and points `last.ckpt` at the newest one. It also saves when the job receives
  SIGTERM.
- `AtomicCheckpointIO` writes every checkpoint to a temporary file and renames it
  into place, so a kill during a write never leaves a truncated file behind.

The `best.ckpt` of the default `model_checkpoint` callback is unchanged; the
default epoch-end `last.ckpt` writer is disabled because the periodic callback
owns that file.

## Resuming

```bash
# Explicit file, continues the checkpoint's MLflow run
autoware-ml train --config-name <config> --resume-checkpoint <run>/artifacts/checkpoints/last.ckpt

# Newest loadable checkpoint of a directory, or a fresh run when there is none
autoware-ml train --config-name <config> --resume-latest <run>/artifacts/checkpoints
```

`--resume-latest` tries `last.ckpt` first and then every other checkpoint from
the newest to the oldest, skipping files that do not load. It is meant for
relaunch scripts: the same command works for the first launch and for every
relaunch. `--weights` may be given alongside it; the weights initialize the
fresh start and are ignored once a checkpoint exists.

The training entrypoint writes the checkpoint directory of its run to the file
named by `AUTOWARE_ML_RUN_POINTER_FILE`, when that variable is set, so a
relaunch can find the run of its own job.

## Scheduler wrappers

A batch-scheduler wrapper (slurm or similar) needs three hooks, all generic:

- relaunch with `--resume-latest <checkpoint dir>`, so the first launch and every relaunch
  run the same command;
- set `AUTOWARE_ML_RUN_POINTER_FILE` to a per-job path, so a relaunch finds the
  checkpoint directory of its own run;
- set `AUTOWARE_ML_STOP_FILE` and create that file shortly before the time limit: every
  rank sees it at the next batch boundary, the run saves a checkpoint and exits cleanly, and
  the wrapper requeues the job. A signal handler cannot do this safely in a DDP run because
  it would need a collective. A preemption needs none of it: the relaunch resumes from the
  newest periodic file.

## Memory of the 6 cm foundation model

The PT-v3m3 model at 6 cm over the 122 m range holds up to about a million voxels per
frame (three frames: the current one, one past and one future sweep). With the
hardware-aligned widths, bf16 and activation checkpointing
(`model.encoder.activation_checkpointing: true`, set by `tasks/foundation/ptv3/common.yaml`) one frame peaks at 13 to 16 GB on a 96 GB GPU, and a stage 2 run on eight H100
used about 42 GB per GPU. The foundation stage configs use one frame per GPU with four
accumulation steps on eight GPUs for a global batch of 32. Checkpointing recomputes the
attention and MLP of every block in the backward pass, roughly a third more compute per
step; it is kept on so that a heavy frame cannot run out of memory and stall a resumed run
on the same frame.
