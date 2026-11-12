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
restarts).

### Resuming with another number of GPUs or batch size

A relaunch may use another number of GPUs, batch size or gradient accumulation, for
example when fewer GPUs are free. Three things keep the training consistent then:

- The dataloader converts the saved position to the new layout through the samples
  already trained, rounding down, so it repeats a few samples rather than skipping any.
- The trainer's position in the epoch is moved to the batch the dataloader continues
  at. Lightning counts the restored position in batches of the old layout; without
  the correction the epoch would run out of batches before Lightning saw its last
  one, so the end-of-epoch validation and the epoch's scheduler step were skipped and
  an epoch-wise learning rate lagged one epoch for the rest of the run.
- `SchedulerHorizon` (part of `/defaults/preemption`) refits schedulers sized from the
  number of optimizer steps of the run (`OneCycleLR` and every scheduler whose
  `total_steps` the framework fills). The checkpoint restores the horizon of the old
  layout: with a larger global batch the schedule would never reach its end, with a
  smaller one `OneCycleLR` raises once it steps past it. The callback rebuilds the
  scheduler for the new layout and moves it to the step that leaves exactly the
  optimizer steps the resumed run still takes, so it ends on the run's last step.

A schedule in steps is still traversed at the new rate from the resume on, and the
optimizer's hyperparameters are not rescaled: changing the global batch changes the
optimization, keeping it (fewer GPUs, more accumulation) does not.

## Checkpoint cadence

`/defaults/preemption` adds two components to a task config:

```yaml
defaults:
  - /tasks/multi/ptv3/t4dataset/base_voxel012_122m
  - /defaults/amp_bf16
  - /defaults/preemption
  - _self_
```

- `PeriodicCheckpoint` saves the full state `saves_per_epoch` times per epoch
  (default 4, the period is derived from the epoch length in optimizer steps)
  and at the end of every epoch, keeps the newest `keep_last` files (default 2)
  and points `last.ckpt` at the newest one. It also saves when the run is asked to stop
  through the stop file (see [Stopping a run by hand](#stopping-a-run-by-hand)), and on
  Ctrl-C or Lightning's SIGTERM exception. It does not save after other errors: one rank may
  fail alone, and a save needs every rank.
- `AtomicCheckpointIO` writes every checkpoint to a temporary file and renames it
  into place, so a kill during a write never leaves a truncated file behind.

The `best.ckpt` of the default `model_checkpoint` callback is unchanged; the
default epoch-end `last.ckpt` writer is disabled because the periodic callback
owns that file.

The pieces live in the `autoware_ml/resume/` package: `periodic_checkpoint.py`
(`PeriodicCheckpoint`), `atomic_checkpoint_io.py` (`AtomicCheckpointIO`),
`data_loader.py` (the resumable sampler and dataloader and the datamodule hooks
that store their position) and `checkpoint_locator.py` (`CheckpointLocator`
behind `--resume-latest`, and `RunPointer`).

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

## Stopping a run by hand

To free the GPUs and continue later without losing any step, stop the run through its stop
file instead of killing it:

```bash
# The file named by AUTOWARE_ML_STOP_FILE for this run
touch "$AUTOWARE_ML_STOP_FILE"
```

Every rank sees the file at the next batch boundary, `PeriodicCheckpoint` saves a full
checkpoint at that step and the run exits cleanly. A relaunch with `--resume-latest`
continues from exactly that step, with the dataloader at the same batch. A slurm wrapper that
maps a signal to the stop file (as for the time limit above) can be stopped the same way with
`scancel --signal=USR1 --batch <job id>`.

A plain `scancel` or `kill` sends SIGTERM instead. On one GPU, Lightning turns it into an
exception at a batch boundary and the callback still saves. With several GPUs, Lightning's
SIGTERM handler runs a collective inside the signal handler and can hang until the job is
killed, so nothing is saved: the relaunch resumes from the newest periodic checkpoint and
repeats up to `1 / saves_per_epoch` of an epoch. The same holds for a preemption without a
grace period and for a crash.

## Memory of a large PTv3 model at 6 cm

A large PTv3 model (`encoder_large`) at 6 cm over the 122 m range holds up to about a
million voxels per frame (three frames: the current one, one past and one future sweep). With
bf16, the MLP in fp32 and activation checkpointing (`model.encoder.activation_checkpointing:
true`), one frame peaks at 13 to 16 GB on a 96 GB GPU, and a joint segmentation and detection
run on eight H100 used about 42 GB per GPU, with one frame per GPU and four accumulation steps
for a global batch of 32. Checkpointing recomputes the
attention and MLP of every block in the backward pass, roughly a third more compute per
step; it is kept on so that a heavy frame cannot run out of memory and stall a resumed run
on the same frame.
