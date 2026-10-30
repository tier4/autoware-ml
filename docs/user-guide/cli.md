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

| Command          | Purpose                                                   |
| ---------------- | --------------------------------------------------------- |
| `train`          | Train models using PyTorch Lightning                      |
| `test`           | Evaluate models from a checkpoint                         |
| `visualize`      | Preview data and predictions in the visualization backend |
| `deploy`         | Export models to ONNX and TensorRT                        |
| `mlflow ui`      | Launch the MLflow tracking UI                             |
| `mlflow export`  | Export one experiment into its own MLflow store           |
| `session start`  | Start a managed background task                           |
| `session attach` | View live terminal output from a background task          |
| `session detach` | Disconnect raw tmux clients from a managed session        |
| `session ls`     | List managed background tasks                             |
| `session stop`   | Stop a managed background task                            |

## train

Train a model using the specified Hydra configuration.

```bash
autoware-ml train --config-name <config_path> [--weights <path> ...] [--skip-mismatched-weights] [--resume-checkpoint <path>] [--new-run] [hydra_overrides...]
```

**Arguments:**

- `--config-name`: Path to config
- `--weights`: One or more `.ckpt` paths for pretrained weight initialization (repeatable; later checkpoints overwrite earlier ones on overlapping keys). Use this for transfer learning, e.g. initializing a det3d encoder from a seg3d checkpoint. Mutually exclusive with `--resume-checkpoint`.
- `--skip-mismatched-weights`: With `--weights`, skip checkpoint tensors whose name matches a model tensor of a different shape instead of failing. Each skipped tensor is listed in a warning and keeps the model's own initialization. Use this when the architecture drifted from the checkpoint, e.g. a first layer that now takes more input channels. Without it, a shape mismatch is an error.
- `--resume-checkpoint`: Full Lightning checkpoint path to resume an interrupted training run from (restores model weights, optimizer state, and epoch). Training continues inside the checkpoint's source MLflow run: same run ID, metric curves, and checkpoint directory. Mutually exclusive with `--weights`.
- `--new-run`: With `--resume-checkpoint`, fork the training state into a new MLflow run instead of continuing the source run.

All remaining arguments are passed to Hydra as overrides. See [Configuration](configuration.md) for details.

Resuming with a modified configuration is supported: the current config is authoritative for training, callback settings such as early-stopping patience take their configured values, and MLflow keeps the originally logged params (they are immutable) - every changed key is reported in a startup warning and in the run's `param_drift` tag, and the config artifact records the exact resumed configuration.

**Examples:**

```bash
# Basic training
autoware-ml train --config-name <task>/<model>/<config>

# Initialize det3d encoder from a seg3d checkpoint
autoware-ml train --config-name <task>/<model>/<config> \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt

# Initialize from a checkpoint whose first layer has a different input width
autoware-ml train --config-name <task>/<model>/<config> \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt \
    --skip-mismatched-weights

# Resume an interrupted run
autoware-ml train --config-name <task>/<model>/<config> \
    --resume-checkpoint mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/last.ckpt

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

- `--release`: `vMAJOR.MINOR.PATCH` stamped into the ONNX metadata. Without it the artifacts
  are stamped `unversioned`.

The exports are written into the deploy MLflow run, under
`mlruns/<experiment_group_name>/<experiment_name>/<deploy_run_id>/artifacts/exports/`. The
output location is not configurable on its own, see [Deployment](deployment.md).

**Single-task example:**

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt
```

**Multi-head example:**

```bash
autoware-ml deploy \
    --config-name detection3d/ptv3/t4dataset/voxel012_b8_30e_t4dataset_122m \
    --weights mlruns/ptv3_seg3d/T4Dataset-base-v1.0.0/t4dataset/voxel012_b16_50e_t4dataset_122m/<run_id>/artifacts/checkpoints/best.ckpt \
    --weights mlruns/ptv3/T4Dataset-base-v1.0.0/t4dataset/voxel012_b8_30e_t4dataset_122m/<run_id>/artifacts/checkpoints/best.ckpt
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
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt
```

**Multi-head PTv3 detection example** (merge a pretrained PTv3 encoder checkpoint with a detection checkpoint):

```bash
autoware-ml test \
    --config-name detection3d/ptv3/t4dataset/voxel012_b8_30e_t4dataset_122m \
    --weights mlruns/ptv3_seg3d/T4Dataset-base-v1.0.0/t4dataset/voxel012_b16_50e_t4dataset_122m/<run_id>/artifacts/checkpoints/best.ckpt \
    --weights mlruns/ptv3/T4Dataset-base-v1.0.0/t4dataset/voxel012_b8_30e_t4dataset_122m/<run_id>/artifacts/checkpoints/best.ckpt
```

## visualize

Preview consecutive samples of a dataset split, and the predictions of a model,
in the visualization backend.

```bash
autoware-ml visualize \
    --config-name <config_path> \
    [--weights <path> ...] \
    [--mode auto|predictions|data] \
    [--split test|predict|val|train] \
    [--sample-index N] \
    [--max-samples N] \
    [--prediction-frequency-hz HZ] \
    [--backend rerun|noop] \
    [--device cpu|cuda|auto] \
    [--point-labels/--no-point-labels] \
    [--point-color-mode semantic|intensity|solid] \
    [--camera-frustums/--no-camera-frustums] \
    [--web-port PORT] \
    [--grpc-port PORT] \
    [--wait/--no-wait] \
    [--sync-views/--no-sync-views] \
    [--sync-port PORT] \
    [hydra_overrides...]
```

**Arguments:**

- `--config-name`: Path to config (same as used for training)
- `--weights`: Optional `.ckpt` paths whose parameters are merged into the model, as for `test` (repeatable). Every parameter must be covered by at least one checkpoint.

**Common options:**

- `--mode`: Preview mode. `auto` shows predictions when weights are given and the transformed data otherwise (default: `auto`). The data mode reads no weights, so it also previews configs whose weights are mandatory, such as fine-tuning ones.
- `--split`: Dataset split to preview. Its transform pipeline and collation run as in training or evaluation, and with weights the model's own preprocessing and decoding produce the predictions (default: `test`)
- `--sample-index`: First sample index to preview (default: `0`)
- `--max-samples`: Number of consecutive samples to preview (default: `1`)
- `--prediction-frequency-hz`: Also predict the lidar frames each record stores between two samples, at this rate. Prediction mode only; unset shows the samples alone (default: unset)
- `--backend`: Visualization backend (default: `rerun`)
- `--device`: Execution device for preview inference (default: `auto`, which uses CUDA when available)
- `--point-labels` / `--no-point-labels`: Log per-point text labels. Disabled by default because large semantic point clouds become slow when every point has text.
- `--point-color-mode`: Initially active point-cloud tab: semantic class, normalized LiDAR intensity, or solid geometry color (default: `semantic`). All available modes remain selectable in Rerun.
- `--camera-frustums` / `--no-camera-frustums`: Initial visibility of camera frustums and projected image planes in 3D views (default: hidden). When cameras are available, use the **Camera projections OFF** / **Camera projections ON** switch inside a 3D comparison to change them dynamically.
- `--web-port`: Rerun web viewer HTTP port (default: `9090`)
- `--grpc-port`: Rerun SDK gRPC port used by the web viewer proxy (default: `9876`)
- `--wait` / `--no-wait`: Keep the Rerun web server alive after logging (default: `--wait`)
- `--recording-id`: Optional explicit recording ID
- `--sync-views` / `--no-sync-views`: Also serve the synced comparison page, which shows the prediction and one selectable comparison in two viewers that share one eye: rotating, panning, zooming, or flying in either side moves both (default: `--no-sync-views`)
- `--sync-port`: HTTP port of the synced comparison page (default: `9091`)

Backend modes:

- `rerun`: serves the Rerun web viewer and logs the browser URL. Forward both `--web-port` and `--grpc-port` when running in Docker.
- `noop`: runs the preview path and drops all visualization events. Use this for smoke tests and CI.

Each sample is logged on the `frame` timeline at its dataset index, with its
lidar timestamp on `sensor_time`. Only the points of the current lidar frame are
shown: past sweeps merged in by the pipeline carry no annotation of their own.
Class names come from the taxonomy of the database; set
`+visualization.segmentation3d_class_names=[...]` or
`+visualization.detection3d_class_names=[...]` to override them.

The 3D views of the regular Rerun page each keep their own eye. To compare the
prediction with GT, intensity, entropy, or probability from exactly the same
viewpoint, add `--sync-views` and open `http://localhost:<sync-port>/`. The
page embeds two viewers side by side, mirrors the mouse and keyboard input of
one into the other, and offers the comparison, the camera projections, and the
frame cursor in its toolbar. Forward `--sync-port` together with the other two
ports when running in Docker; the page connects to `--grpc-port` directly, so
keep that port reachable from the browser. See
[Visualization Design](../framework/visualization.md#synced-comparison-page).

**Examples:**

Compare the predictions of a multi-task checkpoint with the ground truth from
one shared viewpoint:

```bash
autoware-ml visualize \
    --config-name multi/ptv3/t4dataset/voxel012_b8_100e_t4dataset_122m \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt \
    --split test \
    --max-samples 2 \
    --prediction-frequency-hz 10 \
    --sync-views \
    --sync-port 9091
```

Preview the transformed data only, without weights:

```bash
autoware-ml visualize \
    --config-name segmentation3d/ptv3/t4dataset/voxel012_b16_50e_t4dataset_122m \
    --mode data \
    --device cpu \
    --split train
```

**Frames between samples:**

`--prediction-frequency-hz 10` turns 1 Hz annotated samples into a 10 Hz
recording: each sample is logged at step `10 * i` and the nine lidar frames its
record stores after it fill the steps in between, with predictions only. GT is
logged on the samples and is never interpolated or held, and the frames in
between show no cameras because the record lists them for its own frame only.

`--max-samples` still counts samples, so three samples at 10 Hz are 30 frames.
The browser-side Rerun viewer has its own heap limit of roughly 2.3 GiB, which
is independent of `server_memory_limit`. A recording that exceeds it reports
`Reached memory limit of 2.3 GiB, dropping oldest data`, discards the start of
the stream, and then never becomes active, so the synced page waits forever on
`connecting the left viewer`. With six cameras and dense point clouds the
practical ceiling is about 20 frames. Prefer `--max-samples 2
--prediction-frequency-hz 10`, or lower the frequency to keep more samples in
the same budget, as in `--max-samples 3 --prediction-frequency-hz 5`.

**Opening the viewer:**

The regular Rerun page is logged on startup as
`http://localhost:<web-port>/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A<grpc-port>%2Fproxy`.
With `--sync-views`, the synced comparison page is served at
`http://localhost:<sync-port>/`. Append `?renderer=webgl` to either URL when
the default renderer misbehaves in the browser.

Both pages replay the whole recording when they connect, so a recording of
twenty camera-equipped frames can take a few minutes before the synced page
reports `synced` and its toolbar becomes usable. That wait is normal and is not
a failure.

Use the toolbar arrows, the frame slider, or the Rerun time panel to step
between frames one at a time.

Current visualization coverage:

- calibration status: camera image, projected lidar overlay, fused depth and intensity image, status labels, and confidence summary
- segmentation3d: GT/PD point clouds, semantic/intensity/solid coloring, and pointwise entropy computed from the predicted class probabilities
- detection3d: GT/PD boxes overlaid on point clouds with class and score labels, and line-and-point plots of same-class yaw-aware 3D IoU and pre-threshold best-overlap signals
- multi-task: combined segmentation and detection comparisons with optional calibrated camera frustums, and prediction-only frames between annotated samples

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
    train --config-name segmentation3d/ptv3/nuscenes/voxel005_b16_50e_nuscenes_51m
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
