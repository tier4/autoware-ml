---
icon: lucide/binoculars
---

# Visualization Design

Autoware-ML includes an isolated visualization package intended for
prediction preview, calibration debugging, and dataset inspection. The initial
backend target is [Rerun](https://rerun.io/), but the task-facing API is kept
backend-neutral so future backends or offline exporters can reuse the same
task adapters.

## Design Goals

- Keep visualization out of model and transform core logic.
- Make visualization callable from tests, scripts, and CLI commands.
- Reuse one neutral scene-event layer across calibration status, segmentation3d,
  and detection3d.
- Keep the first integration narrow: sample-at-a-time preview rather than
  framework-wide logging during training.

## Layering

```mermaid
flowchart LR
    Task[Task code or test] --> Session[VisualizationSession]
    Session --> Adapter[Task adapter]
    Adapter --> Events[Visualization events]
    Events --> Backend[Visualization backend]
    Backend --> Rerun[Rerun SDK]
```

### 1. Visualization Session

`autoware_ml.visualization.session.VisualizationSession` is the public entrypoint. It
owns a backend and exposes task-oriented helpers:

- `log_calibration_status(...)`
- `log_segmentation3d(...)`
- `log_segmentation3d_data(...)`
- `log_detection3d(...)`
- `log_detection3d_data(...)`

This keeps later CLI or test integration simple and avoids leaking Rerun calls
into the rest of the framework.

### 2. Task Adapters

Task adapters turn native Autoware-ML data into neutral scene events:

- `visualization/calibration_status.py`
- `visualization/segmentation3d.py`
- `visualization/detection3d.py`

These adapters understand task semantics, for example:

- calibration camera intrinsics and lidar-to-camera transforms
- segmentation point labels and confidences
- detection decoded 3D boxes and class scores
- LiDAR point-cloud intensity and semantic point coloring
- pointwise class probabilities for uncertainty visualization

### 3. Visualization Events

The shared event layer in `visualization/events.py` defines primitives such as:

- `AnnotationContextEvent`
- `ImageEvent`
- `PointCloud3DEvent`
- `Points2DEvent`
- `Boxes3DEvent`
- `Transform3DEvent`
- `PinholeEvent`
- `ScalarEvent`
- `TextEvent`

This is the isolation boundary between task code and any concrete backend.

### 4. Backend

`visualization/rerun_backend.py` is the only place that knows the Rerun SDK.
It converts the neutral events into `rerun.Image`, `rerun.Points3D`,
`rerun.Boxes3D`, `rerun.Transform3D`, and related entities.

There is also a `NoOpVisualizationBackend` for disabled or test-only paths. The
Rerun backend is imported lazily, so the `noop` backend and the smoke tests
built on it stay usable without the Rerun SDK installed.

### Dependency compatibility

`rerun-sdk` 0.23.1 declares `numpy>=1.23`, but its generated `__array__`
implementations forward a `copy` argument that only NumPy 2.0 and later accept.
Under the pinned `numpy==1.26.4` this makes every `AnnotationContext` serialize
to an empty list, and because Rerun reports the failure as a warning rather than
an exception, class legends disappear from the viewer while the recording still
looks healthy.

`rerun_backend.py` therefore patches `rerun.datatypes.ClassId.__array__` to drop
the keyword when it is `None`, which restores NumPy 1.x support and leaves
NumPy 2.x behaviour untouched. Backend startup then logs a probe legend and
raises if it still serializes empty, so an incompatible dependency bump fails
loudly instead of quietly dropping every legend. Remove the patch once the
repository moves to NumPy 2.x or a Rerun release that fixes the conversion.

## Task Coverage

### Calibration Status

The initial calibration adapter can log:

- camera intrinsics
- lidar-to-camera extrinsics
- raw camera image
- projected lidar points overlaid on the image plane
- optional fused image preview
- optional 3D lidar points
- predicted and ground-truth calibration status
- prediction confidence and a readable status summary

The adapter takes the lidar-to-camera transform and the intrinsics of the shown
image as plain matrices. The calibration pipelines undistort every image before
any other transform, so a pinhole projection with those intrinsics places the
lidar points exactly where the fused depth channel puts them.

### Segmentation 3D

The segmentation adapter can log:

- semantic class legend
- point cloud positions
- predicted semantic labels as per-point colors
- optional ground-truth labels
- sample metadata and point counts
- optional mean confidence from `pred_probs`
- pointwise normalized entropy computed from `pred_probs`
  (`-sum(p * log(p)) / log(num_classes)`)
- selectable point coloring: semantic labels, LiDAR intensity, or a solid color

This matches the segmentation prediction contract, `Segmentation3DPredictions`,
which carries `pred_labels` and `pred_probs` for every scored point.

### Combined Detection and Segmentation

A combined detection and segmentation sample is logged below one `scene`
hierarchy. The Rerun blueprint presents ground truth and prediction side by
side, with the matching detection boxes overlaid on each point cloud.

The explicit scene layout contains:

- a **Semantic** comparison with GT and predicted pointwise classes
- an **Intensity** comparison using normalized LiDAR return intensity
- a **Geometry** comparison with a neutral point color
- an **Uncertainty** comparison between prediction entropy and predicted semantics
- a **Cameras** tab for the image streams, while every 3D comparison also
  contains optional calibrated camera frustums and image planes
- compact 3D IoU quality and match-count plots below GT/PD detection comparisons

The three point-cloud tabs make the coloring mode selectable in the viewer.
The session's `point_color_mode` only chooses which one is initially active.

Every scene view also lists `scene/meta/extent`, two invisible corner points
spanning the LiDAR points and every GT and predicted box of the frame. Rerun
fits the initial eye of a 3D view to the bounds of what that view shows, so
without the shared extent the GT and prediction views would start from slightly
different eyes whenever only one of them shows a box that sticks out of the
point cloud.

### Detection 3D

The detection adapter can log:

- semantic class legend
- optional lidar points
- decoded predicted 3D boxes
- optional ground-truth boxes
- class-colored boxes with class/score labels
- sample metadata and prediction/ground-truth counts
- the same point-cloud color modes, including LiDAR intensity from column 4
- same-class, yaw-aware 3D IoU matching at threshold 0.5, with per-frame
  true-positive, false-positive, false-negative, precision, recall, and mean
  matched-IoU scalars
- threshold-independent mean-best-GT and frame-maximum IoU scalars, so near
  misses remain visible even when no prediction reaches the matching threshold

Detection and segmentation use sibling `scene/prediction` and
`scene/ground_truth` entities. A frame with no detection annotations remains a
segmentation-only preview; no synthetic detection objects are created.

### Camera and timing behavior

Each available camera is logged as a transform, pinhole calibration, and image.
The Rerun blueprint places the camera frustums in the same 3D scene as the
LiDAR points and boxes, enabling live 3D camera views and projection inspection.
The backend JPEG-encodes RGB camera frames at quality 95 and broadcasts constant
point radii instead of repeating them per point. These transport optimizations
preserve scene geometry and calibration while keeping 10 Hz recordings usable
over an SSH tunnel.
Prediction box labels are visible by default and contain the class name and
confidence score. They can still be hidden dynamically with Rerun's **Show
labels** property when a dense set of proposals obscures the point cloud.
When calibrated cameras are present, every 3D comparison contains a prominent
**Camera projections OFF** / **Camera projections ON** tab switch. Switching
tabs immediately hides or shows the complete camera subtree, including both
frustums and projected image planes, without restarting inference. Rerun's eye
control beside the `cameras` subtree remains available for per-view adjustment.
`VisualizationSessionConfig.camera_frustums_visible` chooses the initially
active switch position. Pure LiDAR previews do not show the switch and keep the
Blueprint panel collapsed.
Every explicit 3D view uses `scene` as its origin. Automatic views are disabled,
and the Selection and Time panels start collapsed, so a root `/` text view or
stream is not shown in the working layout. The Time panel can be expanded when
the 10 Hz timeline needs to be scrubbed.
Camera overlays are pointwise LiDAR segmentation only; camera/pixel
segmentation is outside this scope.

Detection statistics are rendered as both line and point series. Point markers
keep a single GT keyframe visible and expose its exact value on hover. A
cursor-relative ±10-frame query window keeps the plot focused on the surrounding
GT anchors. Loading additional GT anchors extends the same plots into normal
time-series curves; frames without ground truth do not fabricate
statistics. The statistics row receives one third of the comparison height so
its axes and markers remain readable without displacing the primary point-cloud
views. Match-count axes auto-scale as later GT anchors add larger values, without
republishing the blueprint or resetting interactive camera visibility.

Before each timeline step, the dynamic LiDAR, prediction, ground-truth, camera,
and metadata entities are cleared recursively, so points from previous frames
cannot accumulate. Detection metrics retain their timestamped history for the
plots.

The point color mode accepts `semantic` (default), `intensity`, or `solid`. For
`intensity`, the fourth point feature is normalized per frame and rendered with
the shared scalar heatmap.

The detection adapter reads the `bboxes_3d` / `scores_3d` / `labels_3d` fields
of `Detection3DSamplePredictions`, and also accepts the `bboxes` / `scores` /
`labels` key set.

## Recommended Usage Pattern

A call site builds a session from its config and logs one frame at a time:

```python
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.session import VisualizationSession

session = VisualizationSession.from_config(
    VisualizationSessionConfig(
        backend="rerun",
        recording_id="calibration-debug",
        web_port=9090,
        grpc_port=9876,
    )
)
session.begin_frame(sample_index)
session.log_calibration_status(lidar_to_camera=..., image_from_camera=..., image=...)
```

That keeps the dependency on Rerun confined to one backend module. Use
`rerun` for browser-based Docker workflows, and `noop` for smoke tests that
only validate the preview pipeline.
