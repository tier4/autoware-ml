# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-neutral viewer layouts for the built-in task adapters.

This module owns task paths and presentation choices. Concrete backends only
translate :class:`BlueprintEvent` primitives, so adding a task-specific layout
does not require importing or modifying a backend SDK integration.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from autoware_ml.visualization.events import (
    BlueprintEvent,
    ImageEvent,
    LayoutGroup,
    TextDocumentEvent,
    ViewOverride,
    ViewSpec,
    VisualizationEvent,
)

#: Invisible corner points spanning every drawn entity of a frame, shared by
#: all scene views so they fit the same initial eye.
SCENE_EXTENT_PATH = "scene/meta/extent"

_DETECTION_METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "mean_best_iou": "GT mean best IoU",
    "max_iou": "Frame max IoU",
    "mean_matched_iou": "Matched mean IoU (>=0.5)",
    "true_positives": "True positives",
    "false_positives": "False positives",
    "false_negatives": "False negatives",
    "num_ground_truth": "GT boxes",
    "num_predictions": "Predicted boxes",
}

_SEGMENTATION_METRIC_LABELS = {
    "mean_confidence": "Mean confidence",
    "mean_entropy": "Mean normalized entropy",
    "num_points": "Points",
}


def build_calibration_blueprint(
    events: Iterable[VisualizationEvent],
) -> BlueprintEvent | None:
    """Build the raw/fused calibration image comparison when available."""
    fused_paths = [
        event.path
        for event in events
        if isinstance(event, ImageEvent) and event.path.endswith("/camera/fused")
    ]
    raw_paths = [
        event.path
        for event in events
        if isinstance(event, ImageEvent) and event.path.endswith("/camera/image")
    ]
    if not fused_paths:
        return None
    fused_path = fused_paths[0]
    children: list[ViewSpec] = []
    if raw_paths:
        raw_path = raw_paths[0]
        children.append(
            ViewSpec(
                kind="spatial2d",
                name="Raw camera",
                origin=raw_path,
                contents=(raw_path, f"{raw_path}/projected_points"),
            )
        )
    children.append(
        ViewSpec(
            kind="spatial2d",
            name="Fused camera",
            origin=fused_path,
            contents=(fused_path, f"{fused_path}/depth", f"{fused_path}/intensity"),
        )
    )
    images = LayoutGroup(kind="horizontal", children=tuple(children))
    verdict_paths = [
        event.path
        for event in events
        if isinstance(event, TextDocumentEvent) and event.path.endswith("/status/verdict")
    ]
    # The viewer's side and time panels are collapsed, as for the 3D layouts,
    # so the images and the verdict get the window; the time panel still
    # exposes the frame cursor when collapsed.
    panels = {
        "blueprint_panel_expanded": False,
        "selection_panel_expanded": False,
        "time_panel_expanded": False,
    }
    if not verdict_paths:
        return BlueprintEvent(layout=images, auto_views=False, **panels)
    # The verdict pane sits under the images so the status is read without
    # opening the log stream. Two headings need about a third of the height.
    verdict = ViewSpec(
        kind="text_document",
        name="Calibration verdict",
        origin=verdict_paths[0],
        contents=(verdict_paths[0],),
    )
    return BlueprintEvent(
        layout=LayoutGroup(kind="vertical", children=(images, verdict), shares=(0.7, 0.3)),
        auto_views=False,
        **panels,
    )


def _scene_view(
    observed_paths: set[str],
    camera_paths: Sequence[str],
    *,
    name: str,
    point_path: str | None,
    detection_path: str | None,
    camera_frustums_visible: bool,
) -> ViewSpec:
    """Build one 3D scene specification."""
    contents = []
    if point_path is not None and point_path in observed_paths:
        contents.append(point_path)
    if detection_path is not None and detection_path in observed_paths:
        contents.append(detection_path)
    if SCENE_EXTENT_PATH in observed_paths:
        contents.append(SCENE_EXTENT_PATH)
    overrides = []
    if camera_paths:
        contents.append("scene/cameras/**")
        overrides.append(ViewOverride(path="/scene/cameras", visible=camera_frustums_visible))
    if detection_path is not None and detection_path.startswith("scene/prediction/"):
        overrides.append(ViewOverride(path=f"/{detection_path}", show_labels=True))
    return ViewSpec(
        kind="spatial3d",
        name=name,
        origin="scene",
        contents=tuple(contents),
        overrides=tuple(overrides),
    )


def _series_overrides(paths: list[str], labels: dict[str, str]) -> tuple[ViewOverride, ...]:
    """Name and size the series of one statistics plot."""
    return tuple(
        ViewOverride(
            path=path,
            series_name=labels[path.rsplit("/", 1)[-1]],
            marker_size=8.0,
        )
        for path in paths
    )


def _segmentation_statistics_views(observed_paths: set[str], timeline: str) -> list[ViewSpec]:
    """Build the per-frame segmentation plots: confidence and point count."""
    metrics_root = "scene/metrics/segmentation"
    confidence_paths = [
        f"{metrics_root}/{name}"
        for name in ("mean_confidence", "mean_entropy")
        if f"{metrics_root}/{name}" in observed_paths
    ]
    count_paths = [
        f"{metrics_root}/num_points" for _ in (0,) if f"{metrics_root}/num_points" in observed_paths
    ]
    views = []
    if confidence_paths:
        views.append(
            ViewSpec(
                kind="time_series",
                name="Segmentation confidence",
                origin=metrics_root,
                contents=tuple(confidence_paths),
                overrides=_series_overrides(confidence_paths, _SEGMENTATION_METRIC_LABELS),
                y_range=(0.0, 1.0),
                visible_time_range=(-10, 10),
                timeline=timeline,
            )
        )
    if count_paths:
        views.append(
            ViewSpec(
                kind="time_series",
                name="Points per frame",
                origin=metrics_root,
                contents=tuple(count_paths),
                overrides=_series_overrides(count_paths, _SEGMENTATION_METRIC_LABELS),
                visible_time_range=(-10, 10),
                timeline=timeline,
            )
        )
    return views


def _statistics_views(observed_paths: set[str], timeline: str) -> list[ViewSpec]:
    """Build every per-frame statistics plot the observed entities support.

    Detection plots come first because they carry the match quality; the
    segmentation plots follow so a multi-task preview shows both rows' worth.
    """
    return _detection_statistics_views(observed_paths, timeline) + _segmentation_statistics_views(
        observed_paths, timeline
    )


def _detection_statistics_views(observed_paths: set[str], timeline: str) -> list[ViewSpec]:
    """Build compact IoU quality and match-count plot specifications.

    Without predictions only the ground-truth box count exists, so the count
    plot then reports counts instead of matches; it still gives data previews
    one curve to follow across frames.
    """
    metrics_root = "scene/metrics/detection"
    quality_paths = [
        f"{metrics_root}/{name}"
        for name in (
            "precision",
            "recall",
            "mean_best_iou",
            "max_iou",
            "mean_matched_iou",
        )
        if f"{metrics_root}/{name}" in observed_paths
    ]
    match_paths = [
        f"{metrics_root}/{name}"
        for name in ("true_positives", "false_positives", "false_negatives")
        if f"{metrics_root}/{name}" in observed_paths
    ]
    count_paths = match_paths + [
        f"{metrics_root}/{name}"
        for name in ("num_ground_truth", "num_predictions")
        if f"{metrics_root}/{name}" in observed_paths
    ]

    def series_overrides(paths: list[str]) -> tuple[ViewOverride, ...]:
        return _series_overrides(paths, _DETECTION_METRIC_LABELS)

    views = []
    if quality_paths:
        views.append(
            ViewSpec(
                kind="time_series",
                name="3D IoU quality",
                origin=metrics_root,
                contents=tuple(quality_paths),
                overrides=series_overrides(quality_paths),
                y_range=(0.0, 1.0),
                visible_time_range=(-10, 10),
                timeline=timeline,
            )
        )
    if count_paths:
        views.append(
            ViewSpec(
                kind="time_series",
                name="Detection matches" if match_paths else "Detection counts",
                origin=metrics_root,
                contents=tuple(count_paths),
                overrides=series_overrides(count_paths),
                visible_time_range=(-10, 10),
                timeline=timeline,
            )
        )
    return views


def _scene_task_name(observed_paths: set[str]) -> str:
    """Infer the display name of the task represented by one scene."""
    has_segmentation = any(path.endswith("/segmentation") for path in observed_paths)
    has_detection = any(path.endswith("/detections") for path in observed_paths)
    if has_segmentation and has_detection:
        return "Multi"
    if has_segmentation:
        return "Segmentation3D"
    if has_detection:
        return "Detection3D"
    return "Scene"


def _task_comparison(
    observed_paths: set[str],
    camera_paths: Sequence[str],
    *,
    task_name: str,
    camera_frustums_visible: bool,
    preferred_comparison: str,
    timeline: str,
) -> LayoutGroup:
    """Build a fixed prediction view with a selectable comparison view."""

    def scene_comparison(
        camera_visible: bool,
        *,
        layout_name: str,
    ) -> LayoutGroup:
        prediction_points = (
            "scene/prediction/segmentation"
            if "scene/prediction/segmentation" in observed_paths
            else "scene/lidar/solid"
        )
        ground_truth_points = (
            "scene/ground_truth/segmentation"
            if "scene/ground_truth/segmentation" in observed_paths
            else "scene/lidar/solid"
        )
        prediction_available = any(
            path in observed_paths
            for path in ("scene/prediction/segmentation", "scene/prediction/detections")
        )
        comparison_views: list[tuple[str, ViewSpec]] = []
        ground_truth_available = any(
            path in observed_paths
            for path in (
                "scene/ground_truth/segmentation",
                "scene/ground_truth/detections",
            )
        )
        if ground_truth_available:
            comparison_views.append(
                (
                    "GT",
                    _scene_view(
                        observed_paths,
                        camera_paths,
                        name=f"GT · {task_name}",
                        point_path=ground_truth_points,
                        detection_path="scene/ground_truth/detections",
                        camera_frustums_visible=camera_visible,
                    ),
                )
            )
        if "scene/lidar/intensity" in observed_paths:
            comparison_views.append(
                (
                    "Intensity",
                    _scene_view(
                        observed_paths,
                        camera_paths,
                        name="Intensity",
                        point_path="scene/lidar/intensity",
                        detection_path=None,
                        camera_frustums_visible=camera_visible,
                    ),
                )
            )
        if "scene/prediction/entropy" in observed_paths:
            comparison_views.append(
                (
                    "Entropy",
                    _scene_view(
                        observed_paths,
                        camera_paths,
                        name="Normalized entropy",
                        point_path="scene/prediction/entropy",
                        detection_path=None,
                        camera_frustums_visible=camera_visible,
                    ),
                )
            )
        if "scene/prediction/probability" in observed_paths:
            comparison_views.append(
                (
                    "Probability",
                    _scene_view(
                        observed_paths,
                        camera_paths,
                        name="Probability",
                        point_path="scene/prediction/probability",
                        detection_path=None,
                        camera_frustums_visible=camera_visible,
                    ),
                )
            )

        children: list[ViewSpec | LayoutGroup] = []
        if prediction_available:
            children.append(
                _scene_view(
                    observed_paths,
                    camera_paths,
                    name=f"Prediction · {task_name}",
                    point_path=prediction_points,
                    detection_path="scene/prediction/detections",
                    camera_frustums_visible=camera_visible,
                )
            )
        elif ground_truth_available and len(comparison_views) > 1:
            # A data preview has no prediction to anchor the left side, so the
            # ground truth takes that place and the remaining variants stay
            # selectable on the right, which keeps the two-pane reading.
            children.append(comparison_views.pop(0)[1])
        if comparison_views:
            comparison_names = [name for name, _ in comparison_views]
            active_comparison = (
                comparison_names.index(preferred_comparison)
                if preferred_comparison in comparison_names
                else 0
            )
            children.append(
                LayoutGroup(
                    kind="tabs",
                    children=tuple(view for _, view in comparison_views),
                    name="Comparison",
                    active=active_comparison,
                )
            )
        return LayoutGroup(
            kind="horizontal",
            children=tuple(children),
            name=layout_name,
        )

    if camera_paths:
        comparison = LayoutGroup(
            kind="tabs",
            children=(
                scene_comparison(False, layout_name="Camera projections OFF"),
                scene_comparison(True, layout_name="Camera projections ON"),
            ),
            name=f"{task_name} comparison",
            active=int(camera_frustums_visible),
        )
    else:
        comparison = scene_comparison(
            camera_frustums_visible,
            layout_name=f"{task_name} comparison",
        )
    statistics = _statistics_views(observed_paths, timeline)
    if not statistics:
        return comparison
    return LayoutGroup(
        kind="vertical",
        children=(
            comparison,
            LayoutGroup(
                kind="horizontal",
                children=tuple(statistics),
                name="Statistics",
            ),
        ),
        name=f"{task_name} comparison",
        shares=(2.0, 1.0),
    )


def _camera_tab(observed_paths: set[str], camera_paths: Sequence[str]) -> LayoutGroup:
    """Build per-camera GT/prediction views with persistent projections."""
    camera_tabs: list[ViewSpec | LayoutGroup] = []
    for path in camera_paths:
        camera_name = path.rsplit("/", 1)[-1]
        ground_truth_paths = sorted(
            observed_path
            for observed_path in observed_paths
            if observed_path.startswith(f"{path}/projected/ground_truth/")
        )
        prediction_paths = sorted(
            observed_path
            for observed_path in observed_paths
            if observed_path.startswith(f"{path}/projected/prediction/")
        )
        shared_paths = sorted(
            observed_path
            for observed_path in observed_paths
            if observed_path.startswith(f"{path}/projected/")
            and observed_path not in ground_truth_paths
            and observed_path not in prediction_paths
        )
        comparison_views = []
        if prediction_paths:
            comparison_views.append(
                ViewSpec(
                    kind="spatial2d",
                    name=f"{camera_name} · Prediction",
                    origin=path,
                    contents=(path, *shared_paths, *prediction_paths),
                )
            )
        if ground_truth_paths:
            comparison_views.append(
                ViewSpec(
                    kind="spatial2d",
                    name=f"{camera_name} · GT",
                    origin=path,
                    contents=(path, *shared_paths, *ground_truth_paths),
                )
            )
        if not comparison_views:
            comparison_views.append(
                ViewSpec(
                    kind="spatial2d",
                    name=camera_name,
                    origin=path,
                    contents=(path, *shared_paths),
                )
            )
        camera_tabs.append(
            comparison_views[0]
            if len(comparison_views) == 1
            else LayoutGroup(
                kind="horizontal",
                children=tuple(comparison_views),
                name=f"{camera_name} comparison",
            )
        )
    return LayoutGroup(kind="tabs", children=tuple(camera_tabs), name="Cameras")


def build_scene_blueprint(
    observed_paths: Iterable[str],
    camera_paths: Iterable[str],
    *,
    point_color_mode: str,
    camera_frustums_visible: bool,
    timeline: str,
) -> BlueprintEvent | None:
    """Build the complete built-in scene layout from neutral entity paths."""
    paths = set(observed_paths)
    cameras = sorted(camera_paths)
    scene_paths = {path for path in paths if path == "scene" or path.startswith("scene/")}
    if not scene_paths:
        return None

    task_name = _scene_task_name(scene_paths)
    tabs: list[tuple[str, ViewSpec | LayoutGroup]] = []
    has_task_output = any(path.endswith(("/segmentation", "/detections")) for path in scene_paths)
    if has_task_output:
        preferred_comparison = "Intensity" if point_color_mode == "intensity" else "GT"
        tabs.append(
            (
                task_name,
                _task_comparison(
                    paths,
                    cameras,
                    task_name=task_name,
                    camera_frustums_visible=camera_frustums_visible,
                    preferred_comparison=preferred_comparison,
                    timeline=timeline,
                ),
            )
        )
    if cameras:
        tabs.append(("Cameras", _camera_tab(paths, cameras)))
    if not tabs:
        # A custom adapter that uses the conventional scene root still gets a
        # useful view even before it supplies a tailored BlueprintEvent.
        tabs.append(
            (
                "Scene",
                ViewSpec(
                    kind="spatial3d",
                    name="Scene",
                    origin="scene",
                    contents=tuple(sorted(scene_paths)),
                ),
            )
        )

    return BlueprintEvent(
        layout=LayoutGroup(
            kind="tabs",
            children=tuple(tab for _, tab in tabs),
            active=0,
            name=f"{task_name} visualization",
        ),
        blueprint_panel_expanded=bool(cameras),
        selection_panel_expanded=False,
        time_panel_expanded=False,
        auto_layout=False,
        auto_views=False,
        make_active=True,
        make_default=True,
    )
