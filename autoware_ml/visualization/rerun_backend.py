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

"""Rerun-backed visualization backend."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.common import POINT_COLOR_MODES
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    BlueprintEvent,
    Boxes3DEvent,
    ClearEvent,
    ImageEvent,
    LayoutGroup,
    LineStrips2DEvent,
    PinholeEvent,
    PointCloud3DEvent,
    Points2DEvent,
    ScalarEvent,
    TextDocumentEvent,
    TextEvent,
    Transform3DEvent,
    ViewOverride,
    ViewSpec,
    VisualizationEvent,
)
from autoware_ml.visualization.layouts import (
    build_calibration_blueprint,
    build_scene_blueprint,
    build_synced_scene_layouts,
)
from autoware_ml.visualization.synced_viewer import SyncedViewerManifest, SyncedViewerServer

logger = logging.getLogger(__name__)


def _load_rerun_module() -> Any:
    """Load the optional rerun dependency lazily."""
    return import_module("rerun")


def _patch_class_id_array_protocol() -> None:
    """Make ``rerun.datatypes.ClassId`` convertible to NumPy under NumPy 1.x.

    ``rerun-sdk`` 0.23.1 declares ``numpy>=1.23`` but its generated
    ``__array__`` implementations forward their ``copy`` argument straight into
    ``numpy.asarray``.  NumPy only accepts that keyword from 2.0 onwards, and
    NumPy 1.x invokes ``__array__`` without it, so ``copy`` keeps its ``None``
    default and the forwarded call raises ``TypeError``.  Rerun swallows that
    error while serializing, which makes every ``AnnotationContext`` collapse to
    an empty list and silently strips class legends from the viewer.

    Dropping the keyword when it is ``None`` restores serialization on NumPy 1.x
    and leaves NumPy 2.x behaviour untouched, since NumPy 2 always passes an
    explicit ``copy`` value.
    """
    class_id_module = import_module("rerun.datatypes.class_id")
    class_id_type = class_id_module.ClassId

    def __array__(self: Any, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Convert one class id to a NumPy array across NumPy 1.x and 2.x."""
        if copy is None:
            return np.asarray(self.id, dtype=dtype)
        return np.asarray(self.id, dtype=dtype, copy=copy)

    class_id_type.__array__ = __array__


def _patch_time_int_array_protocol() -> None:
    """Make Rerun blueprint time ranges serializable under NumPy 1.x.

    ``rerun-sdk`` 0.23.1 generates the same incompatible ``__array__``
    implementation for ``TimeInt`` as it does for ``ClassId``. Cursor-relative
    plot ranges exercise this datatype, so patch it before building the
    detection-statistics blueprint.
    """
    time_int_module = import_module("rerun.datatypes.time_int")
    time_int_type = time_int_module.TimeInt

    def __array__(self: Any, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Convert one time integer to a NumPy array across NumPy 1.x and 2.x."""
        if copy is None:
            return np.asarray(self.value, dtype=dtype)
        return np.asarray(self.value, dtype=dtype, copy=copy)

    time_int_type.__array__ = __array__


def _verify_annotation_context_support(rerun_module: Any) -> None:
    """Fail loudly when semantic legends cannot reach the viewer.

    Rerun reports serialization problems as warnings rather than exceptions, so
    a broken ``AnnotationContext`` would otherwise degrade silently into a
    viewer without class names or colors.
    """
    probe = rerun_module.AnnotationContext(
        [rerun_module.AnnotationInfo(id=0, label="probe", color=(255, 0, 0, 255))]
    )
    for batch in probe.as_component_batches():
        if "AnnotationContext#" not in str(batch.component_descriptor()):
            continue
        if batch.as_arrow_array().to_pylist() == [[]]:
            raise RuntimeError(
                "Rerun discarded a probe AnnotationContext, so class legends would be "
                "missing from the viewer. This indicates an incompatible "
                "rerun-sdk/numpy combination; expected rerun-sdk 0.23.1 with numpy 1.26.4."
            )
        return
    raise RuntimeError(
        "Rerun did not emit an AnnotationContext component for a probe legend; "
        "the installed rerun-sdk is incompatible with this backend."
    )


def _stable_view_id(identity: str) -> uuid.UUID:
    """Derive the Rerun view id of a layout identity.

    Rerun keys the per-view viewer state, including the eye of a 3D view, by
    the view id. Deriving it from the identity lets re-published layouts keep
    that state instead of resetting it with a fresh random id.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"autoware-ml/visualization/{identity}")


def _yaw_to_quaternions(yaws: np.ndarray) -> np.ndarray:
    """Convert z-axis yaw angles to quaternions in xyzw order."""
    half_angles = yaws * 0.5
    quaternions = np.zeros((yaws.shape[0], 4), dtype=np.float32)
    quaternions[:, 2] = np.sin(half_angles)
    quaternions[:, 3] = np.cos(half_angles)
    return quaternions


class _RerunVisualizationBackendBase:
    """Shared Rerun event translation."""

    def _initialize_recording(self, config: VisualizationSessionConfig, *, spawn: bool) -> None:
        """Initialize one Rerun recording."""
        self.timeline = config.timeline
        self.rr = _load_rerun_module()
        _patch_class_id_array_protocol()
        _patch_time_int_array_protocol()
        _verify_annotation_context_support(self.rr)
        self.rr.init(
            config.application_id,
            recording_id=config.recording_id,
            spawn=spawn,
        )
        self._fused_blueprint_sent = False
        if config.point_color_mode not in POINT_COLOR_MODES:
            raise ValueError("point color mode must be 'semantic', 'intensity', or 'solid'")
        self.point_color_mode = config.point_color_mode
        self.camera_frustums_visible = config.camera_frustums_visible
        self._observed_paths: set[str] = set()
        self._camera_paths: set[str] = set()
        self._scene_blueprint_signature: frozenset[str] = frozenset()
        self._explicit_blueprint_received = False
        self.application_id = config.application_id
        self._synced_server: SyncedViewerServer | None = None

    def wait_until_interrupted(self) -> None:
        """Return immediately because no viewer is served by default."""

    def set_step(self, step: int) -> None:
        """Advance the rerun timeline to one integer step."""
        self.rr.set_time(self.timeline, sequence=int(step))

    def set_timestamp(self, timestamp: float) -> None:
        """Set the source sensor time on a second, timestamp-valued timeline."""
        self.rr.set_time("sensor_time", timestamp=float(timestamp))

    def log_event(self, event: VisualizationEvent) -> None:
        """Translate one visualization event into rerun entities."""
        if isinstance(event, BlueprintEvent):
            self._send_blueprint(event)
            return

        if isinstance(event, AnnotationContextEvent):
            # Logged statically so one legend covers every frame on the timeline
            # instead of being resolved per step by latest-at semantics.
            self.rr.log(
                event.path,
                self.rr.AnnotationContext(
                    [
                        self.rr.AnnotationInfo(
                            id=annotation.id,
                            label=annotation.label,
                            color=annotation.color,
                        )
                        for annotation in event.annotations
                    ]
                ),
                static=True,
            )
            return

        if isinstance(event, ImageEvent):
            # Camera frames dominate remote recordings when logged as raw RGB.
            # Rerun preserves the same image entity and projection behavior
            # while JPEG compression substantially reduces memory and transfer.
            self.rr.log(event.path, self.rr.Image(event.image).compress(jpeg_quality=95))
            return

        if isinstance(event, PointCloud3DEvent):
            self.rr.log(
                event.path,
                self.rr.Points3D(
                    event.positions,
                    colors=event.colors,
                    labels=event.labels,
                    radii=event.radii,
                    show_labels=event.labels is not None,
                    class_ids=event.class_ids,
                ),
            )
            return

        if isinstance(event, Points2DEvent):
            self.rr.log(
                event.path,
                self.rr.Points2D(
                    event.positions,
                    colors=event.colors,
                    labels=event.labels,
                    radii=event.radii,
                    show_labels=event.labels is not None,
                    class_ids=event.class_ids,
                ),
            )
            return

        if isinstance(event, LineStrips2DEvent):
            self.rr.log(
                event.path,
                self.rr.LineStrips2D(
                    event.strips,
                    colors=event.colors,
                    labels=event.labels,
                    radii=event.radii,
                    show_labels=event.labels is not None,
                    class_ids=event.class_ids,
                ),
            )
            return

        if isinstance(event, Boxes3DEvent):
            self.rr.log(
                event.path,
                self.rr.Boxes3D(
                    centers=event.centers,
                    sizes=event.sizes,
                    quaternions=_yaw_to_quaternions(event.yaws),
                    colors=event.colors,
                    labels=event.labels,
                    radii=event.radii,
                    show_labels=event.labels is not None,
                    class_ids=event.class_ids,
                ),
            )
            return

        if isinstance(event, Transform3DEvent):
            self.rr.log(
                event.path,
                self.rr.Transform3D(
                    translation=event.translation,
                    mat3x3=event.rotation_matrix,
                    relation=self.rr.TransformRelation.ChildFromParent,
                ),
            )
            return

        if isinstance(event, PinholeEvent):
            width, height = event.resolution
            self.rr.log(
                event.path,
                self.rr.Pinhole(
                    image_from_camera=event.image_from_camera,
                    resolution=(width, height),
                ),
            )
            return

        if isinstance(event, ScalarEvent):
            self.rr.log(event.path, self.rr.Scalars(event.value))
            return

        if isinstance(event, TextEvent):
            self.rr.log(event.path, self.rr.TextLog(event.text, level=event.level))
            return

        if isinstance(event, TextDocumentEvent):
            document_kwargs = {}
            if event.markdown:
                document_kwargs["media_type"] = self.rr.MediaType.MARKDOWN
            self.rr.log(event.path, self.rr.TextDocument(event.text, **document_kwargs))
            return

        if isinstance(event, ClearEvent):
            self.rr.log(event.path, self.rr.Clear(recursive=event.recursive))
            return

        raise TypeError(f"Unsupported visualization event: {type(event)!r}")

    def _convert_view_override(self, override: ViewOverride) -> Any:
        """Translate one neutral entity override into Rerun components."""
        components = []
        if override.visible is not None:
            components.append(self.rr.blueprint.EntityBehavior(visible=override.visible))
        if override.show_labels is not None:
            components.append(self.rr.Boxes3D(show_labels=override.show_labels))
        if override.series_name is not None or override.marker_size is not None:
            components.extend(
                [
                    self.rr.blueprint.VisualizerOverrides(["SeriesLines", "SeriesPoints"]),
                    self.rr.SeriesPoints(
                        names=override.series_name,
                        marker_sizes=override.marker_size,
                    ),
                ]
            )
        if not components:
            raise ValueError(f"View override for {override.path!r} has no behavior")
        return components[0] if len(components) == 1 else components

    def _convert_layout(self, layout: ViewSpec | LayoutGroup) -> Any:
        """Translate a backend-neutral recursive layout into Rerun blueprint parts."""
        if isinstance(layout, LayoutGroup):
            children = [self._convert_layout(child) for child in layout.children]
            kwargs: dict[str, Any] = {}
            if layout.name is not None:
                kwargs["name"] = layout.name
            if layout.kind == "horizontal":
                if layout.shares is not None:
                    kwargs["column_shares"] = list(layout.shares)
                return self.rr.blueprint.Horizontal(*children, **kwargs)
            if layout.kind == "vertical":
                if layout.shares is not None:
                    kwargs["row_shares"] = list(layout.shares)
                return self.rr.blueprint.Vertical(*children, **kwargs)
            if layout.kind == "tabs":
                if layout.active is not None:
                    kwargs["active_tab"] = layout.active
                return self.rr.blueprint.Tabs(*children, **kwargs)
            raise ValueError(f"Unknown layout group kind: {layout.kind}")

        overrides = {
            override.path: self._convert_view_override(override) for override in layout.overrides
        }
        kwargs = {
            "name": layout.name,
            "origin": layout.origin,
            "contents": list(layout.contents),
        }
        if overrides:
            kwargs["overrides"] = overrides
        if layout.kind == "spatial3d":
            view = self.rr.blueprint.Spatial3DView(**kwargs)
        elif layout.kind == "spatial2d":
            view = self.rr.blueprint.Spatial2DView(**kwargs)
        elif layout.kind == "text_log":
            view = self.rr.blueprint.TextLogView(**kwargs)
        elif layout.kind == "text_document":
            view = self.rr.blueprint.TextDocumentView(**kwargs)
        elif layout.kind == "time_series":
            if layout.y_range is not None:
                kwargs["axis_y"] = self.rr.blueprint.ScalarAxis(
                    range=layout.y_range,
                    zoom_lock=True,
                )
            if layout.visible_time_range is not None:
                start, end = layout.visible_time_range
                kwargs["time_ranges"] = [
                    self.rr.blueprint.VisibleTimeRange(
                        layout.timeline or self.timeline,
                        start=self.rr.blueprint.TimeRangeBoundary.cursor_relative(seq=start),
                        end=self.rr.blueprint.TimeRangeBoundary.cursor_relative(seq=end),
                    )
                ]
            view = self.rr.blueprint.TimeSeriesView(**kwargs)
        else:
            raise ValueError(f"Unknown view kind: {layout.kind}")
        if layout.identity is not None:
            view.id = _stable_view_id(layout.identity)
        return view

    def _send_blueprint(self, event: BlueprintEvent) -> None:
        """Translate and publish one backend-neutral blueprint request."""
        parts = [self._convert_layout(event.layout)]
        if event.blueprint_panel_expanded is not None:
            parts.append(self.rr.blueprint.BlueprintPanel(expanded=event.blueprint_panel_expanded))
        if event.selection_panel_expanded is not None:
            parts.append(self.rr.blueprint.SelectionPanel(expanded=event.selection_panel_expanded))
        if event.time_panel_expanded is not None:
            parts.append(self.rr.blueprint.TimePanel(expanded=event.time_panel_expanded))
        blueprint_options: dict[str, Any] = {"auto_views": event.auto_views}
        if event.auto_layout is not None:
            blueprint_options["auto_layout"] = event.auto_layout
        send_options = {}
        if event.make_active is not None:
            send_options["make_active"] = event.make_active
        if event.make_default is not None:
            send_options["make_default"] = event.make_default
        self.rr.send_blueprint(
            self.rr.blueprint.Blueprint(*parts, **blueprint_options),
            **send_options,
        )

    def _send_fused_blueprint_if_needed(self, events: list[VisualizationEvent]) -> None:
        """Publish the neutral calibration layout once its images arrive."""
        if self._fused_blueprint_sent:
            return
        blueprint = build_calibration_blueprint(events)
        if blueprint is None:
            return
        self._send_blueprint(blueprint)
        self._fused_blueprint_sent = True

    def _send_scene_blueprint_if_needed(self) -> None:
        """Publish the neutral built-in scene layout as entities appear."""
        if self._explicit_blueprint_received:
            return
        scene_paths = frozenset(
            path for path in self._observed_paths if path == "scene" or path.startswith("scene/")
        )
        signature = scene_paths | frozenset(f"@camera:{path}" for path in self._camera_paths)
        if signature == self._scene_blueprint_signature:
            return
        blueprint = build_scene_blueprint(
            scene_paths,
            self._camera_paths,
            point_color_mode=self.point_color_mode,
            camera_frustums_visible=self.camera_frustums_visible,
            timeline=self.timeline,
        )
        if blueprint is None:
            return
        self._send_blueprint(blueprint)
        self._scene_blueprint_signature = signature
        self._publish_synced_layouts(scene_paths)

    def _publish_synced_layouts(self, scene_paths: frozenset[str]) -> None:
        """Save the per-side blueprints of the synced comparison page."""
        if self._synced_server is None:
            return
        layouts = build_synced_scene_layouts(
            scene_paths,
            self._camera_paths,
            point_color_mode=self.point_color_mode,
            camera_frustums_visible=self.camera_frustums_visible,
            timeline=self.timeline,
        )
        if layouts is None:
            return
        left = {
            state: self._save_synced_blueprint(f"left-{state}.rbl", layout)
            for state, layout in layouts.left.items()
        }
        right = {
            key: {
                state: self._save_synced_blueprint(f"right-{key}-{state}.rbl", layout)
                for state, layout in states.items()
            }
            for key, states in layouts.right.items()
        }
        version = self._synced_server.publish(
            SyncedViewerManifest(
                comparisons=tuple(
                    {"key": comparison.key, "label": comparison.label}
                    for comparison in layouts.comparisons
                ),
                camera_states=layouts.camera_states,
                initial_comparison=layouts.initial_comparison,
                initial_camera_state=layouts.initial_camera_state,
                left=left,
                right=right,
            )
        )
        logger.info("Synced comparison layouts published (version %s)", version)

    def _save_synced_blueprint(self, file_name: str, layout: ViewSpec | LayoutGroup) -> str:
        """Write one side layout as a blueprint file and return its file name."""
        assert self._synced_server is not None
        blueprint = self.rr.blueprint.Blueprint(
            self._convert_layout(layout),
            self.rr.blueprint.BlueprintPanel(expanded=False),
            self.rr.blueprint.SelectionPanel(expanded=False),
            self.rr.blueprint.TimePanel(expanded=False),
            auto_views=False,
            auto_layout=False,
        )
        path = self._synced_server.blueprint_dir / file_name
        partial = path.with_name(path.name + ".partial")
        blueprint.save(self.application_id, str(partial))
        # The page may fetch a file while it is being rewritten.
        os.replace(partial, path)
        return file_name

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """Log multiple visualization events."""
        event_list = list(events)
        if any(isinstance(event, BlueprintEvent) for event in event_list):
            self._explicit_blueprint_received = True
        for event in event_list:
            if not isinstance(event, (BlueprintEvent, ClearEvent)):
                self._observed_paths.add(event.path)
                if isinstance(event, PinholeEvent):
                    self._camera_paths.add(event.path)
            self.log_event(event)
        self._send_fused_blueprint_if_needed(event_list)
        self._send_scene_blueprint_if_needed()


class RerunVisualizationBackend(_RerunVisualizationBackendBase):
    """Emit visualization events through the Rerun web viewer."""

    def __init__(self, config: VisualizationSessionConfig) -> None:
        """Initialize one web-served Rerun recording."""
        self._initialize_recording(config, spawn=False)
        self.rr.serve_web(
            open_browser=False,
            web_port=config.web_port,
            grpc_port=config.grpc_port,
            server_memory_limit=config.server_memory_limit,
        )
        self.web_url = (
            f"http://localhost:{config.web_port}"
            f"?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A{config.grpc_port}%2Fproxy"
        )
        self.wait = config.wait
        logger.info("Rerun web viewer: %s", self.web_url)
        if config.sync_views:
            self._synced_server = SyncedViewerServer(
                port=config.sync_port,
                web_port=config.web_port,
                grpc_port=config.grpc_port,
                timeline=config.timeline,
                blueprint_dir=Path(tempfile.mkdtemp(prefix="autoware-ml-synced-viewer-")),
            )
            self._synced_server.start()
            logger.info("Synced comparison page: %s", self._synced_server.url)

    def wait_until_interrupted(self) -> None:
        """Flush all logged data, then keep the web viewer alive if requested."""
        recording = self.rr.get_global_data_recording()
        if recording is None:
            raise RuntimeError("Rerun did not create a global data recording.")
        recording.flush(blocking=True)
        if not self.wait:
            return
        logger.info("Rerun web viewer is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(3600)
