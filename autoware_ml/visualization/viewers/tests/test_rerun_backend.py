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

"""Tests for the Rerun visualization backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
import json
from pathlib import Path
import socket
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import urllib.request
import uuid
import warnings

import numpy as np

from autoware_ml.visualization.viewers.contracts import VisualizationSessionConfig
from autoware_ml.visualization.viewers.events import (
    AnnotationContextEvent,
    AnnotationInfo,
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
    ViewSpec,
    VisualizationEvent,
)
from autoware_ml.visualization.viewers.rerun_backend import (
    RerunVisualizationBackend,
    _patch_class_id_array_protocol,
    _patch_time_int_array_protocol,
    _verify_annotation_context_support,
)

_ANNOTATION_DESCRIPTOR = (
    "rerun.archetypes.AnnotationContext:rerun.components.AnnotationContext#context"
)
_LOAD_RERUN_MODULE = "autoware_ml.visualization.viewers.rerun_backend._load_rerun_module"


@dataclass
class _RerunCalls:
    """Record every call the fake ``rerun`` module receives."""

    init: tuple[str, Mapping[str, object]] | None = None
    serve_web: Mapping[str, object] | None = None
    #: Entity path, logged payload, and keyword arguments of every ``rr.log`` call.
    logs: list[tuple[str, object, Mapping[str, object]]] = field(default_factory=list)
    #: Timeline, sequence, and timestamp of every ``rr.set_time`` call.
    times: list[tuple[str, int | None, float | None]] = field(default_factory=list)
    #: Blueprint and send options of every ``rr.send_blueprint`` call.
    blueprints: list[tuple[object, Mapping[str, object]]] = field(default_factory=list)
    flushes: list[bool] = field(default_factory=list)
    visible_time_ranges: list[tuple[tuple[object, ...], Mapping[str, object]]] = field(
        default_factory=list
    )
    #: Application id, file path, and blueprint of every saved blueprint file.
    saved_blueprints: list[tuple[str, str, _FakeBlueprint]] = field(default_factory=list)

    def logged(self, path: str) -> Sequence[object]:
        """Return the payloads logged to one entity path."""
        return [payload for logged_path, payload, _ in self.logs if logged_path == path]


class _FakeArrowArray:
    """Expose only the ``to_pylist`` surface the compatibility probe uses."""

    def __init__(self, payload: object) -> None:
        """Wrap one payload."""
        self._payload = payload

    def to_pylist(self) -> object:
        """Return the recorded payload."""
        return self._payload


class _FakeComponentBatch:
    """Mimic one Rerun component batch."""

    def __init__(self, descriptor: str, payload: object) -> None:
        """Store the descriptor and payload of the batch."""
        self._descriptor = descriptor
        self._payload = payload

    def component_descriptor(self) -> str:
        """Return the component descriptor string."""
        return self._descriptor

    def as_arrow_array(self) -> _FakeArrowArray:
        """Return the payload wrapped as an arrow-like array."""
        return _FakeArrowArray(self._payload)


class _FakeAnnotationContext:
    """Mimic ``rr.AnnotationContext`` including its serialized payload."""

    #: When true, mimic the serialization failure that silently drops legends.
    serializes_empty = False

    def __init__(self, context: Sequence[Mapping[str, object]]) -> None:
        """Store the annotation infos of the context."""
        self.context = context

    def as_component_batches(self) -> Sequence[_FakeComponentBatch]:
        """Return an indicator batch plus the annotation-context batch."""
        entries = [{"class_id": info["id"], "label": info["label"]} for info in self.context]
        payload = [[]] if type(self).serializes_empty else [entries]
        return (
            _FakeComponentBatch("rerun.components.AnnotationContextIndicator", [None]),
            _FakeComponentBatch(_ANNOTATION_DESCRIPTOR, payload),
        )


class _FakeImage:
    """Mimic ``rr.Image`` and its JPEG compression."""

    def __init__(self, image: np.ndarray) -> None:
        """Store the raw image."""
        self.image = image

    def compress(self, *, jpeg_quality: int) -> tuple[str, Mapping[str, object]]:
        """Return a stand-in for the encoded image."""
        return ("EncodedImage", {"image": self.image, "jpeg_quality": jpeg_quality})


class _FakeRecording:
    """Mimic the global data recording and record its flushes."""

    def __init__(self, calls: _RerunCalls) -> None:
        """Record into ``calls``."""
        self._calls = calls

    def flush(self, *, blocking: bool) -> None:
        """Record one flush."""
        self._calls.flushes.append(blocking)


class _Part(dict):
    """A recorded blueprint part that also accepts attributes such as ``id``."""


class _FakeBlueprint(_Part):
    """A recorded blueprint that can also be saved to a file."""

    def __init__(self, calls: _RerunCalls, /, **items: object) -> None:
        """Record the blueprint items and where saved files are reported."""
        super().__init__(**items)
        self._calls = calls

    def save(self, application_id: str, path: str) -> None:
        """Write a stand-in blueprint file and record the save."""
        Path(path).write_bytes(f"blueprint:{application_id}".encode())
        self._calls.saved_blueprints.append((application_id, path, self))


class _FakeTimeRangeBoundary:
    """Mimic ``rrb.TimeRangeBoundary``."""

    @staticmethod
    def cursor_relative(**kwargs: object) -> Mapping[str, object]:
        """Return a recorded cursor-relative boundary."""
        return {"kind": "cursor_relative", **kwargs}


class _FakeBlueprintModule:
    """Mimic ``rerun.blueprint`` by recording every part as a dictionary."""

    TimeRangeBoundary = _FakeTimeRangeBoundary

    def __init__(self, calls: _RerunCalls) -> None:
        """Record into ``calls``."""
        self._calls = calls

    @staticmethod
    def _part(kind: str, *args: object, **kwargs: object) -> _Part:
        """Record one blueprint part."""
        return _Part(kind=kind, args=args, **kwargs)

    def Blueprint(self, *args: object, **kwargs: object) -> _FakeBlueprint:
        """Record the whole blueprint."""
        return _FakeBlueprint(self._calls, kind="Blueprint", args=args, **kwargs)

    def Spatial3DView(self, *args: object, **kwargs: object) -> _Part:
        """Record a 3D view."""
        return self._part("Spatial3DView", *args, **kwargs)

    def Spatial2DView(self, *args: object, **kwargs: object) -> _Part:
        """Record a 2D view."""
        return self._part("Spatial2DView", *args, **kwargs)

    def TimeSeriesView(self, *args: object, **kwargs: object) -> _Part:
        """Record a time series view."""
        return self._part("TimeSeriesView", *args, **kwargs)

    def TextLogView(self, *args: object, **kwargs: object) -> _Part:
        """Record a text log view."""
        return self._part("TextLogView", *args, **kwargs)

    def TextDocumentView(self, *args: object, **kwargs: object) -> _Part:
        """Record a text document view."""
        return self._part("TextDocumentView", *args, **kwargs)

    def Horizontal(self, *args: object, **kwargs: object) -> _Part:
        """Record a horizontal container."""
        return self._part("Horizontal", *args, **kwargs)

    def Vertical(self, *args: object, **kwargs: object) -> _Part:
        """Record a vertical container."""
        return self._part("Vertical", *args, **kwargs)

    def Tabs(self, *args: object, **kwargs: object) -> _Part:
        """Record a tab container."""
        return self._part("Tabs", *args, **kwargs)

    def ScalarAxis(self, *args: object, **kwargs: object) -> _Part:
        """Record a scalar axis."""
        return self._part("ScalarAxis", *args, **kwargs)

    def EntityBehavior(self, *args: object, **kwargs: object) -> _Part:
        """Record an entity behavior override."""
        return self._part("EntityBehavior", *args, **kwargs)

    def VisualizerOverrides(self, *args: object, **kwargs: object) -> _Part:
        """Record a visualizer override."""
        return self._part("VisualizerOverrides", *args, **kwargs)

    def VisibleTimeRange(self, *args: object, **kwargs: object) -> _Part:
        """Record a visible time range, also in the call log."""
        self._calls.visible_time_ranges.append((args, kwargs))
        return self._part("VisibleTimeRange", *args, **kwargs)

    def BlueprintPanel(self, *args: object, **kwargs: object) -> _Part:
        """Record the blueprint panel state."""
        return self._part("BlueprintPanel", *args, **kwargs)

    def SelectionPanel(self, *args: object, **kwargs: object) -> _Part:
        """Record the selection panel state."""
        return self._part("SelectionPanel", *args, **kwargs)

    def TimePanel(self, *args: object, **kwargs: object) -> _Part:
        """Record the time panel state."""
        return self._part("TimePanel", *args, **kwargs)


class _FakeTransformRelation:
    """Mimic ``rr.TransformRelation``."""

    ChildFromParent = "ChildFromParent"


class _FakeMediaType:
    """Mimic ``rr.MediaType``."""

    MARKDOWN = "text/markdown"


class _FakeRerun:
    """A fake ``rerun`` module that records every call it receives."""

    AnnotationContext = _FakeAnnotationContext
    TransformRelation = _FakeTransformRelation
    MediaType = _FakeMediaType

    def __init__(self, calls: _RerunCalls) -> None:
        """Record into ``calls``."""
        self._calls = calls
        self.blueprint = _FakeBlueprintModule(calls)

    def init(self, application_id: str, **kwargs: object) -> None:
        """Record the recording initialization."""
        self._calls.init = (application_id, kwargs)

    def serve_web(self, **kwargs: object) -> None:
        """Record the web viewer settings."""
        self._calls.serve_web = kwargs

    def get_global_data_recording(self) -> _FakeRecording:
        """Return the recording whose flushes are recorded."""
        return _FakeRecording(self._calls)

    def set_time(
        self, timeline: str, *, sequence: int | None = None, timestamp: float | None = None
    ) -> None:
        """Record one timeline update."""
        self._calls.times.append((timeline, sequence, timestamp))

    def log(self, path: str, payload: object, **kwargs: object) -> None:
        """Record one logged payload."""
        self._calls.logs.append((path, payload, kwargs))

    def send_blueprint(self, payload: object, **kwargs: object) -> None:
        """Record one streamed blueprint."""
        self._calls.blueprints.append((payload, kwargs))

    @staticmethod
    def AnnotationInfo(**kwargs: object) -> Mapping[str, object]:
        """Return the annotation info as a dictionary."""
        return kwargs

    @staticmethod
    def Image(image: np.ndarray) -> _FakeImage:
        """Return a fake image."""
        return _FakeImage(image)

    @staticmethod
    def Points3D(
        *args: object, **kwargs: object
    ) -> tuple[str, tuple[object, ...], Mapping[str, object]]:
        """Record a 3D point cloud."""
        return ("Points3D", args, kwargs)

    @staticmethod
    def Points2D(
        *args: object, **kwargs: object
    ) -> tuple[str, tuple[object, ...], Mapping[str, object]]:
        """Record 2D points."""
        return ("Points2D", args, kwargs)

    @staticmethod
    def LineStrips2D(
        *args: object, **kwargs: object
    ) -> tuple[str, tuple[object, ...], Mapping[str, object]]:
        """Record 2D line strips."""
        return ("LineStrips2D", args, kwargs)

    @staticmethod
    def Boxes3D(**kwargs: object) -> tuple[str, Mapping[str, object]]:
        """Record 3D boxes."""
        return ("Boxes3D", kwargs)

    @staticmethod
    def Transform3D(**kwargs: object) -> tuple[str, Mapping[str, object]]:
        """Record a 3D transform."""
        return ("Transform3D", kwargs)

    @staticmethod
    def Pinhole(**kwargs: object) -> tuple[str, Mapping[str, object]]:
        """Record a pinhole camera."""
        return ("Pinhole", kwargs)

    @staticmethod
    def Scalars(value: float) -> tuple[str, float]:
        """Record a scalar."""
        return ("Scalars", value)

    @staticmethod
    def SeriesPoints(**kwargs: object) -> tuple[str, Mapping[str, object]]:
        """Record series point styling."""
        return ("SeriesPoints", kwargs)

    @staticmethod
    def TextLog(text: str, **kwargs: object) -> tuple[str, str, Mapping[str, object]]:
        """Record a text log entry."""
        return ("TextLog", text, kwargs)

    @staticmethod
    def TextDocument(text: str, **kwargs: object) -> tuple[str, str, Mapping[str, object]]:
        """Record a text document."""
        return ("TextDocument", text, kwargs)

    @staticmethod
    def Clear(**kwargs: object) -> tuple[str, Mapping[str, object]]:
        """Record a clear."""
        return ("Clear", kwargs)


def _free_port() -> int:
    """Return a TCP port that is free right now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _zero_points(path: str) -> PointCloud3DEvent:
    """Build a two-point cloud at the origin on one entity path."""
    return PointCloud3DEvent(path=path, positions=np.zeros((2, 3), dtype=np.float32))


def _unit_box(path: str) -> Boxes3DEvent:
    """Build one unit box at the origin on one entity path."""
    return Boxes3DEvent(
        path=path,
        centers=np.zeros((1, 3), dtype=np.float32),
        sizes=np.ones((1, 3), dtype=np.float32),
        yaws=np.zeros((1,), dtype=np.float32),
    )


def _front_pinhole() -> PinholeEvent:
    """Build the pinhole of the front camera."""
    return PinholeEvent(
        path="scene/cameras/front",
        image_from_camera=np.eye(3, dtype=np.float32),
        resolution=(64, 36),
    )


def _multitask_scene_events() -> Sequence[VisualizationEvent]:
    """Return the events of one complete multitask frame with cameras and metrics."""
    return (
        _zero_points("scene/prediction/segmentation"),
        _zero_points("scene/ground_truth/segmentation"),
        _zero_points("scene/lidar/intensity"),
        _zero_points("scene/prediction/entropy"),
        _zero_points("scene/prediction/probability"),
        _front_pinhole(),
        Points2DEvent(
            path="scene/cameras/front/projected/ground_truth/segmentation",
            positions=np.zeros((2, 2), np.float32),
        ),
        LineStrips2DEvent(
            path="scene/cameras/front/projected/prediction/detections",
            strips=[np.zeros((2, 2), np.float32)],
        ),
        _unit_box("scene/prediction/detections"),
        _unit_box("scene/ground_truth/detections"),
        ScalarEvent(path="scene/metrics/detection/precision", value=1.0),
        ScalarEvent(path="scene/metrics/detection/recall", value=1.0),
        ScalarEvent(path="scene/metrics/detection/mean_best_iou", value=1.0),
        ScalarEvent(path="scene/metrics/detection/max_iou", value=1.0),
        ScalarEvent(path="scene/metrics/detection/mean_matched_iou", value=1.0),
        ScalarEvent(path="scene/metrics/detection/true_positives", value=20.0),
        ScalarEvent(path="scene/metrics/detection/false_positives", value=12.0),
        ScalarEvent(path="scene/metrics/detection/false_negatives", value=8.0),
    )


def _scene_view_ids(part: object) -> Sequence[uuid.UUID]:
    """Collect the ids of the 3D views nested in one recorded blueprint part."""
    if not isinstance(part, _Part):
        return []
    ids = [part.id] if part.get("kind") == "Spatial3DView" else []
    for child in part.get("args", ()):
        ids.extend(_scene_view_ids(child))
    return ids


class _RerunBackendTestCase(unittest.TestCase):
    """Install a fake ``rerun`` module that records every call it receives."""

    def setUp(self) -> None:
        """Route the lazy ``rerun`` import of the backend to the fake module."""
        self.calls = _RerunCalls()
        self.enterContext(patch(_LOAD_RERUN_MODULE, return_value=_FakeRerun(self.calls)))

    def build_backend(self, **config: object) -> RerunVisualizationBackend:
        """Build one Rerun backend wired to the fake module."""
        return RerunVisualizationBackend(
            VisualizationSessionConfig(web_port=9091, grpc_port=9877, wait=False, **config)
        )


class TestRecordingLifecycle(_RerunBackendTestCase):
    """The backend initializes, serves, steps, and flushes the recording."""

    def test_backend_initializes_and_serves_the_web_viewer(self) -> None:
        """The recording is initialized without spawning and served on the configured ports."""
        backend = self.build_backend()

        self.assertEqual(self.calls.init[0], "autoware-ml")
        self.assertIs(self.calls.init[1]["spawn"], False)
        self.assertEqual(
            self.calls.serve_web,
            {
                "open_browser": False,
                "web_port": 9091,
                "grpc_port": 9877,
                "server_memory_limit": "25%",
            },
        )
        self.assertEqual(
            backend.web_url,
            "http://localhost:9091?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9877%2Fproxy",
        )

    def test_backend_forwards_timeline_steps(self) -> None:
        """Steps go to the frame timeline and timestamps to the sensor-time timeline."""
        backend = self.build_backend()

        backend.set_step(3)
        backend.set_timestamp(12.5)

        self.assertEqual(self.calls.times, [("frame", 3, None), ("sensor_time", None, 12.5)])

    def test_backend_flushes_pending_data_before_returning(self) -> None:
        """Waiting flushes the recording with blocking semantics."""
        backend = self.build_backend()

        backend.wait_until_interrupted()

        self.assertEqual(self.calls.flushes, [True])


class TestEventTranslation(_RerunBackendTestCase):
    """Every neutral event is translated into the matching Rerun archetype."""

    def test_backend_logs_annotation_context_statically(self) -> None:
        """Legends must be static so one context covers every frame on the timeline."""
        backend = self.build_backend()

        backend.log_event(
            AnnotationContextEvent(
                path="detection3d",
                annotations=(AnnotationInfo(id=0, label="car", color=(255, 0, 0, 255)),),
            )
        )

        path, payload, kwargs = self.calls.logs[0]
        self.assertEqual(path, "detection3d")
        self.assertEqual(kwargs, {"static": True})
        self.assertIsInstance(payload, _FakeAnnotationContext)
        self.assertEqual(
            list(payload.context), [{"id": 0, "label": "car", "color": (255, 0, 0, 255)}]
        )

    def test_backend_uses_the_supported_scalars_api(self) -> None:
        """``rr.Scalar`` is deprecated since rerun 0.23, so ``rr.Scalars`` must be used."""
        backend = self.build_backend()

        backend.log_event(ScalarEvent(path="scene/metrics/detection/num_predictions", value=4.0))

        self.assertEqual(
            self.calls.logged("scene/metrics/detection/num_predictions"), [("Scalars", 4.0)]
        )

    def test_backend_translates_every_supported_event(self) -> None:
        """Each supported event kind is logged once and a scene blueprint follows."""
        backend = self.build_backend()

        backend.log_events(
            [
                ImageEvent(path="scene/cameras/front", image=np.zeros((4, 4, 3), dtype=np.uint8)),
                _zero_points("scene/lidar/solid"),
                Points2DEvent(
                    path="scene/cameras/front/overlay", positions=np.zeros((2, 2), np.float32)
                ),
                LineStrips2DEvent(
                    path="scene/cameras/front/boxes", strips=[np.zeros((2, 2), np.float32)]
                ),
                _unit_box("scene/prediction/detections"),
                Transform3DEvent(
                    path="scene/cameras/front",
                    translation=np.zeros((3,), dtype=np.float32),
                    rotation_matrix=np.eye(3, dtype=np.float32),
                ),
                _front_pinhole(),
                TextEvent(path="meta/sample", text="sample-1"),
                ClearEvent(path="scene/ground_truth"),
            ]
        )

        self.assertEqual(
            [payload[0] for _, payload, _ in self.calls.logs],
            [
                "EncodedImage",
                "Points3D",
                "Points2D",
                "LineStrips2D",
                "Boxes3D",
                "Transform3D",
                "Pinhole",
                "TextLog",
                "Clear",
            ],
        )
        self.assertTrue(self.calls.blueprints)

    def test_backend_compresses_images_for_remote_streaming(self) -> None:
        """Camera images are JPEG-compressed before they are streamed."""
        backend = self.build_backend()
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        backend.log_event(ImageEvent(path="scene/cameras/front", image=image))

        encoded = self.calls.logged("scene/cameras/front")[0]
        self.assertEqual(encoded[0], "EncodedImage")
        self.assertEqual(encoded[1]["jpeg_quality"], 95)
        self.assertIs(encoded[1]["image"], image)

    def test_backend_converts_yaw_to_a_z_axis_quaternion(self) -> None:
        """Box yaws become xyzw quaternions about the z axis."""
        backend = self.build_backend()

        backend.log_event(
            Boxes3DEvent(
                path="scene/prediction/detections",
                centers=np.zeros((1, 3), dtype=np.float32),
                sizes=np.ones((1, 3), dtype=np.float32),
                yaws=np.array([np.pi / 2], dtype=np.float32),
            )
        )

        _, payload, _ = self.calls.logs[0]
        np.testing.assert_allclose(
            payload[1]["quaternions"],
            [[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]],
            atol=1e-6,
        )

    def test_backend_logs_a_markdown_document(self) -> None:
        """A markdown document is logged with the markdown media type."""
        backend = self.build_backend()

        backend.log_event(
            TextDocumentEvent(
                path="calibration_status/status/verdict", text="# Prediction: CALIBRATED"
            )
        )

        path, payload, _ = self.calls.logs[0]
        self.assertEqual(path, "calibration_status/status/verdict")
        self.assertEqual(
            payload,
            ("TextDocument", "# Prediction: CALIBRATED", {"media_type": "text/markdown"}),
        )

    def test_backend_logs_a_plain_document_without_a_media_type(self) -> None:
        """A plain document is logged without a media type."""
        backend = self.build_backend()

        backend.log_event(TextDocumentEvent(path="notes", text="plain", markdown=False))

        _, payload, _ = self.calls.logs[0]
        self.assertEqual(payload, ("TextDocument", "plain", {}))

    def test_backend_rejects_unknown_events(self) -> None:
        """An object that is not a visualization event is refused."""
        backend = self.build_backend()

        with self.assertRaisesRegex(TypeError, "Unsupported visualization event"):
            backend.log_event(object())  # type: ignore[arg-type]


class TestBlueprints(_RerunBackendTestCase):
    """Neutral layouts are translated into Rerun blueprints."""

    def test_backend_builds_named_comparison_views_without_root_origins(self) -> None:
        """The multitask scene blueprint names every view and never uses the root origin."""
        backend = self.build_backend()

        backend.log_events(_multitask_scene_events())

        blueprint, options = self.calls.blueprints[-1]
        serialized = repr(blueprint)
        self.assertEqual(options, {"make_active": True, "make_default": True})
        for absent in ("Semantic", "Prediction · Intensity", "'origin': '/'"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, serialized)
        for present in (
            "Multi comparison",
            "GT · Multi",
            "Prediction · Multi",
            "Normalized entropy",
            "Probability",
            "front · GT",
            "front · Prediction",
            "scene/cameras/front/projected/ground_truth/segmentation",
            "scene/cameras/front/projected/prediction/detections",
            "3D IoU quality",
            "GT mean best IoU",
            "Frame max IoU",
            "Matched mean IoU (>=0.5)",
            "'/scene/cameras': {'kind': 'EntityBehavior', 'args': (), 'visible': False}",
            "'/scene/cameras': {'kind': 'EntityBehavior', 'args': (), 'visible': True}",
            "'/scene/prediction/detections': ('Boxes3D', {'show_labels': True})",
            "Camera projections OFF",
            "Camera projections ON",
            "'auto_views': False",
            "'kind': 'TimePanel', 'args': (), 'expanded': False",
            "'kind': 'BlueprintPanel', 'args': (), 'expanded': True",
            "'kind': 'VisualizerOverrides', 'args': (['SeriesLines', 'SeriesPoints'],)",
            "('SeriesPoints', {'names': 'Precision', 'marker_sizes': 8.0})",
            "'row_shares': [2.0, 1.0]",
            "'kind': 'VisibleTimeRange', 'args': ('frame',)",
        ):
            with self.subTest(present=present):
                self.assertIn(present, serialized)
        self.assertLess(serialized.index("Prediction · Multi"), serialized.index("GT · Multi"))
        self.assertLess(serialized.index("front · Prediction"), serialized.index("front · GT"))
        # The single task comparison has two metric plots, each with an independent
        # Rerun time-range archetype.
        self.assertEqual(len(self.calls.visible_time_ranges), 2)

    def test_blueprint_panel_stays_collapsed_without_cameras(self) -> None:
        """Without cameras there is nothing to pick in the blueprint panel, so it stays shut."""
        backend = self.build_backend()

        backend.log_events([_zero_points("scene/prediction/segmentation")])

        blueprint, _ = self.calls.blueprints[-1]
        self.assertIn("'kind': 'BlueprintPanel', 'args': (), 'expanded': False", repr(blueprint))

    def test_backend_can_show_camera_geometry_initially(self) -> None:
        """Visible camera frustums open the camera tab and show the camera entities."""
        backend = self.build_backend(camera_frustums_visible=True)

        backend.log_events([_front_pinhole(), _zero_points("scene/prediction/segmentation")])

        blueprint, _ = self.calls.blueprints[-1]
        serialized = repr(blueprint)
        self.assertIn("'active_tab': 1", serialized)
        self.assertIn(
            "'/scene/cameras': {'kind': 'EntityBehavior', 'args': (), 'visible': True}",
            serialized,
        )

    def test_backend_accepts_adapter_defined_blueprint_without_path_knowledge(self) -> None:
        """A new adapter can own its layout without changing the Rerun backend."""
        backend = self.build_backend()

        backend.log_events(
            [
                _zero_points("scene/custom_task/output"),
                BlueprintEvent(
                    layout=LayoutGroup(
                        kind="tabs",
                        name="Custom task",
                        children=(
                            ViewSpec(
                                kind="spatial3d",
                                name="Adapter output",
                                origin="scene/custom_task",
                                contents=("scene/custom_task/output",),
                            ),
                        ),
                    ),
                    make_active=True,
                ),
            ]
        )

        self.assertEqual(len(self.calls.blueprints), 1)
        self.assertIn("Custom task", repr(self.calls.blueprints[0][0]))
        self.assertIn("scene/custom_task/output", repr(self.calls.blueprints[0][0]))

        backend.log_events([_zero_points("scene/custom_task/second_output")])

        self.assertEqual(len(self.calls.blueprints), 1)

    def test_backend_builds_a_text_document_view(self) -> None:
        """A text document view spec becomes a Rerun text document view."""
        backend = self.build_backend()

        backend.log_event(
            BlueprintEvent(
                layout=ViewSpec(
                    kind="text_document",
                    name="Calibration verdict",
                    origin="calibration_status/status/verdict",
                    contents=("calibration_status/status/verdict",),
                ),
                auto_views=False,
            )
        )

        self.assertEqual(len(self.calls.blueprints), 1)
        self.assertIn("TextDocumentView", repr(self.calls.blueprints[0][0]))
        self.assertIn("Calibration verdict", repr(self.calls.blueprints[0][0]))


class TestSyncedLayouts(_RerunBackendTestCase):
    """The backend serves the synced comparison page only when it is enabled."""

    def test_backend_publishes_synced_layouts_when_enabled(self) -> None:
        """Every side, comparison, and camera state is saved and published to the page."""
        blueprint_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(tempfile, "mkdtemp", return_value=str(blueprint_dir)))
        sync_port = _free_port()
        backend = self.build_backend(sync_views=True, sync_port=sync_port)
        self.assertIsNotNone(backend._synced_server)
        self.addCleanup(backend._synced_server.close)

        backend.log_events(_multitask_scene_events())

        saved = self.calls.saved_blueprints
        names = sorted(Path(path).name for _, path, _ in saved)
        self.assertEqual(
            names,
            sorted(
                ["left-off.rbl.partial", "left-on.rbl.partial"]
                + [
                    f"right-{key}-{state}.rbl.partial"
                    for key in ("gt", "intensity", "entropy", "probability")
                    for state in ("off", "on")
                ]
            ),
        )
        self.assertEqual(
            sorted(path.name for path in blueprint_dir.iterdir()),
            sorted(name.removesuffix(".partial") for name in names),
        )
        self.assertEqual({application_id for application_id, _, _ in saved}, {"autoware-ml"})

        left_ids = {
            view_id
            for _, path, part in saved
            if "left-" in path
            for view_id in _scene_view_ids(part)
        }
        right_ids = {
            view_id
            for _, path, part in saved
            if "right-" in path
            for view_id in _scene_view_ids(part)
        }
        self.assertEqual(len(left_ids), 1, "both camera states of the left side keep one view id")
        self.assertEqual(len(right_ids), 1, "every comparison of the right side keeps one view id")
        self.assertNotEqual(left_ids, right_ids)

        # The streamed blueprint of the plain viewer page is unchanged.
        self.assertEqual(len(self.calls.blueprints), 1)

        with urllib.request.urlopen(f"http://127.0.0.1:{sync_port}/config.json") as response:
            config = json.load(response)
        self.assertEqual(config["version"], 1)
        self.assertEqual(config["timeline"], "frame")
        self.assertEqual(config["grpc_port"], 9877)
        self.assertEqual(config["camera_states"], ["off", "on"])
        self.assertEqual(config["initial_comparison"], "gt")
        self.assertEqual(config["initial_camera_state"], "off")
        self.assertEqual(
            [comparison["key"] for comparison in config["comparisons"]],
            ["gt", "intensity", "entropy", "probability"],
        )
        self.assertEqual(config["left"], {"off": "left-off.rbl", "on": "left-on.rbl"})
        self.assertEqual(
            config["right"]["gt"], {"off": "right-gt-off.rbl", "on": "right-gt-on.rbl"}
        )
        with urllib.request.urlopen(
            f"http://127.0.0.1:{sync_port}/blueprints/right-gt-on.rbl?v=1"
        ) as response:
            self.assertEqual(response.read(), b"blueprint:autoware-ml")

    def test_backend_keeps_the_synced_page_off_by_default(self) -> None:
        """Without ``sync_views`` no page is served and no blueprint file is saved."""
        backend = self.build_backend()

        backend.log_events(_multitask_scene_events())

        self.assertIsNone(backend._synced_server)
        self.assertEqual(self.calls.saved_blueprints, [])


class TestAnnotationContextProbe(unittest.TestCase):
    """The compatibility probe fails loudly when legends would be dropped."""

    def test_verify_annotation_context_support_raises_when_legends_are_dropped(self) -> None:
        """A silently emptied AnnotationContext must fail loudly, not lose the legend."""
        with patch.object(_FakeAnnotationContext, "serializes_empty", True):
            with self.assertRaisesRegex(RuntimeError, "discarded a probe AnnotationContext"):
                _verify_annotation_context_support(_FakeRerun(_RerunCalls()))


class TestInstalledRerun(unittest.TestCase):
    """The installed ``rerun`` package serializes what the backend relies on."""

    def setUp(self) -> None:
        """Import the installed ``rerun`` package, or skip when it is unavailable."""
        try:
            self.rerun: ModuleType = import_module("rerun")
        except ImportError:
            self.skipTest("rerun is not installed")

    def test_installed_rerun_serializes_annotation_contexts(self) -> None:
        """Regression guard for the rerun-sdk 0.23.1 / NumPy 1.x ``__array__`` break.

        Rerun reports the failure as a warning and emits an empty context, so
        without the compatibility patch every class legend disappears from the
        viewer while the recording still looks healthy.
        """
        _patch_class_id_array_protocol()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            context = self.rerun.AnnotationContext(
                [
                    self.rerun.AnnotationInfo(id=0, label="car", color=(255, 0, 0, 255)),
                    self.rerun.AnnotationInfo(id=1, label="pedestrian", color=(0, 255, 0, 255)),
                ]
            )
            payload = next(
                batch.as_arrow_array().to_pylist()
                for batch in context.as_component_batches()
                if "AnnotationContext#" in str(batch.component_descriptor())
            )

        entries = payload[0]
        self.assertEqual([entry["class_id"] for entry in entries], [0, 1])
        self.assertEqual(
            [entry["class_description"]["info"]["label"] for entry in entries],
            ["car", "pedestrian"],
        )

    def test_installed_rerun_passes_the_compatibility_probe(self) -> None:
        """The patched installed ``rerun`` passes the annotation context probe."""
        _patch_class_id_array_protocol()

        _verify_annotation_context_support(self.rerun)

    def test_installed_rerun_serializes_cursor_relative_time_ranges(self) -> None:
        """The metric plot window must not be dropped by Rerun under NumPy 1.x."""
        _patch_time_int_array_protocol()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            view = self.rerun.blueprint.TimeSeriesView(
                origin="scene/metrics/detection",
                time_ranges=[
                    self.rerun.blueprint.VisibleTimeRange(
                        "frame",
                        start=self.rerun.blueprint.TimeRangeBoundary.cursor_relative(seq=-10),
                        end=self.rerun.blueprint.TimeRangeBoundary.cursor_relative(seq=10),
                    )
                ],
            )

        self.assertIsNotNone(view)


if __name__ == "__main__":
    unittest.main()
