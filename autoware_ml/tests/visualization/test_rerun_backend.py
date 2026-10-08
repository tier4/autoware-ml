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

import json
import socket
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.events import (
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
)
from autoware_ml.visualization.rerun_backend import (
    RerunVisualizationBackend,
    _patch_class_id_array_protocol,
    _patch_time_int_array_protocol,
    _verify_annotation_context_support,
)

_ANNOTATION_DESCRIPTOR = (
    "rerun.archetypes.AnnotationContext:rerun.components.AnnotationContext#context"
)


class _FakeArrowArray:
    """Expose only the ``to_pylist`` surface the compatibility probe uses."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def to_pylist(self) -> Any:
        """Return the recorded payload."""
        return self._payload


class _FakeComponentBatch:
    """Mimic one Rerun component batch."""

    def __init__(self, descriptor: str, payload: Any) -> None:
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

    def __init__(self, context: Any) -> None:
        self.context = context

    def as_component_batches(self) -> list[_FakeComponentBatch]:
        """Return an indicator batch plus the annotation-context batch."""
        payload: Any = (
            [[]]
            if type(self).serializes_empty
            else [[{"class_id": info["id"], "label": info["label"]} for info in self.context]]
        )
        return [
            _FakeComponentBatch("rerun.components.AnnotationContextIndicator", [None]),
            _FakeComponentBatch(_ANNOTATION_DESCRIPTOR, payload),
        ]


def _build_fake_rerun(calls: dict[str, Any]) -> Any:
    """Build a fake ``rerun`` module that records every call it receives."""

    class _FakeImage:
        def __init__(self, image: Any) -> None:
            self.image = image

        def compress(self, *, jpeg_quality: int) -> tuple[str, dict[str, Any]]:
            return (
                "EncodedImage",
                {"image": self.image, "jpeg_quality": jpeg_quality},
            )

    class _FakeRecording:
        @staticmethod
        def flush(*, blocking: bool) -> None:
            calls["flushes"].append(blocking)

    class _Part(dict):
        """A recorded blueprint part that also accepts attributes such as ``id``."""

    class _FakeBlueprint(_Part):
        """A recorded blueprint that can also be saved to a file."""

        def save(self, application_id: str, path: str) -> None:
            Path(path).write_bytes(f"blueprint:{application_id}".encode())
            calls.setdefault("saved_blueprints", []).append((application_id, path, self))

    class _FakeBlueprintModule:
        @staticmethod
        def _part(kind: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return _Part(kind=kind, args=args, **kwargs)

        @staticmethod
        def Blueprint(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return _FakeBlueprint(kind="Blueprint", args=args, **kwargs)

        @classmethod
        def Spatial3DView(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("Spatial3DView", *args, **kwargs)

        @classmethod
        def Spatial2DView(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("Spatial2DView", *args, **kwargs)

        @classmethod
        def TimeSeriesView(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("TimeSeriesView", *args, **kwargs)

        @classmethod
        def TextLogView(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("TextLogView", *args, **kwargs)

        @classmethod
        def TextDocumentView(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("TextDocumentView", *args, **kwargs)

        @classmethod
        def Horizontal(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("Horizontal", *args, **kwargs)

        @classmethod
        def Vertical(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("Vertical", *args, **kwargs)

        @classmethod
        def Tabs(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("Tabs", *args, **kwargs)

        @classmethod
        def ScalarAxis(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("ScalarAxis", *args, **kwargs)

        @classmethod
        def EntityBehavior(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("EntityBehavior", *args, **kwargs)

        @classmethod
        def VisualizerOverrides(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("VisualizerOverrides", *args, **kwargs)

        @classmethod
        def VisibleTimeRange(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.setdefault("visible_time_ranges", []).append((args, kwargs))
            return cls._part("VisibleTimeRange", *args, **kwargs)

        class TimeRangeBoundary:
            @staticmethod
            def cursor_relative(**kwargs: Any) -> dict[str, Any]:
                return {"kind": "cursor_relative", **kwargs}

        @classmethod
        def BlueprintPanel(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("BlueprintPanel", *args, **kwargs)

        @classmethod
        def SelectionPanel(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("SelectionPanel", *args, **kwargs)

        @classmethod
        def TimePanel(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return cls._part("TimePanel", *args, **kwargs)

    class _FakeRR:
        AnnotationContext = _FakeAnnotationContext
        blueprint = _FakeBlueprintModule

        class TransformRelation:
            ChildFromParent = "ChildFromParent"

        @staticmethod
        def init(application_id: str, **kwargs: Any) -> None:
            calls["init"] = (application_id, kwargs)

        @staticmethod
        def serve_web(**kwargs: Any) -> None:
            calls["serve_web"] = kwargs

        @staticmethod
        def get_global_data_recording() -> _FakeRecording:
            return _FakeRecording()

        @staticmethod
        def set_time(
            timeline: str,
            *,
            sequence: int | None = None,
            timestamp: float | None = None,
        ) -> None:
            calls["times"].append((timeline, sequence, timestamp))

        @staticmethod
        def log(path: str, payload: Any, **kwargs: Any) -> None:
            calls["logs"].append((path, payload, kwargs))

        @staticmethod
        def send_blueprint(payload: Any, **kwargs: Any) -> None:
            calls["blueprints"].append((payload, kwargs))

        @staticmethod
        def AnnotationInfo(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        @staticmethod
        def Image(image: Any) -> _FakeImage:
            return _FakeImage(image)

        @staticmethod
        def Points3D(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
            return ("Points3D", args, kwargs)

        @staticmethod
        def Points2D(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
            return ("Points2D", args, kwargs)

        @staticmethod
        def LineStrips2D(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
            return ("LineStrips2D", args, kwargs)

        @staticmethod
        def Boxes3D(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Boxes3D", kwargs)

        @staticmethod
        def Transform3D(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Transform3D", kwargs)

        @staticmethod
        def Pinhole(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Pinhole", kwargs)

        @staticmethod
        def Scalars(value: Any) -> tuple[str, Any]:
            return ("Scalars", value)

        @staticmethod
        def SeriesPoints(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("SeriesPoints", kwargs)

        @staticmethod
        def TextLog(text: str, **kwargs: Any) -> tuple[str, str, dict[str, Any]]:
            return ("TextLog", text, kwargs)

        class MediaType:
            MARKDOWN = "text/markdown"

        @staticmethod
        def TextDocument(text: str, **kwargs: Any) -> tuple[str, str, dict[str, Any]]:
            return ("TextDocument", text, kwargs)

        @staticmethod
        def Clear(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return ("Clear", kwargs)

    return _FakeRR


@pytest.fixture
def rerun_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a fake ``rerun`` module and return its recorded calls."""
    calls: dict[str, Any] = {
        "init": None,
        "serve_web": None,
        "logs": [],
        "times": [],
        "blueprints": [],
        "flushes": [],
    }
    monkeypatch.setattr(
        "autoware_ml.visualization.rerun_backend._load_rerun_module",
        lambda: _build_fake_rerun(calls),
    )
    return calls


@pytest.fixture
def backend(rerun_calls: dict[str, Any]) -> RerunVisualizationBackend:
    """Build one Rerun backend wired to the fake module."""
    return RerunVisualizationBackend(
        VisualizationSessionConfig(web_port=9091, grpc_port=9877, wait=False)
    )


def _logged(calls: dict[str, Any], path: str) -> list[Any]:
    """Return the payloads logged to one entity path."""
    return [payload for logged_path, payload, _ in calls["logs"] if logged_path == path]


def test_backend_initializes_and_serves_the_web_viewer(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    assert rerun_calls["init"][0] == "autoware-ml"
    assert rerun_calls["init"][1]["spawn"] is False
    assert rerun_calls["serve_web"] == {
        "open_browser": False,
        "web_port": 9091,
        "grpc_port": 9877,
        "server_memory_limit": "25%",
    }
    assert backend.web_url == (
        "http://localhost:9091?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9877%2Fproxy"
    )


def test_backend_forwards_timeline_steps(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.set_step(3)
    backend.set_timestamp(12.5)

    assert rerun_calls["times"] == [
        ("frame", 3, None),
        ("sensor_time", None, 12.5),
    ]


def test_backend_flushes_pending_data_before_returning(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.wait_until_interrupted()

    assert rerun_calls["flushes"] == [True]


def test_backend_logs_annotation_context_statically(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    """Legends must be static so one context covers every frame on the timeline."""
    backend.log_event(
        AnnotationContextEvent(
            path="detection3d",
            annotations=[AnnotationInfo(id=0, label="car", color=(255, 0, 0, 255))],
        )
    )

    path, payload, kwargs = rerun_calls["logs"][0]
    assert path == "detection3d"
    assert kwargs == {"static": True}
    assert isinstance(payload, _FakeAnnotationContext)
    assert payload.context == [{"id": 0, "label": "car", "color": (255, 0, 0, 255)}]


def test_backend_uses_the_supported_scalars_api(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    """``rr.Scalar`` is deprecated since rerun 0.23, so ``rr.Scalars`` must be used."""
    backend.log_event(ScalarEvent(path="scene/metrics/detection/num_predictions", value=4.0))

    assert _logged(rerun_calls, "scene/metrics/detection/num_predictions") == [("Scalars", 4.0)]


def test_backend_translates_every_supported_event(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.log_events(
        [
            ImageEvent(
                path="scene/cameras/front",
                image=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            PointCloud3DEvent(
                path="scene/lidar/solid",
                positions=np.zeros((2, 3), dtype=np.float32),
            ),
            Points2DEvent(
                path="scene/cameras/front/overlay",
                positions=np.zeros((2, 2), np.float32),
            ),
            LineStrips2DEvent(
                path="scene/cameras/front/boxes",
                strips=[np.zeros((2, 2), np.float32)],
            ),
            Boxes3DEvent(
                path="scene/prediction/detections",
                centers=np.zeros((1, 3), dtype=np.float32),
                sizes=np.ones((1, 3), dtype=np.float32),
                yaws=np.zeros((1,), dtype=np.float32),
            ),
            Transform3DEvent(
                path="scene/cameras/front",
                translation=np.zeros((3,), dtype=np.float32),
                rotation_matrix=np.eye(3, dtype=np.float32),
            ),
            PinholeEvent(
                path="scene/cameras/front",
                image_from_camera=np.eye(3, dtype=np.float32),
                resolution=(64, 36),
            ),
            TextEvent(path="meta/sample", text="sample-1"),
            ClearEvent(path="scene/ground_truth"),
        ]
    )

    logged_kinds = [payload[0] for _, payload, _ in rerun_calls["logs"]]
    assert logged_kinds == [
        "EncodedImage",
        "Points3D",
        "Points2D",
        "LineStrips2D",
        "Boxes3D",
        "Transform3D",
        "Pinhole",
        "TextLog",
        "Clear",
    ]
    assert rerun_calls["blueprints"]


def test_backend_compresses_images_for_remote_streaming(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    backend.log_event(ImageEvent(path="scene/cameras/front", image=image))

    encoded = _logged(rerun_calls, "scene/cameras/front")[0]
    assert encoded[0] == "EncodedImage"
    assert encoded[1]["jpeg_quality"] == 95
    assert encoded[1]["image"] is image


def _free_port() -> int:
    """Return a TCP port that is free right now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _multitask_scene_events() -> list[Any]:
    """Return the events of one complete multitask frame with cameras and metrics."""
    return [
        PointCloud3DEvent(
            path="scene/prediction/segmentation",
            positions=np.zeros((2, 3), dtype=np.float32),
        ),
        PointCloud3DEvent(
            path="scene/ground_truth/segmentation",
            positions=np.zeros((2, 3), dtype=np.float32),
        ),
        PointCloud3DEvent(
            path="scene/lidar/intensity",
            positions=np.zeros((2, 3), dtype=np.float32),
        ),
        PointCloud3DEvent(
            path="scene/prediction/entropy",
            positions=np.zeros((2, 3), dtype=np.float32),
        ),
        PointCloud3DEvent(
            path="scene/prediction/probability",
            positions=np.zeros((2, 3), dtype=np.float32),
        ),
        PinholeEvent(
            path="scene/cameras/front",
            image_from_camera=np.eye(3, dtype=np.float32),
            resolution=(64, 36),
        ),
        Points2DEvent(
            path="scene/cameras/front/projected/ground_truth/segmentation",
            positions=np.zeros((2, 2), np.float32),
        ),
        LineStrips2DEvent(
            path="scene/cameras/front/projected/prediction/detections",
            strips=[np.zeros((2, 2), np.float32)],
        ),
        Boxes3DEvent(
            path="scene/prediction/detections",
            centers=np.zeros((1, 3), dtype=np.float32),
            sizes=np.ones((1, 3), dtype=np.float32),
            yaws=np.zeros((1,), dtype=np.float32),
        ),
        Boxes3DEvent(
            path="scene/ground_truth/detections",
            centers=np.zeros((1, 3), dtype=np.float32),
            sizes=np.ones((1, 3), dtype=np.float32),
            yaws=np.zeros((1,), dtype=np.float32),
        ),
        ScalarEvent(path="scene/metrics/detection/precision", value=1.0),
        ScalarEvent(path="scene/metrics/detection/recall", value=1.0),
        ScalarEvent(path="scene/metrics/detection/mean_best_iou", value=1.0),
        ScalarEvent(path="scene/metrics/detection/max_iou", value=1.0),
        ScalarEvent(path="scene/metrics/detection/mean_matched_iou", value=1.0),
        ScalarEvent(path="scene/metrics/detection/true_positives", value=20.0),
        ScalarEvent(path="scene/metrics/detection/false_positives", value=12.0),
        ScalarEvent(path="scene/metrics/detection/false_negatives", value=8.0),
    ]


def test_backend_builds_named_comparison_views_without_root_origins(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.log_events(_multitask_scene_events())

    blueprint, options = rerun_calls["blueprints"][-1]
    serialized = repr(blueprint)
    assert options == {"make_active": True, "make_default": True}
    assert "Semantic" not in serialized
    assert "Multi comparison" in serialized
    assert "GT · Multi" in serialized
    assert "Prediction · Multi" in serialized
    assert "Prediction · Intensity" not in serialized
    assert "Normalized entropy" in serialized
    assert "Probability" in serialized
    assert serialized.index("Prediction · Multi") < serialized.index("GT · Multi")
    assert "front · GT" in serialized
    assert "front · Prediction" in serialized
    assert serialized.index("front · Prediction") < serialized.index("front · GT")
    assert "scene/cameras/front/projected/ground_truth/segmentation" in serialized
    assert "scene/cameras/front/projected/prediction/detections" in serialized
    assert "3D IoU quality" in serialized
    assert "GT mean best IoU" in serialized
    assert "Frame max IoU" in serialized
    assert "Matched mean IoU (>=0.5)" in serialized
    assert (
        "'/scene/cameras': {'kind': 'EntityBehavior', 'args': (), 'visible': False}" in serialized
    )
    assert "'/scene/cameras': {'kind': 'EntityBehavior', 'args': (), 'visible': True}" in serialized
    assert "'/scene/prediction/detections': ('Boxes3D', {'show_labels': True})" in serialized
    assert "Camera projections OFF" in serialized
    assert "Camera projections ON" in serialized
    assert "'origin': '/'" not in serialized
    assert "'auto_views': False" in serialized
    assert "'kind': 'TimePanel', 'args': (), 'expanded': False" in serialized
    assert "'kind': 'BlueprintPanel', 'args': (), 'expanded': True" in serialized
    assert "'kind': 'VisualizerOverrides', 'args': (['SeriesLines', 'SeriesPoints'],)" in serialized
    assert "('SeriesPoints', {'names': 'Precision', 'marker_sizes': 8.0})" in serialized
    assert "'row_shares': [2.0, 1.0]" in serialized
    assert "'kind': 'VisibleTimeRange', 'args': ('frame',)" in serialized
    # The single task comparison has two metric plots, each with an independent
    # Rerun time-range archetype.
    assert len(rerun_calls["visible_time_ranges"]) == 2


def test_blueprint_panel_stays_collapsed_without_cameras(
    backend: RerunVisualizationBackend,
    rerun_calls: dict[str, Any],
) -> None:
    backend.log_events(
        [
            PointCloud3DEvent(
                path="scene/prediction/segmentation",
                positions=np.zeros((2, 3), dtype=np.float32),
            )
        ]
    )

    blueprint, _ = rerun_calls["blueprints"][-1]
    assert "'kind': 'BlueprintPanel', 'args': (), 'expanded': False" in repr(blueprint)


def test_backend_can_show_camera_geometry_initially(
    rerun_calls: dict[str, Any],
) -> None:
    backend = RerunVisualizationBackend(
        VisualizationSessionConfig(
            web_port=9091,
            grpc_port=9877,
            wait=False,
            camera_frustums_visible=True,
        )
    )

    backend.log_events(
        [
            PinholeEvent(
                path="scene/cameras/front",
                image_from_camera=np.eye(3, dtype=np.float32),
                resolution=(64, 36),
            ),
            PointCloud3DEvent(
                path="scene/prediction/segmentation",
                positions=np.zeros((2, 3), dtype=np.float32),
            ),
        ]
    )

    blueprint, _ = rerun_calls["blueprints"][-1]
    serialized = repr(blueprint)
    assert "'active_tab': 1" in serialized
    assert "'/scene/cameras': {'kind': 'EntityBehavior', 'args': (), 'visible': True}" in serialized


def test_backend_accepts_adapter_defined_blueprint_without_path_knowledge(
    backend: RerunVisualizationBackend,
    rerun_calls: dict[str, Any],
) -> None:
    """A new adapter can own its layout without changing the Rerun backend."""
    backend.log_events(
        [
            PointCloud3DEvent(
                path="scene/custom_task/output",
                positions=np.zeros((2, 3), dtype=np.float32),
            ),
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

    assert len(rerun_calls["blueprints"]) == 1
    assert "Custom task" in repr(rerun_calls["blueprints"][0][0])
    assert "scene/custom_task/output" in repr(rerun_calls["blueprints"][0][0])

    backend.log_events(
        [
            PointCloud3DEvent(
                path="scene/custom_task/second_output",
                positions=np.zeros((2, 3), dtype=np.float32),
            )
        ]
    )

    assert len(rerun_calls["blueprints"]) == 1


def test_backend_converts_yaw_to_a_z_axis_quaternion(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.log_event(
        Boxes3DEvent(
            path="scene/prediction/detections",
            centers=np.zeros((1, 3), dtype=np.float32),
            sizes=np.ones((1, 3), dtype=np.float32),
            yaws=np.array([np.pi / 2], dtype=np.float32),
        )
    )

    _, payload, _ = rerun_calls["logs"][0]
    quaternion = payload[1]["quaternions"]
    np.testing.assert_allclose(
        quaternion, [[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]], atol=1e-6
    )


def test_backend_logs_a_markdown_document(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.log_event(
        TextDocumentEvent(path="calibration_status/status/verdict", text="# Prediction: CALIBRATED")
    )

    path, payload, _ = rerun_calls["logs"][0]
    assert path == "calibration_status/status/verdict"
    assert payload == ("TextDocument", "# Prediction: CALIBRATED", {"media_type": "text/markdown"})


def test_backend_logs_a_plain_document_without_a_media_type(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.log_event(TextDocumentEvent(path="notes", text="plain", markdown=False))

    _, payload, _ = rerun_calls["logs"][0]
    assert payload == ("TextDocument", "plain", {})


def test_backend_builds_a_text_document_view(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
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

    assert len(rerun_calls["blueprints"]) == 1
    assert "TextDocumentView" in repr(rerun_calls["blueprints"][0][0])
    assert "Calibration verdict" in repr(rerun_calls["blueprints"][0][0])


def test_backend_rejects_unknown_events(backend: RerunVisualizationBackend) -> None:
    with pytest.raises(TypeError, match="Unsupported visualization event"):
        backend.log_event(object())  # type: ignore[arg-type]


def test_verify_annotation_context_support_raises_when_legends_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silently emptied AnnotationContext must fail loudly, not lose the legend."""
    monkeypatch.setattr(_FakeAnnotationContext, "serializes_empty", True)
    fake_rerun = _build_fake_rerun(
        {
            "init": None,
            "serve_web": None,
            "logs": [],
            "times": [],
            "blueprints": [],
        }
    )

    with pytest.raises(RuntimeError, match="discarded a probe AnnotationContext"):
        _verify_annotation_context_support(fake_rerun)


def test_installed_rerun_serializes_annotation_contexts() -> None:
    """Regression guard for the rerun-sdk 0.23.1 / NumPy 1.x ``__array__`` break.

    Rerun reports the failure as a warning and emits an empty context, so
    without the compatibility patch every class legend disappears from the
    viewer while the recording still looks healthy.
    """
    rerun = pytest.importorskip("rerun")
    _patch_class_id_array_protocol()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        context = rerun.AnnotationContext(
            [
                rerun.AnnotationInfo(id=0, label="car", color=(255, 0, 0, 255)),
                rerun.AnnotationInfo(id=1, label="pedestrian", color=(0, 255, 0, 255)),
            ]
        )
        payload = next(
            batch.as_arrow_array().to_pylist()
            for batch in context.as_component_batches()
            if "AnnotationContext#" in str(batch.component_descriptor())
        )

    entries = payload[0]
    assert [entry["class_id"] for entry in entries] == [0, 1]
    assert [entry["class_description"]["info"]["label"] for entry in entries] == [
        "car",
        "pedestrian",
    ]


def test_installed_rerun_passes_the_compatibility_probe() -> None:
    rerun = pytest.importorskip("rerun")
    _patch_class_id_array_protocol()

    _verify_annotation_context_support(rerun)


def test_installed_rerun_serializes_cursor_relative_time_ranges() -> None:
    """The metric plot window must not be dropped by Rerun under NumPy 1.x."""
    rerun = pytest.importorskip("rerun")
    _patch_time_int_array_protocol()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        view = rerun.blueprint.TimeSeriesView(
            origin="scene/metrics/detection",
            time_ranges=[
                rerun.blueprint.VisibleTimeRange(
                    "frame",
                    start=rerun.blueprint.TimeRangeBoundary.cursor_relative(seq=-10),
                    end=rerun.blueprint.TimeRangeBoundary.cursor_relative(seq=10),
                )
            ],
        )

    assert view is not None


def _scene_view_ids(part: Any) -> list[Any]:
    """Collect the ids of the 3D views nested in one recorded blueprint part."""
    if not isinstance(part, dict):
        return []
    ids = [part.id] if part.get("kind") == "Spatial3DView" else []
    for child in part.get("args", ()):
        ids.extend(_scene_view_ids(child))
    return ids


def test_backend_publishes_synced_layouts_when_enabled(
    rerun_calls: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix: str(tmp_path))
    sync_port = _free_port()
    backend = RerunVisualizationBackend(
        VisualizationSessionConfig(
            web_port=9091,
            grpc_port=9877,
            wait=False,
            sync_views=True,
            sync_port=sync_port,
        )
    )
    try:
        backend.log_events(_multitask_scene_events())

        saved = rerun_calls["saved_blueprints"]
        names = sorted(Path(path).name for _, path, _ in saved)
        assert names == sorted(
            ["left-off.rbl.partial", "left-on.rbl.partial"]
            + [
                f"right-{key}-{state}.rbl.partial"
                for key in ("gt", "intensity", "entropy", "probability")
                for state in ("off", "on")
            ]
        )
        assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
            name.removesuffix(".partial") for name in names
        )
        assert all(application_id == "autoware-ml" for application_id, _, _ in saved)

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
        assert len(left_ids) == 1, "both camera states of the left side keep one view id"
        assert len(right_ids) == 1, "every comparison of the right side keeps one view id"
        assert left_ids != right_ids

        # The streamed blueprint of the plain viewer page is unchanged.
        assert len(rerun_calls["blueprints"]) == 1

        with urllib.request.urlopen(f"http://127.0.0.1:{sync_port}/config.json") as response:
            config = json.load(response)
        assert config["version"] == 1
        assert config["timeline"] == "frame"
        assert config["grpc_port"] == 9877
        assert config["camera_states"] == ["off", "on"]
        assert config["initial_comparison"] == "gt"
        assert config["initial_camera_state"] == "off"
        assert [comparison["key"] for comparison in config["comparisons"]] == [
            "gt",
            "intensity",
            "entropy",
            "probability",
        ]
        assert config["left"] == {"off": "left-off.rbl", "on": "left-on.rbl"}
        assert config["right"]["gt"] == {"off": "right-gt-off.rbl", "on": "right-gt-on.rbl"}
        with urllib.request.urlopen(
            f"http://127.0.0.1:{sync_port}/blueprints/right-gt-on.rbl?v=1"
        ) as response:
            assert response.read() == b"blueprint:autoware-ml"
    finally:
        assert backend._synced_server is not None
        backend._synced_server.close()


def test_backend_keeps_the_synced_page_off_by_default(
    backend: RerunVisualizationBackend, rerun_calls: dict[str, Any]
) -> None:
    backend.log_events(_multitask_scene_events())

    assert backend._synced_server is None
    assert "saved_blueprints" not in rerun_calls
