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

"""Tests for the sample-at-a-time visualization preview pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from autoware_ml.tests.visualization.conftest import (
    CalibrationPreviewModel,
    DetectionPreviewModel,
    PreviewDataModule,
    PreviewDataset,
    PreviewModelBase,
    RecordingBackend,
    SegmentationPreviewModel,
    VoxelizedSegmentationPreviewModel,
)
from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    Boxes3DEvent,
    ClearEvent,
    ImageEvent,
    LineStrips2DEvent,
    PointCloud3DEvent,
    Points2DEvent,
    TextEvent,
)
from autoware_ml.visualization.preview import (
    SEGMENTATION_LABEL_KEYS,
    VisualizationPreviewConfig,
    _has_segmentation_sample,
    _infer_preview_task,
    _resolve_class_names,
    resolve_preview_device,
    run_visualization_preview,
)

_NOOP_PREVIEW = VisualizationPreviewConfig(
    split="test", session=VisualizationSessionConfig(backend="noop")
)


@pytest.fixture
def preview_session(
    monkeypatch: pytest.MonkeyPatch, recording_backend: RecordingBackend
) -> RecordingBackend:
    """Route every preview session to the recording backend."""
    monkeypatch.setattr(
        "autoware_ml.visualization.preview.VisualizationSession.from_config",
        classmethod(lambda cls, config: cls(recording_backend)),
    )
    return recording_backend


def _detection_sample() -> dict[str, Any]:
    """Build one detection sample with ground truth and class names."""
    return {
        "points": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "gt_boxes": np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
        "gt_labels": np.array([1], dtype=np.int64),
        "class_names": ["pedestrian", "car"],
    }


_DETECTION_COLLATION = {
    "points": "concat",
    "gt_boxes": "concat",
    "gt_labels": "concat",
    "class_names": "list",
}
#: Mirrors production collation, which carries no ``class_names`` key. Real
#: split pipelines drop it, so the preview must recover names elsewhere.
_DETECTION_COLLATION_WITHOUT_CLASS_NAMES = {
    "points": "concat",
    "gt_boxes": "concat",
    "gt_labels": "concat",
}
_SEGMENTATION_COLLATION = {"points": "concat", "segment": "concat"}


def _segmentation_sample() -> dict[str, Any]:
    """Build one segmentation sample with per-point ground truth."""
    return {
        "points": np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "segment": np.array([0, 1], dtype=np.int64),
    }


class _DecodedMultiPreviewModel(PreviewModelBase):
    """Return prediction-only joint outputs with pointwise logits."""

    def predict_step(self, batch_inputs_dict: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        del batch_idx
        inverse = batch_inputs_dict["inverse"].long()
        labels = torch.arange(inverse.shape[0], device=inverse.device) % 2
        logits = torch.nn.functional.one_hot(labels, num_classes=2).float() * 8.0
        return {
            "predictions": [
                {
                    "bboxes_3d": torch.tensor(
                        [[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0]],
                        device=inverse.device,
                    ),
                    "scores_3d": torch.tensor([0.9], device=inverse.device),
                    "labels_3d": torch.tensor([0], device=inverse.device),
                }
            ],
            "seg_pred_labels": labels,
            "seg_pred_logits": logits,
        }


class _SegmentationWithoutLogitsModel(PreviewModelBase):
    """Return an incomplete segmentation contract for fail-loud tests."""

    def predict_step(
        self, batch_inputs_dict: dict[str, Any], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        del batch_inputs_dict, batch_idx
        return {
            "pred_labels": torch.tensor([0, 1]),
            "pred_probs": torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        }


class _MultiWithoutLogitsModel(_DecodedMultiPreviewModel):
    """Return decoded multi-task output without the required entropy source."""

    def predict_step(self, batch_inputs_dict: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        predictions = super().predict_step(batch_inputs_dict, batch_idx)
        del predictions["seg_pred_logits"]
        return predictions


class _IntermediatePreviewDataset(PreviewDataset):
    """Expose one prediction-only record after its GT keyframe."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        intermediate_sample: dict[str, Any],
    ) -> None:
        super().__init__(samples)
        self.intermediate_sample = intermediate_sample

    def get_intermediate_prediction_infos(
        self, index: int, prediction_frequency_hz: float
    ) -> list[dict[str, Any]]:
        assert index == 0
        assert prediction_frequency_hz == 10.0
        return [self.intermediate_sample]


class _IntermediatePreviewDataModule(PreviewDataModule):
    """Create a preview dataset with an intermediate-frame provider."""

    def __init__(
        self,
        keyframe: dict[str, Any],
        intermediate_frame: dict[str, Any],
        collation_map: dict[str, str],
    ) -> None:
        super().__init__([keyframe], collation_map)
        self.intermediate_frame = intermediate_frame

    def _create_dataset(self, split: str, dataset_transforms: Any = None) -> PreviewDataset:
        del split, dataset_transforms
        return _IntermediatePreviewDataset(self.samples, self.intermediate_frame)


def test_preview_logs_a_calibration_sample(
    preview_session: RecordingBackend, preview_calibration_data: CalibrationData
) -> None:
    sample = {
        "calibration_data": preview_calibration_data,
        "points": np.array([[0.0, 0.0, 10.0, 0.4]], dtype=np.float32),
        "img": np.zeros((720, 1280, 3), dtype=np.uint8),
        "fused_img": np.zeros((5, 720, 1280), dtype=np.float32),
        "gt_calibration_status": CalibrationStatus.CALIBRATED.value,
        "img_path": "sample.png",
    }

    visualized = run_visualization_preview(
        CalibrationPreviewModel(),
        PreviewDataModule(
            [sample],
            {
                "calibration_data": "list",
                "points": "concat",
                "img": "stack",
                "fused_img": "stack",
                "gt_calibration_status": "list",
                "img_path": "list",
            },
        ),
        _NOOP_PREVIEW,
    )

    assert visualized == 1
    assert preview_session.steps == [0]
    assert "calibration_status/camera/fused" in preview_session.paths_of(ImageEvent)
    assert "calibration_status/camera/image/projected_points" in preview_session.paths_of(
        Points2DEvent
    )


def test_preview_names_the_missing_calibration_entry(
    preview_session: RecordingBackend,
) -> None:
    """A dropped calibration entry must name itself instead of matching no task."""
    sample = {
        "fused_img": np.zeros((5, 720, 1280), dtype=np.float32),
        "gt_calibration_status": CalibrationStatus.CALIBRATED.value,
    }

    with pytest.raises(ValueError, match=r"calibration_data.*collation_map"):
        run_visualization_preview(
            CalibrationPreviewModel(),
            PreviewDataModule(
                [sample],
                {"fused_img": "stack", "gt_calibration_status": "list"},
            ),
            _NOOP_PREVIEW,
        )


def test_preview_logs_a_segmentation_sample(preview_session: RecordingBackend) -> None:
    visualized = run_visualization_preview(
        SegmentationPreviewModel(),
        PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
        _NOOP_PREVIEW,
    )

    assert visualized == 1
    point_paths = preview_session.paths_of(PointCloud3DEvent)
    assert "scene/prediction/segmentation" in point_paths
    assert "scene/ground_truth/segmentation" in point_paths
    assert "scene/prediction/entropy" in point_paths
    assert "scene/prediction/probability" in point_paths
    assert "scene/meta/extent" in point_paths


def test_preview_requires_logits_for_pointwise_probability_and_entropy(
    preview_session: RecordingBackend,
) -> None:
    with pytest.raises(ValueError, match="pred_logits.*pointwise.*entropy"):
        run_visualization_preview(
            _SegmentationWithoutLogitsModel(),
            PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
            _NOOP_PREVIEW,
        )


def test_preview_reconstructs_points_for_voxelized_segmentation(
    preview_session: RecordingBackend,
) -> None:
    """PTv3 drops raw points, so positions come from ``coord[inverse]``."""
    sample = {
        "coord": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "inverse": np.array([0, 1, 0], dtype=np.int64),
        "origin_segment": np.array([0, 1, 0], dtype=np.int64),
    }

    visualized = run_visualization_preview(
        VoxelizedSegmentationPreviewModel(),
        PreviewDataModule(
            [sample],
            {"coord": "concat", "inverse": "index_concat", "origin_segment": "concat"},
        ),
        _NOOP_PREVIEW,
    )

    assert visualized == 1
    prediction = next(
        event
        for event in preview_session.events
        if isinstance(event, PointCloud3DEvent) and event.path == "scene/prediction/segmentation"
    )
    assert prediction.positions.shape == (3, 3)


def test_preview_reconstructs_intensity_for_voxelized_segmentation(
    preview_session: RecordingBackend,
) -> None:
    sample = {
        "coord": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "strength": np.array([[0.1], [0.9]], dtype=np.float32),
        "inverse": np.array([0, 1, 0], dtype=np.int64),
        "origin_segment": np.array([0, 1, 0], dtype=np.int64),
    }

    run_visualization_preview(
        VoxelizedSegmentationPreviewModel(),
        PreviewDataModule(
            [sample],
            {
                "coord": "concat",
                "strength": "concat",
                "inverse": "index_concat",
                "origin_segment": "concat",
            },
        ),
        _NOOP_PREVIEW,
    )

    intensity = next(
        event
        for event in preview_session.events
        if isinstance(event, PointCloud3DEvent) and event.path == "scene/lidar/intensity"
    )
    assert intensity.colors is not None
    assert not np.array_equal(intensity.colors[0], intensity.colors[1])


def test_preview_recovers_ptv3_intensity_from_built_features(
    preview_session: RecordingBackend,
) -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    sample = {
        "coord": coordinates,
        "feat": np.concatenate((coordinates, np.array([[0.1], [0.9]], dtype=np.float32)), axis=1),
        "inverse": np.array([0, 1, 0], dtype=np.int64),
        "origin_segment": np.array([0, 1, 0], dtype=np.int64),
    }

    run_visualization_preview(
        VoxelizedSegmentationPreviewModel(),
        PreviewDataModule(
            [sample],
            {
                "coord": "concat",
                "feat": "concat",
                "inverse": "index_concat",
                "origin_segment": "concat",
            },
        ),
        _NOOP_PREVIEW,
    )

    intensity = next(
        event
        for event in preview_session.events
        if isinstance(event, PointCloud3DEvent) and event.path == "scene/lidar/intensity"
    )
    assert intensity.colors is not None
    assert not np.array_equal(intensity.colors[0], intensity.colors[1])


def test_multitask_preview_omits_explicitly_unavailable_ground_truth(
    preview_session: RecordingBackend,
) -> None:
    sample = {
        "coord": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "strength": np.array([[0.1], [0.9]], dtype=np.float32),
        "inverse": np.array([0, 1, 0], dtype=np.int64),
        "origin_segment": np.array([0, 1, 0], dtype=np.int64),
        "gt_boxes": np.array([[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0]]),
        "gt_labels": np.array([0], dtype=np.int64),
        "has_detection_ground_truth": False,
        "has_segmentation_ground_truth": False,
        "timestamp": 12.5,
    }

    visualized = run_visualization_preview(
        _DecodedMultiPreviewModel(),
        PreviewDataModule(
            [sample],
            {
                "coord": "concat",
                "strength": "concat",
                "inverse": "index_concat",
                "origin_segment": "concat",
                "gt_boxes": "list",
                "gt_labels": "list",
            },
        ),
        _NOOP_PREVIEW,
    )

    assert visualized == 1
    assert preview_session.timestamps == [12.5]
    assert preview_session.paths_of(ClearEvent)
    assert preview_session.paths_of(Boxes3DEvent) == ["scene/prediction/detections"]
    semantic_paths = preview_session.paths_of(PointCloud3DEvent)
    assert "scene/prediction/segmentation" in semantic_paths
    assert "scene/prediction/entropy" in semantic_paths
    assert "scene/prediction/probability" in semantic_paths
    assert "scene/ground_truth/segmentation" not in semantic_paths


def test_multitask_preview_requires_logits_for_pointwise_probability_and_entropy(
    preview_session: RecordingBackend,
) -> None:
    sample = {
        "coord": np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "inverse": np.array([0, 1], dtype=np.int64),
        "origin_segment": np.array([0, 1], dtype=np.int64),
        "gt_boxes": np.zeros((0, 7), dtype=np.float32),
        "gt_labels": np.zeros((0,), dtype=np.int64),
        "has_detection_ground_truth": True,
        "has_segmentation_ground_truth": True,
    }

    with pytest.raises(ValueError, match="seg_pred_logits.*pointwise.*entropy"):
        run_visualization_preview(
            _MultiWithoutLogitsModel(),
            PreviewDataModule(
                [sample],
                {
                    "coord": "concat",
                    "inverse": "index_concat",
                    "origin_segment": "concat",
                    "gt_boxes": "list",
                    "gt_labels": "list",
                },
            ),
            _NOOP_PREVIEW,
        )


def test_multitask_preview_runs_intermediate_frames_at_10hz_without_gt(
    preview_session: RecordingBackend,
) -> None:
    common = {
        "coord": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "strength": np.array([[0.1], [0.9]], dtype=np.float32),
        "inverse": np.array([0, 1, 0], dtype=np.int64),
    }
    keyframe = {
        **common,
        "origin_segment": np.array([0, 1, 0], dtype=np.int64),
        "gt_boxes": np.array([[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0]]),
        "gt_labels": np.array([0], dtype=np.int64),
        "has_detection_ground_truth": True,
        "has_segmentation_ground_truth": True,
        "timestamp": 100.0,
    }
    intermediate_frame = {
        **common,
        "has_detection_ground_truth": False,
        "has_segmentation_ground_truth": False,
        "timestamp": 100.1,
    }
    collation_map = {
        "coord": "concat",
        "strength": "concat",
        "inverse": "index_concat",
        "origin_segment": "concat",
        "gt_boxes": "list",
        "gt_labels": "list",
    }

    visualized = run_visualization_preview(
        _DecodedMultiPreviewModel(),
        _IntermediatePreviewDataModule(
            keyframe,
            intermediate_frame,
            collation_map,
        ),
        _NOOP_PREVIEW,
    )

    assert visualized == 2
    assert preview_session.steps == [0, 1]
    assert preview_session.timestamps == [100.0, 100.1]
    assert len(preview_session.paths_of(ClearEvent)) == 10
    assert preview_session.paths_of(Boxes3DEvent) == [
        "scene/prediction/detections",
        "scene/ground_truth/detections",
        "scene/prediction/detections",
    ]
    assert preview_session.paths_of(PointCloud3DEvent).count("scene/ground_truth/segmentation") == 1


def test_preview_logs_transformed_voxelized_data_without_a_model(
    preview_session: RecordingBackend,
) -> None:
    """PTv3 carries positions in ``coord``, so ``--mode data`` must still route.

    This path has no predictions to fall back on, so requiring a ``points`` key
    made transformed-data preview unusable for every PTv3 segmentation config.
    """
    sample = {
        "coord": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "inverse": np.array([0, 1, 0], dtype=np.int64),
        "origin_segment": np.array([0, 1, 0], dtype=np.int64),
    }

    visualized = run_visualization_preview(
        None,
        PreviewDataModule(
            [sample],
            {"coord": "concat", "inverse": "index_concat", "origin_segment": "concat"},
        ),
        VisualizationPreviewConfig(
            mode="data",
            split="test",
            session=VisualizationSessionConfig(backend="noop"),
        ),
    )

    assert visualized == 1
    assert "scene/ground_truth/segmentation" in preview_session.paths_of(PointCloud3DEvent)
    logged = next(event for event in preview_session.events if isinstance(event, PointCloud3DEvent))
    assert logged.positions.shape == (3, 3)


def test_preview_routes_combined_detection_and_segmentation_to_both_adapters() -> None:
    batch = {
        "points": np.zeros((2, 4), dtype=np.float32),
        "gt_boxes": np.zeros((1, 7), dtype=np.float32),
        "gt_labels": np.zeros((1,), dtype=np.int64),
        "segment": np.zeros((2,), dtype=np.int64),
    }

    assert _infer_preview_task(batch, None) == "multi"


def test_preview_logs_a_detection_sample(preview_session: RecordingBackend) -> None:
    visualized = run_visualization_preview(
        DetectionPreviewModel(),
        PreviewDataModule([_detection_sample()], _DETECTION_COLLATION),
        _NOOP_PREVIEW,
    )

    assert visualized == 1
    assert preview_session.paths_of(Boxes3DEvent) == [
        "scene/prediction/detections",
        "scene/ground_truth/detections",
    ]


def test_detection_preview_logs_persistent_multiview_camera_overlays(
    preview_session: RecordingBackend,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "front.png"
    cv2.imwrite(str(image_path), np.zeros((36, 64, 3), dtype=np.uint8))
    sample = _detection_sample()
    sample["points"] = np.array([[0.0, 0.0, 10.0, 1.0]], dtype=np.float32)
    sample["images"] = {
        "CAM_FRONT": {
            "img_path": str(image_path),
            "cam2img": [[10.0, 0.0, 32.0], [0.0, 10.0, 18.0], [0.0, 0.0, 1.0]],
            "lidar2cam": np.eye(4, dtype=np.float32),
        }
    }

    run_visualization_preview(
        DetectionPreviewModel(),
        PreviewDataModule([sample], _DETECTION_COLLATION),
        _NOOP_PREVIEW,
    )

    assert preview_session.paths_of(Points2DEvent) == ["scene/cameras/CAM_FRONT/projected/lidar"]
    assert preview_session.paths_of(LineStrips2DEvent) == [
        "scene/cameras/CAM_FRONT/projected/prediction/detections",
        "scene/cameras/CAM_FRONT/projected/ground_truth/detections",
    ]


def test_resolve_class_names_prefers_the_configured_names() -> None:
    config = VisualizationPreviewConfig(class_names=("configured",))

    resolved = _resolve_class_names(
        config,
        {"class_names": [["collated"]]},
        {"class_names": ["raw"]},
        task="segmentation3d",
    )

    assert resolved == ("configured",)


def test_resolve_class_names_falls_back_to_the_raw_dataset_info() -> None:
    """Split pipelines drop class names, leaving raw dataset info the only source."""
    resolved = _resolve_class_names(
        _NOOP_PREVIEW,
        {},
        {"class_names": ["car", "truck"]},
        task="detection3d",
    )

    assert resolved == ["car", "truck"]


def test_resolve_class_names_returns_none_when_no_source_carries_them() -> None:
    assert _resolve_class_names(_NOOP_PREVIEW, {}, None, task="segmentation3d") is None


def test_resolve_class_names_keeps_multitask_legends_independent() -> None:
    config = VisualizationPreviewConfig(
        segmentation_class_names=("road", "vegetation"),
        detection_class_names=("car", "pedestrian"),
    )

    assert _resolve_class_names(config, {}, None, task="segmentation3d") == (
        "road",
        "vegetation",
    )
    assert _resolve_class_names(config, {}, None, task="detection3d") == (
        "car",
        "pedestrian",
    )


def test_preview_names_detection_instances_without_collated_class_names(
    preview_session: RecordingBackend,
) -> None:
    """Boxes and legend must read as names even when collation drops class names."""
    visualized = run_visualization_preview(
        None,
        PreviewDataModule([_detection_sample()], _DETECTION_COLLATION_WITHOUT_CLASS_NAMES),
        VisualizationPreviewConfig(
            mode="data",
            split="test",
            session=VisualizationSessionConfig(backend="noop"),
        ),
    )

    assert visualized == 1
    boxes = next(event for event in preview_session.events if isinstance(event, Boxes3DEvent))
    assert boxes.labels == ["car"]
    legend = next(
        event for event in preview_session.events if isinstance(event, AnnotationContextEvent)
    )
    assert [annotation.label for annotation in legend.annotations] == [
        "pedestrian",
        "car",
    ]


def test_preview_logs_transformed_data_without_a_model(
    preview_session: RecordingBackend,
) -> None:
    visualized = run_visualization_preview(
        None,
        PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
        VisualizationPreviewConfig(
            mode="data",
            split="test",
            session=VisualizationSessionConfig(backend="noop"),
        ),
    )

    assert visualized == 1
    assert "scene/ground_truth/segmentation" in preview_session.paths_of(PointCloud3DEvent)
    assert "scene/meta/sample" in preview_session.paths_of(TextEvent)


def test_preview_scrubs_multiple_samples_on_the_shared_timeline(
    preview_session: RecordingBackend,
) -> None:
    samples = [_detection_sample() for _ in range(3)]

    visualized = run_visualization_preview(
        DetectionPreviewModel(),
        PreviewDataModule(samples, _DETECTION_COLLATION),
        VisualizationPreviewConfig(
            split="test",
            max_samples=3,
            session=VisualizationSessionConfig(backend="noop"),
        ),
    )

    assert visualized == 3
    assert preview_session.steps == [0, 1, 2]


def test_preview_starts_at_the_requested_sample_index(
    preview_session: RecordingBackend,
) -> None:
    samples = [_detection_sample() for _ in range(4)]

    run_visualization_preview(
        DetectionPreviewModel(),
        PreviewDataModule(samples, _DETECTION_COLLATION),
        VisualizationPreviewConfig(
            split="test",
            sample_index=2,
            max_samples=2,
            session=VisualizationSessionConfig(backend="noop"),
        ),
    )

    assert preview_session.steps == [2, 3]


def test_preview_waits_for_the_viewer_after_logging(
    preview_session: RecordingBackend,
) -> None:
    """The preview must hand control to the backend instead of duck-typing it."""
    run_visualization_preview(
        DetectionPreviewModel(),
        PreviewDataModule([_detection_sample()], _DETECTION_COLLATION),
        _NOOP_PREVIEW,
    )

    assert preview_session.waited is True


def test_preview_requires_a_model_for_prediction_mode() -> None:
    with pytest.raises(ValueError, match="Model must be provided"):
        run_visualization_preview(
            None,
            PreviewDataModule(
                [{"points": np.zeros((1, 4), dtype=np.float32)}], {"points": "concat"}
            ),
            VisualizationPreviewConfig(mode="predictions"),
        )


def test_preview_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="Unknown visualization mode: heatmap"):
        run_visualization_preview(
            None,
            PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
            VisualizationPreviewConfig(mode="heatmap"),  # type: ignore[arg-type]
        )


def test_preview_rejects_a_non_positive_sample_count() -> None:
    with pytest.raises(ValueError, match="max_samples must be greater than zero"):
        run_visualization_preview(
            None,
            PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
            VisualizationPreviewConfig(max_samples=0),
        )


def test_preview_rejects_an_out_of_range_sample_index() -> None:
    with pytest.raises(IndexError, match="out of range"):
        run_visualization_preview(
            None,
            PreviewDataModule([_segmentation_sample()], _SEGMENTATION_COLLATION),
            VisualizationPreviewConfig(
                mode="data",
                sample_index=5,
                session=VisualizationSessionConfig(backend="noop"),
            ),
        )


def test_preview_rejects_incomplete_detection_ground_truth(
    preview_session: RecordingBackend,
) -> None:
    sample = _detection_sample()
    del sample["gt_labels"]

    with pytest.raises(ValueError, match="gt_labels.*required"):
        run_visualization_preview(
            None,
            PreviewDataModule([sample], {"points": "concat", "gt_boxes": "concat"}),
            VisualizationPreviewConfig(
                mode="data",
                session=VisualizationSessionConfig(backend="noop"),
            ),
        )


def test_preview_skips_declared_but_empty_optional_camera_metadata(
    preview_session: RecordingBackend,
) -> None:
    sample = _segmentation_sample()
    sample["images"] = {}

    count = run_visualization_preview(
        SegmentationPreviewModel(),
        PreviewDataModule([sample], _SEGMENTATION_COLLATION),
        _NOOP_PREVIEW,
    )

    assert count == 1
    assert preview_session.paths_of(ImageEvent) == []


def test_preview_reports_observed_keys_when_no_task_matches(
    preview_session: RecordingBackend,
) -> None:
    """An unroutable sample must name what it saw instead of guessing a task."""
    with pytest.raises(ValueError, match=r"Could not infer a visualization task.*points"):
        run_visualization_preview(
            None,
            PreviewDataModule(
                [{"points": np.zeros((1, 4), dtype=np.float32)}], {"points": "concat"}
            ),
            VisualizationPreviewConfig(
                mode="data",
                split="test",
                session=VisualizationSessionConfig(backend="noop"),
            ),
        )


def test_preview_routes_a_sample_matching_two_tasks(
    preview_session: RecordingBackend,
) -> None:
    """Multi-task samples are rendered through both task adapters."""
    sample = {
        "points": np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "segment": np.array([0, 1], dtype=np.int64),
        "gt_boxes": np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32),
        "gt_labels": np.array([1], dtype=np.int64),
    }

    visualized = run_visualization_preview(
        None,
        PreviewDataModule(
            [sample],
            {
                "points": "concat",
                "segment": "concat",
                "gt_boxes": "concat",
                "gt_labels": "concat",
            },
        ),
        VisualizationPreviewConfig(
            mode="data",
            split="test",
            session=VisualizationSessionConfig(backend="noop"),
        ),
    )

    assert visualized == 1
    assert "scene/ground_truth/segmentation" in preview_session.paths_of(PointCloud3DEvent)
    assert "scene/ground_truth/detections" in preview_session.paths_of(Boxes3DEvent)


@pytest.mark.parametrize("position_key", ["points", "coord"])
@pytest.mark.parametrize("label_key", SEGMENTATION_LABEL_KEYS)
def test_segmentation_matcher_accepts_every_position_and_label_key(
    position_key: str, label_key: str
) -> None:
    """Both point-source keys pair with every supported per-point label key."""
    batch = {
        position_key: np.zeros((2, 3), dtype=np.float32),
        label_key: np.zeros((2,), dtype=np.int64),
    }

    assert _has_segmentation_sample(batch) is True


@pytest.mark.parametrize(
    "batch",
    [
        {"coord": np.zeros((2, 3), dtype=np.float32)},
        {"segment": np.zeros((2,), dtype=np.int64)},
        {},
    ],
    ids=["positions-only", "labels-only", "empty"],
)
def test_segmentation_matcher_needs_both_positions_and_labels(
    batch: dict[str, Any],
) -> None:
    assert _has_segmentation_sample(batch) is False


def test_resolve_preview_device_honors_explicit_devices() -> None:
    assert resolve_preview_device("cpu").type == "cpu"
    assert resolve_preview_device("auto").type in {"cpu", "cuda"}
