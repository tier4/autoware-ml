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

"""Tests for the sample-at-a-time visualization preview."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import polars as pl
import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch, ModelGTSample
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.calibration_status.predictions import (
    CalibrationStatusPredictions,
)
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.predictions import Segmentation3DPredictions
from autoware_ml.datamodule.base_dataset import BaseDataset, ConcatDataset
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.types.dataset import SplitType
from autoware_ml.types.geometry import (
    Box3DCenterCoordinateType,
    Box3DFieldIndex,
    PointFeatureName,
)
from autoware_ml.visualization import preview
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.events import (
    AnnotationContextEvent,
    Boxes3DEvent,
    ImageEvent,
    PointCloud3DEvent,
    TextDocumentEvent,
    Transform3DEvent,
)
from autoware_ml.visualization.preview import VisualizationPreviewConfig
from autoware_ml.visualization.session import VisualizationSession
from autoware_ml.visualization.tests.conftest import RecordingBackend

_SEGMENTATION_CLASSES = ("road", "car")
_DETECTION_CLASSES = ("car", "pedestrian")
_IGNORE_INDEX = 2
_LIDAR_FEATURES = (
    PointFeatureName.X,
    PointFeatureName.Y,
    PointFeatureName.Z,
    PointFeatureName.INTENSITY,
    PointFeatureName.TIMESTAMP_DIFFERENCE,
)


def _points(time_lags: Sequence[float]) -> LiDARPoints:
    """Build one point per time lag, spread along x, with the time lag in the last column."""
    num_points = len(time_lags)
    points = torch.zeros((num_points, len(_LIDAR_FEATURES)), dtype=torch.float32)
    points[:, 0] = torch.arange(num_points, dtype=torch.float32) + 5.0
    points[:, 3] = 0.5
    points[:, 4] = torch.tensor(time_lags, dtype=torch.float32)
    return LiDARPoints(
        points=points,
        point_feature_names=_LIDAR_FEATURES,
        timestamp=0.0,
        timestamp_difference_dim=len(_LIDAR_FEATURES) - 1,
    )


def _boxes(num_boxes: int) -> LidarBBoxes3D:
    """Build ``num_boxes`` unit cars spread along x."""
    bbox_params = torch.zeros((num_boxes, len(Box3DFieldIndex)), dtype=torch.float32)
    bbox_params[:, Box3DFieldIndex.X] = torch.arange(num_boxes, dtype=torch.float32) + 5.0
    bbox_params[:, Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT + 1] = 1.0
    return LidarBBoxes3D(
        bbox_params=bbox_params,
        bbox_labels=torch.zeros(num_boxes, dtype=torch.int32),
        bbox_label_names=["car"] * num_boxes,
        bbox_num_lidar_points=torch.ones(num_boxes, dtype=torch.int32),
        bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
    )


def _lidar_frame(path: str, timestamp: float) -> LiDARPointCloudSample:
    """Describe the current lidar frame of a sample."""
    return LiDARPointCloudSample(
        point_cloud_path=path,
        timestamp=timestamp,
        num_features=5,
        intensity_scale=255.0,
        sensor_to_ego_pose_matrix=torch.eye(4),
        lidar_to_ego_pose_to_global_matrix=torch.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=torch.eye(4),
    )


def _camera(image_path: Path) -> ImageSample:
    """Describe one camera looking along the lidar x axis."""
    lidar2cam = torch.tensor(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    intrinsic = torch.tensor([[20.0, 0.0, 32.0], [0.0, 20.0, 18.0], [0.0, 0.0, 1.0]])
    lidar2image = torch.eye(4)
    lidar2image[:3, :3] = intrinsic
    return ImageSample(
        image_path=str(image_path),
        camera_name="CAM_FRONT",
        timestamp=1.5,
        camera_intrinsic=intrinsic,
        lidar2cam=lidar2cam,
        lidar2image=lidar2image @ lidar2cam,
        distortion_model="plumb_bob",
        distortion_coefficients=torch.zeros(5),
    )


def _sample(
    *,
    time_lags: Sequence[float] = (0.0, 0.0, 0.0, 0.1),
    num_boxes: int | None = None,
    segmentation: bool = False,
    image_samples: Sequence[ImageSample] | None = None,
    timestamp: float = 1.0,
) -> ModelGTSample:
    """Build a transformed sample with the requested ground truth."""
    points = _points(time_lags)
    return ModelGTSample(
        lidar_point_cloud_samples=[_lidar_frame("/data/scene/lidar/00010.pcd.bin", timestamp)],
        image_samples=image_samples,
        point_cloud_data=points,
        camera_image_data=None,
        detection3d_gt_bboxes_3d=_boxes(num_boxes) if num_boxes is not None else None,
        segmentation3d_gt_sample=(
            Segmentation3DGTSample(
                gt_semantic_mask=torch.arange(len(time_lags), dtype=torch.int64) % 2,
                ignore_index=_IGNORE_INDEX,
            )
            if segmentation
            else None
        ),
    )


def _calibration_sample(status: int) -> ModelGTSample:
    """Build a calibration-status sample of one 8x8 camera."""
    intrinsics = torch.tensor([[[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]])
    lidar2cams = torch.eye(4).unsqueeze(0)
    lidar2images = torch.eye(4).unsqueeze(0)
    lidar2images[0, :3, :3] = intrinsics[0]
    images = BaseImages(
        images=torch.full((1, 3, 8, 8), 0.5),
        depth_maps=torch.zeros((1, 2, 8, 8)),
        timestamps=torch.zeros(1, dtype=torch.float64),
        camera_intrinsics=intrinsics,
        camera_names=["CAM_FRONT"],
        lidar2images=lidar2images,
        lidar2cams=lidar2cams,
        distortion_models=["plumb_bob"],
        distortion_coefficients=[torch.zeros(5)],
        augmented_camera_intrinsics=intrinsics.clone(),
        image_augmentation_matrices=torch.eye(4).unsqueeze(0),
        calibration_statuses=torch.tensor([status]),
    )
    sample = _sample()
    return sample._replace(camera_image_data=images)


class _SampleDataset(BaseDataset):
    """Serve pre-built transformed samples."""

    def __init__(
        self,
        samples: Sequence[ModelGTSample],
        *,
        det3d_supervised: bool = True,
        seg3d_supervised: bool = True,
    ) -> None:
        """Initialize the dataset, bound to one record per sample."""
        super().__init__(
            database_root_path="/data",
            max_num_3d_gt_bboxes=4,
            dataset_records_dataframe=pl.DataFrame({"record": list(range(len(samples)))}),
            transforms=None,
            det3d_supervised=det3d_supervised,
            seg3d_supervised=seg3d_supervised,
        )
        self.samples = list(samples)

    def get_data_sample(self, index: int) -> ModelGTSample:
        """Return the sample of one index."""
        return self.samples[index]


class _PreviewDataModule:
    """Serve one split of pre-built samples with the taxonomy of the preview tests."""

    def __init__(self, dataset: _SampleDataset) -> None:
        """Initialize the datamodule around one dataset."""
        self.dataset = dataset
        self.datasets: dict[SplitType, ConcatDataset] = {}
        self.prepared = False
        self.stages: list[str | None] = []
        taxonomy = SimpleNamespace(
            segmentation3d=SimpleNamespace(
                class_names=_SEGMENTATION_CLASSES, ignore_index=_IGNORE_INDEX
            ),
            detection3d=SimpleNamespace(class_names=_DETECTION_CLASSES),
        )
        self.source = SimpleNamespace(database=SimpleNamespace(taxonomy=taxonomy))

    def prepare_data(self) -> None:
        """Record that the record tables were prepared."""
        self.prepared = True

    def setup(self, stage: str | None = None) -> None:
        """Serve the dataset as the test split."""
        self.stages.append(stage)
        self.datasets[SplitType.TEST] = ConcatDataset(datasets=[self.dataset], repeats=[1])

    def split_sources(self, split: SplitType) -> dict[str, Any]:
        """Return the single source of every split."""
        del split
        return {"preview": self.source}


class _PreviewModel:
    """Run a fixed prediction function through the Lightning hooks the preview drives."""

    def __init__(
        self,
        predict: Callable[[ModelBatchInputs], ModelPredictions],
        voxels: Callable[[ModelGTBatch], VoxelsData] | None = None,
    ) -> None:
        """Initialize the model from its prediction function and optional voxelizer."""
        self.predict = predict
        self.voxels = voxels
        self.device: torch.device | None = None
        self.evaluating = False

    def to(self, device: torch.device) -> _PreviewModel:
        """Record the device."""
        self.device = device
        return self

    def eval(self) -> _PreviewModel:
        """Record evaluation mode."""
        self.evaluating = True
        return self

    def transfer_batch_to_device(
        self, batch: ModelGTBatch, device: torch.device, dataloader_idx: int
    ) -> ModelGTBatch:
        """Move the batch like the base model."""
        del dataloader_idx
        return batch.to_device(device)

    def on_after_batch_transfer(self, batch: ModelGTBatch, dataloader_idx: int) -> ModelBatchInputs:
        """Wrap the batch into model inputs, voxelized when the model voxelizes."""
        del dataloader_idx
        inputs = ModelBatchInputs.from_gt_batch(batch)
        if self.voxels is not None:
            inputs = inputs.replace(voxels_data=self.voxels(batch))
        return inputs

    def predict_step(self, multi_task_batch_inputs: ModelBatchInputs, batch_idx: int) -> Any:
        """Return the fixed predictions."""
        del batch_idx
        return self.predict(multi_task_batch_inputs)


def _voxels(point_voxel_indices: Sequence[int]) -> Callable[[ModelGTBatch], VoxelsData]:
    """Build a voxelizer that assigns the points the given voxel rows."""

    def voxelize(batch: ModelGTBatch) -> VoxelsData:
        del batch
        return VoxelsData(
            voxels=torch.zeros((1, 1, 5)),
            coords=torch.zeros((1, 3), dtype=torch.int32),
            num_points=torch.ones(1, dtype=torch.int32),
            batch_indices=torch.zeros(1, dtype=torch.int32),
            point_voxel_indices=torch.tensor(point_voxel_indices, dtype=torch.int64),
            num_dropped_voxels=torch.tensor(0),
        )

    return voxelize


def _segmentation_predictions(num_points: int) -> ModelPredictions:
    """Predict alternating classes with confident probabilities."""
    labels = torch.arange(num_points, dtype=torch.int64) % 2
    probabilities = torch.nn.functional.one_hot(labels, num_classes=2).float() * 0.8 + 0.1
    return ModelPredictions(
        segmentation3d_predictions=Segmentation3DPredictions(
            pred_labels=labels, pred_probs=probabilities
        )
    )


def _detection_predictions() -> ModelPredictions:
    """Predict one car."""
    return ModelPredictions(
        detection3d_predictions=[
            Detection3DSamplePredictions(
                bboxes_3d=torch.tensor([[6.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1]]),
                scores_3d=torch.tensor([0.9]),
                labels_3d=torch.tensor([0], dtype=torch.int64),
            )
        ]
    )


@pytest.fixture
def run_preview(
    recording_backend: RecordingBackend, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., RecordingBackend]:
    """Run a preview into the recording backend and return the backend."""
    monkeypatch.setattr(
        VisualizationSession,
        "from_config",
        classmethod(lambda cls, config: cls(recording_backend)),
    )

    def run(
        dataset: _SampleDataset,
        model: _PreviewModel | None = None,
        **config: Any,
    ) -> RecordingBackend:
        preview.run_visualization_preview(
            model, _PreviewDataModule(dataset), VisualizationPreviewConfig(device="cpu", **config)
        )
        return recording_backend

    return run


def _legend_labels(backend: RecordingBackend) -> set[str]:
    """Return every class label of the logged legends."""
    return {
        annotation.label
        for event in backend.events
        if isinstance(event, AnnotationContextEvent)
        for annotation in event.annotations
    }


def _point_cloud(backend: RecordingBackend, path: str) -> PointCloud3DEvent:
    """Return the last point cloud logged at one entity path."""
    return [e for e in backend.events if isinstance(e, PointCloud3DEvent) and e.path == path][-1]


def test_config_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="split"):
        VisualizationPreviewConfig(split="validation")


def test_prediction_mode_requires_a_model(run_preview: Callable[..., RecordingBackend]) -> None:
    with pytest.raises(ValueError, match="Model must be provided"):
        run_preview(_SampleDataset([_sample(segmentation=True)]), mode="predictions")


def test_sample_index_out_of_range_names_the_split_size(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    with pytest.raises(IndexError, match="with 1 samples"):
        run_preview(_SampleDataset([_sample(segmentation=True)]), sample_index=3)


def test_data_preview_shows_the_current_frame_ground_truth(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    """Sweep points carry no annotation of their own, so they stay out of the preview."""
    backend = run_preview(_SampleDataset([_sample(segmentation=True)]))

    ground_truth = _point_cloud(backend, "scene/ground_truth/segmentation")
    assert ground_truth.positions.shape == (3, 3)
    assert ground_truth.class_ids.tolist() == [0, 1, 0]
    assert backend.steps == [0]
    assert backend.timestamps == [1.0]
    assert backend.waited


def test_preview_labels_classes_from_the_database_taxonomy(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    backend = run_preview(_SampleDataset([_sample(segmentation=True)]))

    assert set(_SEGMENTATION_CLASSES) <= _legend_labels(backend)


def test_configured_class_names_override_the_taxonomy(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    backend = run_preview(
        _SampleDataset([_sample(segmentation=True)]),
        segmentation_class_names=("ground", "vehicle"),
    )

    labels = _legend_labels(backend)
    assert {"ground", "vehicle"} <= labels
    assert "road" not in labels


def test_voxel_predictions_align_with_the_scored_points(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    """A voxel head labels the current-frame points inside the grid, in point order."""
    # Point 1 falls outside the grid and point 3 belongs to a past sweep.
    model = _PreviewModel(
        lambda inputs: _segmentation_predictions(2), voxels=_voxels([0, -1, 0, 0])
    )
    backend = run_preview(_SampleDataset([_sample(segmentation=True)]), model)

    prediction = _point_cloud(backend, "scene/prediction/segmentation")
    ground_truth = _point_cloud(backend, "scene/ground_truth/segmentation")
    assert prediction.positions[:, 0].tolist() == [5.0, 7.0]
    assert ground_truth.class_ids.tolist() == [0, 0]
    assert any(e.path == "scene/prediction/entropy" for e in backend.events)
    assert model.evaluating


def test_point_level_predictions_cover_every_point(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    model = _PreviewModel(lambda inputs: _segmentation_predictions(4))
    backend = run_preview(_SampleDataset([_sample(segmentation=True)]), model)

    assert _point_cloud(backend, "scene/prediction/segmentation").positions.shape == (4, 3)


def test_misaligned_segmentation_predictions_are_reported(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    model = _PreviewModel(lambda inputs: _segmentation_predictions(1))

    with pytest.raises(ValueError, match="Cannot align 1 segmentation predictions"):
        run_preview(_SampleDataset([_sample(segmentation=True)]), model)


def test_unsupervised_segmentation_shows_no_ground_truth(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    model = _PreviewModel(lambda inputs: _segmentation_predictions(4))
    backend = run_preview(
        _SampleDataset([_sample(segmentation=True)], seg3d_supervised=False), model
    )

    assert "scene/ground_truth/segmentation" not in backend.paths_of(PointCloud3DEvent)


def test_detection_preview_draws_predicted_and_ground_truth_boxes(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    model = _PreviewModel(lambda inputs: _detection_predictions())
    backend = run_preview(_SampleDataset([_sample(num_boxes=2)]), model)

    boxes = {e.path: e for e in backend.events if isinstance(e, Boxes3DEvent)}
    assert boxes["scene/prediction/detections"].centers.shape == (1, 3)
    assert boxes["scene/ground_truth/detections"].centers.shape == (2, 3)


def test_unsupervised_detection_shows_only_predictions(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    model = _PreviewModel(lambda inputs: _detection_predictions())
    backend = run_preview(_SampleDataset([_sample(num_boxes=2)], det3d_supervised=False), model)

    assert backend.paths_of(Boxes3DEvent) == ["scene/prediction/detections"]


def test_multitask_data_preview_shows_both_ground_truths(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    backend = run_preview(_SampleDataset([_sample(num_boxes=1, segmentation=True)]))

    assert "scene/ground_truth/segmentation" in backend.paths_of(PointCloud3DEvent)
    assert backend.paths_of(Boxes3DEvent) == ["scene/ground_truth/detections"]


def test_multitask_prediction_preview_decodes_both_tasks(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    def predict(inputs: ModelBatchInputs) -> ModelPredictions:
        del inputs
        return ModelPredictions(
            detection3d_predictions=_detection_predictions().detection3d_predictions,
            segmentation3d_predictions=_segmentation_predictions(3).segmentation3d_predictions,
        )

    model = _PreviewModel(predict, voxels=_voxels([0, 0, 0, 0]))
    backend = run_preview(_SampleDataset([_sample(num_boxes=1, segmentation=True)]), model)

    assert "scene/prediction/segmentation" in backend.paths_of(PointCloud3DEvent)
    assert set(backend.paths_of(Boxes3DEvent)) == {
        "scene/prediction/detections",
        "scene/ground_truth/detections",
    }


def test_cameras_are_drawn_from_the_untransformed_sample(
    run_preview: Callable[..., RecordingBackend], tmp_path: Path
) -> None:
    image_path = tmp_path / "front.jpg"
    cv2.imwrite(str(image_path), np.zeros((36, 64, 3), dtype=np.uint8))
    sample = _sample(segmentation=True, image_samples=[_camera(image_path)])
    backend = run_preview(_SampleDataset([sample]))

    assert "scene/cameras/CAM_FRONT" in backend.paths_of(Transform3DEvent)
    assert "scene/cameras/CAM_FRONT" in backend.paths_of(ImageEvent)


def test_consecutive_samples_are_logged_on_their_dataset_index(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    samples = [_sample(segmentation=True, timestamp=float(index)) for index in range(4)]
    backend = run_preview(_SampleDataset(samples), sample_index=1, max_samples=2)

    assert backend.steps == [1, 2]
    assert backend.timestamps == [1.0, 2.0]


def test_calibration_preview_logs_the_status_verdict(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    model = _PreviewModel(
        lambda inputs: ModelPredictions(
            calibration_status_predictions=CalibrationStatusPredictions(
                probabilities=torch.tensor([[0.2, 0.8]])
            )
        )
    )
    backend = run_preview(_SampleDataset([_calibration_sample(status=1)]), model)

    verdict = next(e for e in backend.events if isinstance(e, TextDocumentEvent))
    assert verdict.path == "calibration_status/status/verdict"
    assert verdict.text.splitlines()[0] == "# MATCH"
    assert "calibration_status/camera/fused" in backend.paths_of(ImageEvent)


def test_calibration_data_preview_is_logged_under_the_dataset_root(
    run_preview: Callable[..., RecordingBackend],
) -> None:
    backend = run_preview(_SampleDataset([_calibration_sample(status=0)]))

    assert backend.paths_of(Transform3DEvent) == ["dataset/calibration_status/camera"]


def test_noop_session_runs_a_full_preview() -> None:
    """The noop backend lets the whole pipeline run without the Rerun SDK."""
    datamodule = _PreviewDataModule(_SampleDataset([_sample(segmentation=True)]))

    count = preview.run_visualization_preview(
        None,
        datamodule,
        VisualizationPreviewConfig(
            device="cpu", session=VisualizationSessionConfig(backend="noop")
        ),
    )

    assert count == 1
    assert datamodule.prepared
    assert datamodule.stages == ["test"]
