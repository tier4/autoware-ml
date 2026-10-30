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

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import polars as pl
import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch, ModelGTSample
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample
from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample
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
    TransformationName,
)
from autoware_ml.visualization import preview
from autoware_ml.visualization.viewers.contracts import VisualizationSessionConfig
from autoware_ml.visualization.viewers.events import (
    AnnotationContextEvent,
    Boxes3DEvent,
    ImageEvent,
    PointCloud3DEvent,
    TextDocumentEvent,
    Transform3DEvent,
)
from autoware_ml.visualization.preview import VisualizationPreviewConfig
from autoware_ml.visualization.session import VisualizationSession
from autoware_ml.visualization.tests.doubles import RecordingBackend

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


def _lidar_frame(
    path: str, timestamp: float, to_frame: torch.Tensor | None = None
) -> LiDARPointCloudSample:
    """Describe one stored lidar frame, mapped from the record's frame by ``to_frame``."""
    return LiDARPointCloudSample(
        point_cloud_path=path,
        timestamp=timestamp,
        num_features=5,
        intensity_scale=255.0,
        sensor_to_ego_pose_matrix=torch.eye(4),
        lidar_to_ego_pose_to_global_matrix=torch.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=to_frame if to_frame is not None else torch.eye(4),
    )


def _translation(x: float) -> torch.Tensor:
    """Build a transform that shifts points along x."""
    matrix = torch.eye(4)
    matrix[0, 3] = x
    return matrix


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
    stored_frames: Sequence[LiDARPointCloudSample] = (),
    scene_token: str | None = None,
) -> ModelGTSample:
    """Build a transformed sample with the requested ground truth and stored lidar frames."""
    points = _points(time_lags)
    return ModelGTSample(
        lidar_point_cloud_samples=[
            _lidar_frame("/data/scene/lidar/00010.pcd.bin", timestamp),
            *stored_frames,
        ],
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
        frame_meta=(
            FrameMetaSample(ego2global=torch.eye(4), scene_token=scene_token, timestamp=timestamp)
            if scene_token is not None
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
        self.transformed: list[ModelGTSample] = []

    def get_data_sample(self, index: int) -> ModelGTSample:
        """Return the sample of one index."""
        return self.samples[index]

    def apply_transforms(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Record every sample the pipeline receives and return it unchanged."""
        self.transformed.append(model_gt_sample)
        return model_gt_sample


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

    def split_sources(self, split: SplitType) -> Mapping[str, SimpleNamespace]:
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

    def predict_step(
        self, multi_task_batch_inputs: ModelBatchInputs, batch_idx: int
    ) -> ModelPredictions:
        """Return the fixed predictions."""
        del batch_idx
        return self.predict(multi_task_batch_inputs)


class _FixedVoxelizer:
    """Voxelize any batch into one voxel that assigns the points fixed voxel rows."""

    def __init__(self, point_voxel_indices: Sequence[int]) -> None:
        """Initialize the voxelizer from the voxel row of every point, -1 for a dropped point."""
        self.point_voxel_indices = point_voxel_indices

    def __call__(self, batch: ModelGTBatch) -> VoxelsData:
        """Return the fixed voxels, whatever the batch."""
        del batch
        return VoxelsData(
            voxels=torch.zeros((1, 1, 5)),
            coords=torch.zeros((1, 3), dtype=torch.int32),
            num_points=torch.ones(1, dtype=torch.int32),
            batch_indices=torch.zeros(1, dtype=torch.int32),
            point_voxel_indices=torch.tensor(self.point_voxel_indices, dtype=torch.int64),
            num_dropped_voxels=torch.tensor(0),
        )


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


def _multitask_predictions(inputs: ModelBatchInputs) -> ModelPredictions:
    """Predict one car and the segmentation of the three current-frame points."""
    del inputs
    return ModelPredictions(
        detection3d_predictions=_detection_predictions().detection3d_predictions,
        segmentation3d_predictions=_segmentation_predictions(3).segmentation3d_predictions,
    )


def _calibration_predictions(inputs: ModelBatchInputs) -> ModelPredictions:
    """Predict a matching calibration with high confidence."""
    del inputs
    return ModelPredictions(
        calibration_status_predictions=CalibrationStatusPredictions(
            probabilities=torch.tensor([[0.2, 0.8]])
        )
    )


def _keyframe_with_stored_frames(timestamp: float, scene_token: str = "scene-a") -> ModelGTSample:
    """Build a 1 Hz keyframe whose record stores a past frame and future frames at 10 Hz."""
    return _sample(
        segmentation=True,
        timestamp=timestamp,
        scene_token=scene_token,
        stored_frames=[
            _lidar_frame("past.bin", timestamp - 0.1, _translation(-1.0)),
            _lidar_frame("next1.bin", timestamp + 0.1, _translation(1.0)),
            _lidar_frame("next2.bin", timestamp + 0.2, _translation(2.0)),
            # Lands on the step of next2 and must not replace it.
            _lidar_frame("next2b.bin", timestamp + 0.24, _translation(2.4)),
            _lidar_frame("late.bin", timestamp + 1.1, _translation(11.0)),
        ],
    )


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


class _PreviewTestCase(unittest.TestCase):
    """Run previews into a fresh recording backend instead of a configured one."""

    def setUp(self) -> None:
        """Route every session the preview opens to a new recording backend."""
        self.backend = RecordingBackend()
        self.enterContext(
            patch.object(
                VisualizationSession,
                "from_config",
                return_value=VisualizationSession(self.backend),
            )
        )

    def run_preview(
        self,
        dataset: _SampleDataset,
        model: _PreviewModel | None = None,
        **config: object,
    ) -> RecordingBackend:
        """Run a CPU preview of the dataset and return the recording backend."""
        preview.run_visualization_preview(
            model, _PreviewDataModule(dataset), VisualizationPreviewConfig(device="cpu", **config)
        )
        return self.backend


class TestPreviewConfiguration(_PreviewTestCase):
    """Invalid preview settings are rejected with a message naming the problem."""

    def test_config_rejects_unknown_split(self) -> None:
        """A split name outside the known splits is rejected by the configuration."""
        with self.assertRaisesRegex(ValueError, "split"):
            VisualizationPreviewConfig(split="validation")

    def test_prediction_mode_requires_a_model(self) -> None:
        """The predictions mode cannot run without a model."""
        with self.assertRaisesRegex(ValueError, "Model must be provided"):
            self.run_preview(_SampleDataset([_sample(segmentation=True)]), mode="predictions")

    def test_sample_index_out_of_range_names_the_split_size(self) -> None:
        """An out-of-range sample index reports how many samples the split has."""
        with self.assertRaisesRegex(IndexError, "with 1 samples"):
            self.run_preview(_SampleDataset([_sample(segmentation=True)]), sample_index=3)

    def test_prediction_frequency_needs_predictions(self) -> None:
        """Intermediate frames carry predictions only, so the data mode rejects them."""
        with self.assertRaisesRegex(ValueError, "needs the predictions mode"):
            self.run_preview(
                _SampleDataset([_sample(segmentation=True)]), prediction_frequency_hz=10.0
            )


class TestDataPreview(_PreviewTestCase):
    """A preview without a model shows the ground truth labeled with the split's classes."""

    def test_data_preview_shows_the_current_frame_ground_truth(self) -> None:
        """Sweep points carry no annotation of their own, so they stay out of the preview."""
        backend = self.run_preview(_SampleDataset([_sample(segmentation=True)]))

        ground_truth = _point_cloud(backend, "scene/ground_truth/segmentation")
        self.assertEqual(ground_truth.positions.shape, (3, 3))
        self.assertEqual(ground_truth.class_ids.tolist(), [0, 1, 0])
        self.assertEqual(backend.steps, [0])
        self.assertEqual(backend.timestamps, [1.0])
        self.assertTrue(backend.waited)

    def test_preview_labels_classes_from_the_database_taxonomy(self) -> None:
        """Without configured class names the legend uses the database taxonomy."""
        backend = self.run_preview(_SampleDataset([_sample(segmentation=True)]))

        self.assertLessEqual(set(_SEGMENTATION_CLASSES), _legend_labels(backend))

    def test_configured_class_names_override_the_taxonomy(self) -> None:
        """Configured segmentation class names replace the taxonomy in the legend."""
        backend = self.run_preview(
            _SampleDataset([_sample(segmentation=True)]),
            segmentation_class_names=("ground", "vehicle"),
        )

        labels = _legend_labels(backend)
        self.assertLessEqual({"ground", "vehicle"}, labels)
        self.assertNotIn("road", labels)


class TestSegmentationPredictions(_PreviewTestCase):
    """Segmentation predictions are aligned with the points the model scored."""

    def test_voxel_predictions_align_with_the_scored_points(self) -> None:
        """A voxel head labels the current-frame points inside the grid, in point order."""
        # Point 1 falls outside the grid and point 3 belongs to a past sweep.
        model = _PreviewModel(
            lambda inputs: _segmentation_predictions(2), voxels=_FixedVoxelizer([0, -1, 0, 0])
        )
        backend = self.run_preview(_SampleDataset([_sample(segmentation=True)]), model)

        prediction = _point_cloud(backend, "scene/prediction/segmentation")
        ground_truth = _point_cloud(backend, "scene/ground_truth/segmentation")
        self.assertEqual(prediction.positions[:, 0].tolist(), [5.0, 7.0])
        self.assertEqual(ground_truth.class_ids.tolist(), [0, 0])
        self.assertTrue(any(e.path == "scene/prediction/entropy" for e in backend.events))
        self.assertTrue(model.evaluating)

    def test_point_level_predictions_cover_every_point(self) -> None:
        """A point head without voxels labels every point of the batch."""
        model = _PreviewModel(lambda inputs: _segmentation_predictions(4))
        backend = self.run_preview(_SampleDataset([_sample(segmentation=True)]), model)

        self.assertEqual(
            _point_cloud(backend, "scene/prediction/segmentation").positions.shape, (4, 3)
        )

    def test_misaligned_segmentation_predictions_are_reported(self) -> None:
        """Predictions matching neither the points nor the voxels raise an error."""
        model = _PreviewModel(lambda inputs: _segmentation_predictions(1))

        with self.assertRaisesRegex(ValueError, "Cannot align 1 segmentation predictions"):
            self.run_preview(_SampleDataset([_sample(segmentation=True)]), model)

    def test_unsupervised_segmentation_shows_no_ground_truth(self) -> None:
        """A dataset without segmentation supervision logs no segmentation ground truth."""
        model = _PreviewModel(lambda inputs: _segmentation_predictions(4))
        backend = self.run_preview(
            _SampleDataset([_sample(segmentation=True)], seg3d_supervised=False), model
        )

        self.assertNotIn("scene/ground_truth/segmentation", backend.paths_of(PointCloud3DEvent))


class TestDetectionPredictions(_PreviewTestCase):
    """Detection previews draw predicted boxes next to the supervised ground truth."""

    def test_detection_preview_draws_predicted_and_ground_truth_boxes(self) -> None:
        """Predicted and ground-truth boxes are logged on separate entities."""
        model = _PreviewModel(lambda inputs: _detection_predictions())
        backend = self.run_preview(_SampleDataset([_sample(num_boxes=2)]), model)

        boxes = {e.path: e for e in backend.events if isinstance(e, Boxes3DEvent)}
        self.assertEqual(boxes["scene/prediction/detections"].centers.shape, (1, 3))
        self.assertEqual(boxes["scene/ground_truth/detections"].centers.shape, (2, 3))

    def test_unsupervised_detection_shows_only_predictions(self) -> None:
        """A dataset without detection supervision logs only the predicted boxes."""
        model = _PreviewModel(lambda inputs: _detection_predictions())
        backend = self.run_preview(
            _SampleDataset([_sample(num_boxes=2)], det3d_supervised=False), model
        )

        self.assertEqual(backend.paths_of(Boxes3DEvent), ["scene/prediction/detections"])


class TestMultiTask(_PreviewTestCase):
    """A sample carrying both 3D tasks shows both of them."""

    def test_multitask_data_preview_shows_both_ground_truths(self) -> None:
        """The data preview logs the segmentation and detection ground truth."""
        backend = self.run_preview(_SampleDataset([_sample(num_boxes=1, segmentation=True)]))

        self.assertIn("scene/ground_truth/segmentation", backend.paths_of(PointCloud3DEvent))
        self.assertEqual(backend.paths_of(Boxes3DEvent), ["scene/ground_truth/detections"])

    def test_multitask_prediction_preview_decodes_both_tasks(self) -> None:
        """The prediction preview logs the predictions of both heads."""
        model = _PreviewModel(_multitask_predictions, voxels=_FixedVoxelizer([0, 0, 0, 0]))
        backend = self.run_preview(_SampleDataset([_sample(num_boxes=1, segmentation=True)]), model)

        self.assertIn("scene/prediction/segmentation", backend.paths_of(PointCloud3DEvent))
        self.assertEqual(
            set(backend.paths_of(Boxes3DEvent)),
            {"scene/prediction/detections", "scene/ground_truth/detections"},
        )


class TestCamerasAndTimeline(_PreviewTestCase):
    """Cameras come from the source sample and samples land on their dataset index."""

    def test_cameras_are_drawn_from_the_untransformed_sample(self) -> None:
        """The cameras of the source sample are logged even though transforms drop them."""
        tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))
        image_path = tmp_path / "front.jpg"
        cv2.imwrite(str(image_path), np.zeros((36, 64, 3), dtype=np.uint8))
        sample = _sample(segmentation=True, image_samples=[_camera(image_path)])
        backend = self.run_preview(_SampleDataset([sample]))

        self.assertIn("scene/cameras/CAM_FRONT", backend.paths_of(Transform3DEvent))
        self.assertIn("scene/cameras/CAM_FRONT", backend.paths_of(ImageEvent))

    def test_cameras_follow_the_lidar_augmentation_of_the_sample(self) -> None:
        """
        Input: a sample whose transforms flipped the points laterally and shifted them along x.
        Expected: the camera extrinsic is composed with the inverse augmentation, so the
            augmented points and boxes project where the untransformed ones did.
        Check: the logged camera transform equals the calibration times the inverse
            augmentation.
        """
        tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))
        image_path = tmp_path / "front.jpg"
        cv2.imwrite(str(image_path), np.zeros((36, 64, 3), dtype=np.uint8))
        camera = _camera(image_path)
        augmentation = torch.eye(4)
        augmentation[1, 1] = -1.0
        augmentation[0, 3] = 2.0
        sample = _sample(segmentation=True, image_samples=[camera])._replace(
            lidar_transformation_sample=LiDARTransformationSample(
                transformation_matrix=augmentation,
                transformation_order=[
                    TransformationName.HORIZONTAL_FLIP,
                    TransformationName.TRANSLATION,
                ],
            )
        )
        backend = self.run_preview(_SampleDataset([sample]))

        expected = (camera.lidar2cam @ torch.linalg.inv(augmentation)).numpy()
        transform = next(
            event
            for event in backend.events
            if isinstance(event, Transform3DEvent) and event.path == "scene/cameras/CAM_FRONT"
        )
        np.testing.assert_allclose(transform.rotation_matrix, expected[:3, :3], atol=1e-6)
        np.testing.assert_allclose(transform.translation, expected[:3, 3], atol=1e-6)

    def test_consecutive_samples_are_logged_on_their_dataset_index(self) -> None:
        """Each sample is logged on the step of its dataset index and its own timestamp."""
        samples = [_sample(segmentation=True, timestamp=float(index)) for index in range(4)]
        backend = self.run_preview(_SampleDataset(samples), sample_index=1, max_samples=2)

        self.assertEqual(backend.steps, [1, 2])
        self.assertEqual(backend.timestamps, [1.0, 2.0])


class TestCalibrationStatus(_PreviewTestCase):
    """Calibration-status samples are previewed by the calibration adapter."""

    def test_calibration_preview_logs_the_status_verdict(self) -> None:
        """A calibration prediction logs the verdict document and the fused camera image."""
        model = _PreviewModel(_calibration_predictions)
        backend = self.run_preview(_SampleDataset([_calibration_sample(status=1)]), model)

        verdict = next(e for e in backend.events if isinstance(e, TextDocumentEvent))
        self.assertEqual(verdict.path, "calibration_status/status/verdict")
        self.assertEqual(verdict.text.splitlines()[0], "# MATCH")
        self.assertIn("calibration_status/camera/fused", backend.paths_of(ImageEvent))

    def test_calibration_data_preview_is_logged_under_the_dataset_root(self) -> None:
        """A calibration data preview logs its camera under the dataset entity root."""
        backend = self.run_preview(_SampleDataset([_calibration_sample(status=0)]))

        self.assertEqual(backend.paths_of(Transform3DEvent), ["dataset/calibration_status/camera"])


class TestNoopSession(unittest.TestCase):
    """The configured session backend drives a complete preview."""

    def test_noop_session_runs_a_full_preview(self) -> None:
        """The noop backend lets the whole pipeline run without the Rerun SDK."""
        datamodule = _PreviewDataModule(_SampleDataset([_sample(segmentation=True)]))

        count = preview.run_visualization_preview(
            None,
            datamodule,
            VisualizationPreviewConfig(
                device="cpu", session=VisualizationSessionConfig(backend="noop")
            ),
        )

        self.assertEqual(count, 1)
        self.assertTrue(datamodule.prepared)
        self.assertEqual(datamodule.stages, ["test"])


class TestIntermediateFrames(_PreviewTestCase):
    """Stored lidar frames between two samples are predicted at the requested frequency."""

    def test_frames_between_samples_are_predicted_on_their_own_steps(self) -> None:
        """Stored frames fill the steps up to the next sample of the same scene."""
        dataset = _SampleDataset(
            [_keyframe_with_stored_frames(1.0), _keyframe_with_stored_frames(2.0)]
        )
        model = _PreviewModel(lambda inputs: _segmentation_predictions(4))
        backend = self.run_preview(dataset, model, max_samples=1, prediction_frequency_hz=10.0)

        self.assertEqual(backend.steps, [0, 1, 2])
        np.testing.assert_allclose(backend.timestamps, [1.0, 1.1, 1.2], rtol=1e-6)
        # The keyframe is the only frame with ground truth.
        point_cloud_paths = list(backend.paths_of(PointCloud3DEvent))
        self.assertEqual(point_cloud_paths.count("scene/ground_truth/segmentation"), 1)
        self.assertEqual(point_cloud_paths.count("scene/prediction/segmentation"), 3)

    def test_last_sample_of_a_scene_keeps_its_stored_frames_within_one_stride(self) -> None:
        """Without a following sample of its scene a sample keeps the frames of one stride."""
        dataset = _SampleDataset(
            [_keyframe_with_stored_frames(1.0), _keyframe_with_stored_frames(2.0, "scene-b")]
        )
        model = _PreviewModel(lambda inputs: _segmentation_predictions(4))
        backend = self.run_preview(dataset, model, max_samples=2, prediction_frequency_hz=10.0)

        # The frame 1.1 s after the first keyframe would land on the next sample's step.
        self.assertEqual(backend.steps, [0, 1, 2, 10, 11, 12])

    def test_samples_one_frame_apart_have_no_intermediate_frames(self) -> None:
        """A database serving every lidar frame as a sample already shows every frame."""
        dataset = _SampleDataset(
            [_keyframe_with_stored_frames(1.0), _keyframe_with_stored_frames(1.1)]
        )
        model = _PreviewModel(lambda inputs: _segmentation_predictions(4))
        backend = self.run_preview(dataset, model, max_samples=1, prediction_frequency_hz=10.0)

        self.assertEqual(backend.steps, [0])

    def test_intermediate_frames_are_reanchored_and_unlabeled(self) -> None:
        """An intermediate frame becomes the current frame and drops the record's annotations."""
        dataset = _SampleDataset([_keyframe_with_stored_frames(1.0)])
        dataset.samples[0] = dataset.samples[0]._replace(detection3d_gt_bboxes_3d=_boxes(2))
        model = _PreviewModel(lambda inputs: _segmentation_predictions(4))
        self.run_preview(dataset, model, max_samples=1, prediction_frequency_hz=10.0)

        # The keyframe goes through the pipeline first, then the frame 0.1 s after it.
        intermediate = dataset.transformed[1]
        frames = intermediate.lidar_point_cloud_samples
        self.assertEqual(
            [frame.point_cloud_path for frame in frames],
            [
                "next1.bin",
                "/data/scene/lidar/00010.pcd.bin",
                "past.bin",
                "next2.bin",
                "next2b.bin",
                "late.bin",
            ],
        )
        shifts = [float(frame.lidar_sensor_to_lidar_sweep_matrix[0, 3]) for frame in frames]
        np.testing.assert_allclose(shifts, [0.0, -1.0, -2.0, 1.0, 1.4, 10.0], rtol=1e-6, atol=1e-6)
        self.assertIsNone(intermediate.segmentation3d_gt_sample)
        self.assertIsNone(intermediate.image_samples)
        self.assertEqual(intermediate.frame_meta.scene_token, "scene-a")
        self.assertAlmostEqual(intermediate.frame_meta.timestamp, 1.1)
        self.assertEqual(len(intermediate.detection3d_gt_bboxes_3d), 0)
        # The record itself keeps its annotations.
        self.assertEqual(len(dataset.samples[0].detection3d_gt_bboxes_3d), 2)


if __name__ == "__main__":
    unittest.main()
