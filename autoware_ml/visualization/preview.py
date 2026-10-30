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

"""Sample-at-a-time visualization preview of a datamodule split and a model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal, Protocol

from jaxtyping import Float32
import numpy as np
import torch

from autoware_ml.databases.taxonomy import DatabaseTaxonomy
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch, ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.datamodule.base_dataset import BaseDataset, ConcatDataset
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.types.dataset import SplitType
from autoware_ml.visualization.entity_paths import (
    CALIBRATION_STATUS_PATH,
    DATA_CALIBRATION_STATUS_PATH,
)
from autoware_ml.visualization.sensors.camera import (
    CameraBoxProjection,
    CameraCalibration,
    CameraPointProjection,
)
from autoware_ml.visualization.viewers.contracts import VisualizationSessionConfig
from autoware_ml.visualization.session import VisualizationSession

#: Lightning stage that builds the dataset of each split.
_SPLIT_STAGES: dict[SplitType, str] = {
    SplitType.TRAIN: "fit",
    SplitType.VAL: "validate",
    SplitType.TEST: "test",
    SplitType.PREDICT: "predict",
}


class PreviewModel(Protocol):
    """The part of a Lightning model the preview drives."""

    def to(self, device: torch.device) -> PreviewModel:
        """Move the model to a device."""

    def eval(self) -> PreviewModel:
        """Put the model in evaluation mode."""

    def transfer_batch_to_device(
        self, batch: ModelGTBatch, device: torch.device, dataloader_idx: int
    ) -> ModelGTBatch:
        """Move a collated batch to the device of the model."""

    def on_after_batch_transfer(self, batch: ModelGTBatch, dataloader_idx: int) -> ModelBatchInputs:
        """Preprocess a batch on the device into model inputs."""

    def predict_step(
        self, multi_task_batch_inputs: ModelBatchInputs, batch_idx: int
    ) -> ModelPredictions:
        """Decode the predictions of a batch."""


@dataclass(frozen=True)
class VisualizationPreviewConfig:
    """Configure one visualization preview run."""

    mode: Literal["auto", "predictions", "data"] = "auto"
    split: str = SplitType.TEST.value
    sample_index: int = 0
    max_samples: int = 1
    prediction_frequency_hz: float | None = None
    device: str = "auto"
    point_labels: bool = False
    class_names: tuple[str, ...] | None = None
    segmentation_class_names: tuple[str, ...] | None = None
    detection_class_names: tuple[str, ...] | None = None
    session: VisualizationSessionConfig = field(default_factory=VisualizationSessionConfig)

    def __post_init__(self) -> None:
        """Reject invalid preview settings before dataset or model setup."""
        if self.mode not in {"auto", "predictions", "data"}:
            raise ValueError(f"Unknown visualization mode: {self.mode}")
        if self.split not in {split.value for split in SplitType}:
            raise ValueError(f"Unknown visualization split: {self.split}")
        if self.sample_index < 0:
            raise ValueError("sample_index must not be negative")
        if self.max_samples <= 0:
            raise ValueError("max_samples must be greater than zero")
        if self.prediction_frequency_hz is not None and self.prediction_frequency_hz <= 0:
            raise ValueError("prediction_frequency_hz must be greater than zero")
        resolve_preview_device(self.device)


@dataclass(frozen=True)
class _PreviewLabels:
    """Class names and ignore index the adapters label a split with."""

    segmentation_class_names: Sequence[str]
    detection_class_names: Sequence[str]
    segmentation_ignore_index: int


@dataclass(frozen=True)
class _PreviewSample:
    """One collated sample, the model's view of it, and its source metadata."""

    batch: ModelGTBatch
    inputs: ModelBatchInputs | None
    predictions: ModelPredictions | None
    metadata: ModelGTSample
    det3d_supervised: bool
    seg3d_supervised: bool
    #: Augmentation the transforms applied to the points and boxes, mapping the frame of the
    #: untransformed sample to the frame of the batch. None when nothing moved them.
    lidar_augmentation: Float32[torch.Tensor, "4 4"] | None = None

    @property
    def scored_batch(self) -> ModelGTBatch:
        """The batch the model saw after preprocessing, or the collated batch without a model.

        Predictions are aligned with the points of the preprocessed batch, which may crop or
        reorder the collated points, so every 3D layer of a prediction preview reads it.
        """
        return self.inputs.multi_task_gt_batch if self.inputs is not None else self.batch


def resolve_preview_device(device: str) -> torch.device:
    """Resolve the preview execution device."""
    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def run_visualization_preview(
    model: PreviewModel | None,
    datamodule: DataModule,
    config: VisualizationPreviewConfig,
) -> int:
    """Render consecutive samples of a split through a visualization session.

    Args:
        model: Model whose predictions are shown, None to preview the transformed data.
        datamodule: Datamodule serving the split.
        config: Preview settings.

    Returns:
        The number of logged frames.
    """
    mode = _resolve_preview_mode(config.mode, model)
    if mode == "predictions" and model is None:
        raise ValueError("Model must be provided when visualization mode is 'predictions'.")
    if mode == "data" and config.prediction_frequency_hz is not None:
        raise ValueError(
            "prediction_frequency_hz adds prediction-only frames between the samples, so it "
            "needs the predictions mode and weights."
        )

    split = SplitType(config.split)
    dataset = _setup_split(datamodule, split)
    if config.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {config.sample_index} is out of range for split '{split.value}' "
            f"with {len(dataset)} samples."
        )
    labels = _resolve_labels(config, _split_taxonomy(datamodule, split))

    device = resolve_preview_device(config.device)
    if mode == "predictions" and model is not None:
        model.to(device)
        model.eval()
    session = VisualizationSession.from_config(config.session)

    indices = range(
        config.sample_index, min(len(dataset), config.sample_index + config.max_samples)
    )
    # A sample owns the timeline steps up to the next one, its intermediate frames fill them.
    stride = (
        max(1, round(config.prediction_frequency_hz))
        if config.prediction_frequency_hz is not None
        else 1
    )
    frame_count = 0
    with torch.no_grad():
        for dataset_index in indices:
            sample = _load_sample(
                dataset,
                dataset_index,
                model=model if mode == "predictions" else None,
                device=device,
                batch_index=frame_count,
            )
            keyframe_step = dataset_index * stride
            session.begin_frame(keyframe_step, timestamp=_sample_timestamp(sample.metadata))
            _log_preview_sample(session, sample, labels, config, dataset_index)
            frame_count += 1
            if config.prediction_frequency_hz is None or model is None:
                continue

            for step_offset, intermediate in _intermediate_frames(
                dataset, dataset_index, config.prediction_frequency_hz, stride
            ):
                sample = _predict_intermediate_frame(
                    dataset, dataset_index, intermediate, model, device, frame_count
                )
                session.begin_frame(
                    keyframe_step + step_offset, timestamp=_sample_timestamp(sample.metadata)
                )
                _log_preview_sample(session, sample, labels, config, dataset_index)
                frame_count += 1

    session.backend.wait_until_interrupted()
    return frame_count


def _resolve_preview_mode(
    mode: Literal["auto", "predictions", "data"], model: PreviewModel | None
) -> Literal["predictions", "data"]:
    """Resolve automatic preview mode selection."""
    if mode == "auto":
        return "predictions" if model is not None else "data"
    return mode


def _setup_split(datamodule: DataModule, split: SplitType) -> ConcatDataset:
    """Build the record tables when missing and return the dataset of one split."""
    datamodule.prepare_data()
    datamodule.setup(_SPLIT_STAGES[split])
    dataset = datamodule.datasets.get(split)
    if dataset is None:
        raise ValueError(f"Split '{split.value}' is not available for visualization.")
    return dataset


def _split_taxonomy(datamodule: DataModule, split: SplitType) -> DatabaseTaxonomy:
    """Return the taxonomy every source of the split shares."""
    sources = datamodule.split_sources(split)
    return next(iter(sources.values())).database.taxonomy


def _resolve_labels(
    config: VisualizationPreviewConfig, taxonomy: DatabaseTaxonomy
) -> _PreviewLabels:
    """Resolve the class names of both 3D tasks, explicit overrides first."""
    return _PreviewLabels(
        segmentation_class_names=config.segmentation_class_names
        or config.class_names
        or taxonomy.segmentation3d.class_names,
        detection_class_names=config.detection_class_names
        or config.class_names
        or taxonomy.detection3d.class_names,
        segmentation_ignore_index=taxonomy.segmentation3d.ignore_index,
    )


def _load_sample(
    dataset: ConcatDataset,
    dataset_index: int,
    *,
    model: PreviewModel | None,
    device: torch.device,
    batch_index: int,
) -> _PreviewSample:
    """Load, collate and, with a model, predict one sample of the split.

    The untransformed sample of the source dataset is kept next to the batch because the
    transform pipelines drop the camera files and the timestamps the cameras are drawn from.
    """
    source_index, sample_index = dataset.locate(dataset_index)
    source: BaseDataset = dataset.datasets[source_index]
    transformed = dataset[dataset_index]
    batch = dataset.collate_fn([transformed])
    inputs = None
    predictions = None
    if model is not None:
        inputs = model.on_after_batch_transfer(model.transfer_batch_to_device(batch, device, 0), 0)
        predictions = model.predict_step(inputs, batch_index)
    return _PreviewSample(
        batch=batch,
        inputs=inputs,
        predictions=predictions,
        metadata=source.get_data_sample(sample_index),
        det3d_supervised=source.det3d_supervised,
        seg3d_supervised=source.seg3d_supervised,
        lidar_augmentation=_lidar_augmentation(transformed),
    )


def _intermediate_frames(
    dataset: ConcatDataset, dataset_index: int, frequency_hz: float, stride: int
) -> Sequence[tuple[int, ModelGTSample]]:
    """Rebuild the unlabeled lidar frames recorded between a sample and the next one.

    Every record stores the lidar frames that follow its own, so the frames between two
    annotated samples are read from the first one. A frame is kept when it lands on a step of
    the requested frequency before the next sample of the same scene, one frame per step.

    Args:
        dataset: Dataset of the split.
        dataset_index: Index of the sample the frames follow.
        frequency_hz: Rate at which the frames are taken.
        stride: Timeline steps between two samples.

    Returns:
        The step offset after the sample and the re-anchored, unlabeled sample of every frame.
    """
    source_index, sample_index = dataset.locate(dataset_index)
    metadata = dataset.datasets[source_index].get_data_sample(sample_index)
    frames = metadata.lidar_point_cloud_samples
    if not frames:
        return []

    next_timestamp = math.inf
    if dataset_index + 1 < len(dataset):
        next_source_index, next_sample_index = dataset.locate(dataset_index + 1)
        following = dataset.datasets[next_source_index].get_data_sample(next_sample_index)
        if _same_scene(metadata, following) and following.lidar_point_cloud_samples:
            next_timestamp = following.lidar_point_cloud_samples[0].timestamp

    frame_by_step: dict[int, int] = {}
    for frame_index, frame in enumerate(frames[1:], start=1):
        offset = frame.timestamp - frames[0].timestamp
        if offset <= 0 or frame.timestamp >= next_timestamp:
            continue
        step_offset = round(offset * frequency_hz)
        if 1 <= step_offset < stride:
            frame_by_step.setdefault(step_offset, frame_index)
    return [
        (step_offset, _reanchor_on_frame(metadata, frame_by_step[step_offset]))
        for step_offset in sorted(frame_by_step)
    ]


def _same_scene(sample: ModelGTSample, other: ModelGTSample) -> bool:
    """Return whether two samples belong to one scene, False when either names none."""
    if sample.frame_meta is None or other.frame_meta is None:
        return False
    return sample.frame_meta.scene_token == other.frame_meta.scene_token


def _reanchor_on_frame(metadata: ModelGTSample, frame_index: int) -> ModelGTSample:
    """Make one stored lidar frame of a record the current frame of an unlabeled sample.

    The stored frames carry their transform from the record's own frame, so every one is
    re-expressed from the new current frame: a point maps to frame ``X`` through
    ``M_X @ inv(M_F)`` when ``M`` maps the record's frame to each stored one. The sweep
    loader then splits the frames into past and future around the new current frame, and the
    frame metadata takes the pose and the time of the new current frame. Annotations belong
    to the record's frame, so the boxes are emptied and the semantic mask is dropped, and the
    cameras, which the record lists for its own frame only, are left out.
    """
    frames = metadata.lidar_point_cloud_samples
    anchor = frames[frame_index]
    from_anchor = torch.linalg.inv(anchor.lidar_sensor_to_lidar_sweep_matrix)
    reanchored = [
        anchor._replace(
            lidar_sensor_to_lidar_sweep_matrix=torch.eye(
                4, dtype=anchor.lidar_sensor_to_lidar_sweep_matrix.dtype
            )
        )
    ] + [
        frame._replace(
            lidar_sensor_to_lidar_sweep_matrix=frame.lidar_sensor_to_lidar_sweep_matrix
            @ from_anchor
        )
        for index, frame in enumerate(frames)
        if index != frame_index
    ]
    boxes = metadata.detection3d_gt_bboxes_3d
    if boxes is not None:
        boxes = copy.deepcopy(boxes)
        boxes.remove_bboxes(torch.zeros(len(boxes), dtype=torch.bool))
    frame_meta = metadata.frame_meta
    if frame_meta is not None:
        frame_meta = frame_meta._replace(
            ego2global=anchor.lidar_to_ego_pose_to_global_matrix @ anchor.sensor_to_ego_pose_matrix,
            timestamp=float(anchor.timestamp),
        )
    return metadata._replace(
        lidar_point_cloud_samples=reanchored,
        image_samples=None,
        detection3d_gt_bboxes_3d=boxes,
        segmentation3d_gt_sample=None,
        frame_meta=frame_meta,
        detection3d_traffic_cone_barrier_bbox_status=None,
    )


def _predict_intermediate_frame(
    dataset: ConcatDataset,
    dataset_index: int,
    metadata: ModelGTSample,
    model: PreviewModel,
    device: torch.device,
    batch_index: int,
) -> _PreviewSample:
    """Transform, collate and predict one rebuilt frame with the pipeline of its source."""
    source_index, _ = dataset.locate(dataset_index)
    source: BaseDataset = dataset.datasets[source_index]
    transformed = source.apply_transforms(metadata)
    batch = dataset.collate_fn([transformed])
    inputs = model.on_after_batch_transfer(model.transfer_batch_to_device(batch, device, 0), 0)
    return _PreviewSample(
        batch=batch,
        inputs=inputs,
        predictions=model.predict_step(inputs, batch_index),
        metadata=metadata,
        det3d_supervised=False,
        seg3d_supervised=False,
        lidar_augmentation=_lidar_augmentation(transformed),
    )


def _log_preview_sample(
    session: VisualizationSession,
    sample: _PreviewSample,
    labels: _PreviewLabels,
    config: VisualizationPreviewConfig,
    dataset_index: int,
) -> None:
    """Dispatch one preview sample to the adapters of the tasks it carries."""
    predictions = sample.predictions
    sample_name = _sample_name(sample.metadata, dataset_index)
    image_batch = sample.batch.image_gt_batch
    if (image_batch is not None and image_batch.calibration_statuses is not None) or (
        predictions is not None and predictions.calibration_status_predictions is not None
    ):
        _log_calibration_preview(session, sample, sample_name)
        return

    scored_batch = sample.scored_batch
    has_segmentation = scored_batch.segmentation3d_gt_batch is not None or (
        predictions is not None and predictions.segmentation3d_predictions is not None
    )
    has_detection = scored_batch.detection3d_gt_batch is not None or (
        predictions is not None and predictions.detection3d_predictions is not None
    )
    if not has_segmentation and not has_detection:
        raise ValueError(
            "The sample carries neither 3D segmentation nor 3D detection ground truth or "
            "predictions, so there is nothing to visualize."
        )

    current_points = _current_frame_points(scored_batch)
    points = _to_numpy(current_points) if current_points is not None else None
    extent_points = points
    extent_boxes: Sequence[Float32[np.ndarray, "num_boxes box_params"]] = ()
    point_layers: dict[str, CameraPointProjection] = {}
    box_layers: dict[str, CameraBoxProjection] = {}

    if has_segmentation:
        segmentation_points = _log_segmentation(
            session, sample, labels, config, sample_name, point_layers
        )
        if segmentation_points is not None:
            extent_points = segmentation_points
    if has_detection:
        extent_boxes = _log_detection(
            session, sample, points, labels, config, sample_name, box_layers
        )
    if not point_layers and points is not None:
        point_layers["lidar"] = CameraPointProjection(points=points)

    # One union extent for both tasks, so sibling views share their initial eye.
    session.log_scene_extent(extent_points, boxes=extent_boxes)
    cameras = _camera_calibrations(sample.metadata.image_samples, sample.lidar_augmentation)
    if cameras:
        session.log_cameras(cameras, point_layers=point_layers, box_layers=box_layers)


def _log_segmentation(
    session: VisualizationSession,
    sample: _PreviewSample,
    labels: _PreviewLabels,
    config: VisualizationPreviewConfig,
    sample_name: str,
    point_layers: dict[str, CameraPointProjection],
) -> Float32[np.ndarray, "num_points num_features"] | None:
    """Log the segmentation layers of one sample and return the points they cover."""
    batch = sample.scored_batch
    ground_truth = None
    if sample.seg3d_supervised and batch.segmentation3d_gt_batch is not None:
        ground_truth = batch.segmentation3d_gt_batch.gt_semantic_masks
    predictions = (
        sample.predictions.segmentation3d_predictions if sample.predictions is not None else None
    )
    class_names = labels.segmentation_class_names

    if predictions is None:
        if ground_truth is None:
            return None
        mask = _current_frame_mask(batch)
        points = _to_numpy(_point_batch(batch).points[mask])
        gt_labels = _to_numpy(ground_truth[mask])
        session.log_segmentation3d_data(
            points,
            gt_labels,
            class_names=class_names,
            ignore_index=labels.segmentation_ignore_index,
            point_labels=config.point_labels,
            sample_name=sample_name,
        )
        point_layers["ground_truth/segmentation"] = CameraPointProjection(
            points=points,
            labels=gt_labels,
            class_names=class_names,
            ignore_index=labels.segmentation_ignore_index,
        )
        return points

    mask = _scored_point_mask(sample, predictions.pred_labels.shape[0])
    points = _to_numpy(_point_batch(batch).points[mask])
    gt_labels = _to_numpy(ground_truth[mask]) if ground_truth is not None else None
    pred_labels = _to_numpy(predictions.pred_labels)
    session.log_segmentation3d(
        points,
        pred_labels,
        pred_probs=_to_numpy(predictions.pred_probs),
        gt_labels=gt_labels,
        class_names=class_names,
        ignore_index=labels.segmentation_ignore_index,
        point_labels=config.point_labels,
        sample_name=sample_name,
    )
    point_layers["prediction/segmentation"] = CameraPointProjection(
        points=points, labels=pred_labels, class_names=class_names
    )
    if gt_labels is not None:
        point_layers["ground_truth/segmentation"] = CameraPointProjection(
            points=points,
            labels=gt_labels,
            class_names=class_names,
            ignore_index=labels.segmentation_ignore_index,
        )
    return points


def _log_detection(
    session: VisualizationSession,
    sample: _PreviewSample,
    points: Float32[np.ndarray, "num_points num_features"] | None,
    labels: _PreviewLabels,
    config: VisualizationPreviewConfig,
    sample_name: str,
    box_layers: dict[str, CameraBoxProjection],
) -> Sequence[Float32[np.ndarray, "num_boxes box_params"]]:
    """Log the detection layers of one sample and return the boxes they draw."""
    batch = sample.scored_batch
    gt_boxes = None
    gt_labels = None
    if sample.det3d_supervised and batch.detection3d_gt_batch is not None:
        gt_boxes = _to_numpy(batch.detection3d_gt_batch.valid_bboxes_3d()[0])
        gt_labels = _to_numpy(batch.detection3d_gt_batch.valid_labels_3d()[0])
    class_names = labels.detection_class_names
    predictions = (
        sample.predictions.detection3d_predictions if sample.predictions is not None else None
    )
    drawn_boxes: list[Float32[np.ndarray, "num_boxes box_params"]] = []

    if predictions is None:
        if gt_boxes is None:
            return drawn_boxes
        session.log_detection3d_data(
            points=points,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            class_names=class_names,
            sample_name=sample_name,
        )
    else:
        if len(predictions) != 1:
            raise ValueError(
                f"A preview batch holds one sample, got detections of {len(predictions)}."
            )
        pred_boxes = _to_numpy(predictions[0].bboxes_3d)
        pred_labels = _to_numpy(predictions[0].labels_3d)
        session.log_detection3d(
            pred_boxes=pred_boxes,
            pred_scores=_to_numpy(predictions[0].scores_3d),
            pred_labels=pred_labels,
            points=points,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            class_names=class_names,
            sample_name=sample_name,
        )
        box_layers["prediction/detections"] = CameraBoxProjection(
            boxes=pred_boxes, labels=pred_labels, class_names=class_names
        )
        drawn_boxes.append(pred_boxes)

    if gt_boxes is not None:
        box_layers["ground_truth/detections"] = CameraBoxProjection(
            boxes=gt_boxes, labels=gt_labels, class_names=class_names
        )
        drawn_boxes.append(gt_boxes)
    return drawn_boxes


def _log_calibration_preview(
    session: VisualizationSession, sample: _PreviewSample, sample_name: str
) -> None:
    """Log the first camera of a calibration-status sample.

    The images are read from the collated batch rather than the model inputs, so the viewer
    shows the pixels the pipeline produced and not a normalized copy.
    """
    image_batch = sample.batch.image_gt_batch
    if image_batch is None:
        raise ValueError("A calibration-status preview needs the images of the sample.")
    image = image_batch.images[0, 0]
    fused_image = None
    if image_batch.depth_maps is not None:
        fused_image = torch.cat((image, image_batch.depth_maps[0, 0]), dim=0)
    lidar_to_camera = image_batch.lidar2cams[0, 0]
    # lidar2images composes the intrinsics of the shown, augmented image with lidar2cams.
    image_from_camera = (image_batch.lidar2images[0, 0] @ torch.linalg.inv(lidar_to_camera))[:3, :3]

    gt_status = None
    if image_batch.calibration_statuses is not None:
        gt_status = int(image_batch.calibration_statuses[0, 0])
    pred_status = None
    pred_score = None
    predictions = sample.predictions
    if predictions is not None and predictions.calibration_status_predictions is not None:
        probabilities = predictions.calibration_status_predictions.probabilities[0]
        pred_status = int(probabilities.argmax())
        pred_score = float(probabilities.max())

    point_batch = sample.batch.point_cloud_gt_batch
    session.log_calibration_status(
        lidar_to_camera=_to_numpy(lidar_to_camera),
        image_from_camera=_to_numpy(image_from_camera),
        points=_to_numpy(point_batch.points) if point_batch is not None else None,
        image=_to_numpy(image),
        fused_image=_to_numpy(fused_image) if fused_image is not None else None,
        gt_status=gt_status,
        pred_status=pred_status,
        pred_score=pred_score,
        sample_name=sample_name,
        root_path=(
            CALIBRATION_STATUS_PATH if predictions is not None else DATA_CALIBRATION_STATUS_PATH
        ),
    )


def _point_batch(batch: ModelGTBatch) -> PointCloudGTBatch:
    """Return the point cloud of a batch, which every 3D preview needs."""
    if batch.point_cloud_gt_batch is None:
        raise ValueError("A 3D preview needs the point cloud of the sample.")
    return batch.point_cloud_gt_batch


def _current_frame_mask(batch: ModelGTBatch) -> torch.Tensor:
    """Mask the points captured in the current frame, leaving out the merged past sweeps."""
    point_batch = _point_batch(batch)
    if point_batch.timestamp_difference_dim < 0:
        return torch.ones(
            point_batch.points.shape[0], dtype=torch.bool, device=point_batch.points.device
        )
    return point_batch.points[:, point_batch.timestamp_difference_dim] == 0


def _current_frame_points(batch: ModelGTBatch) -> torch.Tensor | None:
    """Return the current-frame points of a batch, None when it carries no point cloud."""
    if batch.point_cloud_gt_batch is None:
        return None
    return batch.point_cloud_gt_batch.points[_current_frame_mask(batch)]


def _scored_point_mask(sample: _PreviewSample, num_predictions: int) -> torch.Tensor:
    """Mask the points a segmentation model labelled, in the order of its predictions.

    Voxel-based heads label the current-frame points that fall inside the voxel grid, and
    point-level heads label every point, so the mask is chosen by which one matches the number
    of predicted labels.
    """
    batch = sample.scored_batch
    mask = _current_frame_mask(batch)
    voxels = sample.inputs.voxels_data if sample.inputs is not None else None
    if voxels is not None:
        mask = mask & (voxels.point_voxel_indices >= 0)
    if int(mask.sum()) == num_predictions:
        return mask
    num_points = _point_batch(batch).points.shape[0]
    if num_points == num_predictions:
        return torch.ones(num_points, dtype=torch.bool, device=mask.device)
    raise ValueError(
        f"Cannot align {num_predictions} segmentation predictions with the {num_points} points "
        f"of the sample, {int(mask.sum())} of which are scored current-frame points."
    )


def _lidar_augmentation(transformed: ModelGTSample) -> Float32[torch.Tensor, "4 4"] | None:
    """
    Return the augmentation the transforms applied to the points and boxes of a sample.

    The geometric augmentations (rotation, scaling, translation and flips) record their
    composed 4x4 matrix in the sample, which maps the untransformed frame to the batch frame.

    Args:
        transformed: Sample after the transform pipeline.

    Returns:
        Float32[torch.Tensor, "4 4"] | None: The matrix, None when no augmentation ran.
    """
    if transformed.lidar_transformation_sample is None:
        return None
    return transformed.lidar_transformation_sample.transformation_matrix


def _camera_calibrations(
    image_samples: Sequence[ImageSample] | None,
    lidar_augmentation: Float32[torch.Tensor, "4 4"] | None,
) -> Mapping[str, CameraCalibration]:
    """
    Describe every camera of a sample by its image file and calibration.

    The calibration refers to the untransformed LiDAR frame, while the logged points and boxes
    are in the augmented one, so the inverse augmentation is composed into every extrinsic.
    Without it, an augmented training sample would project its overlays away from the objects
    in the images.

    Args:
        image_samples: Camera records of the untransformed sample, None when it has none.
        lidar_augmentation: Augmentation applied to the points and boxes, None for none.

    Returns:
        Mapping[str, CameraCalibration]: Calibration of every camera, by camera name.
    """
    from_batch_frame = (
        torch.linalg.inv(lidar_augmentation.to(torch.float64))
        if lidar_augmentation is not None
        else None
    )
    cameras: dict[str, CameraCalibration] = {}
    for image_sample in image_samples or ():
        lidar_to_camera = image_sample.lidar2cam
        if from_batch_frame is not None:
            lidar_to_camera = (lidar_to_camera.to(torch.float64) @ from_batch_frame).to(
                torch.float32
            )
        cameras[image_sample.camera_name] = CameraCalibration(
            image_path=image_sample.image_path,
            intrinsics=_to_numpy(image_sample.camera_intrinsic),
            lidar_to_camera=_to_numpy(lidar_to_camera),
        )
    return cameras


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Copy a tensor to a host NumPy array, the input every visualization adapter takes."""
    return tensor.detach().cpu().numpy()


def _sample_timestamp(metadata: ModelGTSample) -> float | None:
    """Return the timestamp of the current lidar frame, or of the first camera without one."""
    if metadata.lidar_point_cloud_samples:
        return float(metadata.lidar_point_cloud_samples[0].timestamp)
    if metadata.image_samples:
        return float(metadata.image_samples[0].timestamp)
    return None


def _sample_name(metadata: ModelGTSample, dataset_index: int) -> str:
    """Name a sample after its scene and the file of its current lidar frame or first camera."""
    if metadata.lidar_point_cloud_samples:
        file_name = Path(metadata.lidar_point_cloud_samples[0].point_cloud_path).name
    elif metadata.image_samples:
        file_name = Path(metadata.image_samples[0].image_path).name
    else:
        return f"sample-{dataset_index}"
    if metadata.frame_meta is not None:
        return f"{metadata.frame_meta.scene_token}/{file_name}"
    return file_name
