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

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

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
from autoware_ml.visualization.cameras import CameraBoxProjection, CameraPointProjection
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.session import VisualizationSession

PreviewMode = Literal["auto", "predictions", "data"]

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

    mode: PreviewMode = "auto"
    split: str = SplitType.TEST.value
    sample_index: int = 0
    max_samples: int = 1
    device: str = "auto"
    point_labels: bool = False
    class_names: tuple[str, ...] | None = None
    segmentation_class_names: tuple[str, ...] | None = None
    detection_class_names: tuple[str, ...] | None = None
    session: VisualizationSessionConfig = field(default_factory=VisualizationSessionConfig)
    point_color_mode: Literal["semantic", "intensity", "solid"] = "semantic"

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
        if self.point_color_mode not in {"semantic", "intensity", "solid"}:
            raise ValueError("point color mode must be 'semantic', 'intensity', or 'solid'")
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
    with torch.no_grad():
        for batch_index, dataset_index in enumerate(indices):
            sample = _load_sample(
                dataset,
                dataset_index,
                model=model if mode == "predictions" else None,
                device=device,
                batch_index=batch_index,
            )
            session.begin_frame(dataset_index, timestamp=_sample_timestamp(sample.metadata))
            _log_preview_sample(session, sample, labels, config, dataset_index)

    session.backend.wait_until_interrupted()
    return len(indices)


def _resolve_preview_mode(
    mode: PreviewMode, model: PreviewModel | None
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
    batch = dataset.collate_fn([dataset[dataset_index]])
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

    points = _current_frame_points(scored_batch)
    extent_points = points
    extent_boxes: list[torch.Tensor] = []
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
    session.log_scene_extent(extent_points, boxes=tuple(extent_boxes))
    cameras = _camera_metadata(sample.metadata.image_samples)
    if cameras:
        session.log_cameras(cameras, point_layers=point_layers, box_layers=box_layers)


def _log_segmentation(
    session: VisualizationSession,
    sample: _PreviewSample,
    labels: _PreviewLabels,
    config: VisualizationPreviewConfig,
    sample_name: str,
    point_layers: dict[str, CameraPointProjection],
) -> torch.Tensor | None:
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
        points = _point_batch(batch).points[mask]
        gt_labels = ground_truth[mask]
        session.log_segmentation3d_data(
            points,
            gt_labels,
            class_names=class_names,
            ignore_index=labels.segmentation_ignore_index,
            point_labels=config.point_labels,
            sample_name=sample_name,
            point_color_mode=config.point_color_mode,
            root_path="scene",
        )
        point_layers["ground_truth/segmentation"] = CameraPointProjection(
            points=points,
            labels=gt_labels,
            class_names=class_names,
            ignore_index=labels.segmentation_ignore_index,
        )
        return points

    mask = _scored_point_mask(sample, predictions.pred_labels.shape[0])
    points = _point_batch(batch).points[mask]
    gt_labels = ground_truth[mask] if ground_truth is not None else None
    session.log_segmentation3d(
        points,
        predictions.pred_labels,
        pred_probs=predictions.pred_probs,
        gt_labels=gt_labels,
        class_names=class_names,
        ignore_index=labels.segmentation_ignore_index,
        point_labels=config.point_labels,
        sample_name=sample_name,
        point_color_mode=config.point_color_mode,
        root_path="scene",
    )
    point_layers["prediction/segmentation"] = CameraPointProjection(
        points=points, labels=predictions.pred_labels, class_names=class_names
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
    points: torch.Tensor | None,
    labels: _PreviewLabels,
    config: VisualizationPreviewConfig,
    sample_name: str,
    box_layers: dict[str, CameraBoxProjection],
) -> list[torch.Tensor]:
    """Log the detection layers of one sample and return the boxes they draw."""
    batch = sample.scored_batch
    gt_boxes = None
    gt_labels = None
    if sample.det3d_supervised and batch.detection3d_gt_batch is not None:
        gt_boxes = batch.detection3d_gt_batch.valid_bboxes_3d()[0]
        gt_labels = batch.detection3d_gt_batch.valid_labels_3d()[0]
    class_names = labels.detection_class_names
    predictions = (
        sample.predictions.detection3d_predictions if sample.predictions is not None else None
    )
    drawn_boxes: list[torch.Tensor] = []

    if predictions is None:
        if gt_boxes is None:
            return drawn_boxes
        session.log_detection3d_data(
            points=points,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            class_names=class_names,
            sample_name=sample_name,
            point_color_mode=config.point_color_mode,
            root_path="scene",
        )
    else:
        if len(predictions) != 1:
            raise ValueError(
                f"A preview batch holds one sample, got detections of {len(predictions)}."
            )
        prediction = predictions[0]
        session.log_detection3d(
            {
                "bboxes_3d": prediction.bboxes_3d,
                "scores_3d": prediction.scores_3d,
                "labels_3d": prediction.labels_3d,
            },
            points=points,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            class_names=class_names,
            sample_name=sample_name,
            point_color_mode=config.point_color_mode,
            root_path="scene",
        )
        box_layers["prediction/detections"] = CameraBoxProjection(
            boxes=prediction.bboxes_3d, labels=prediction.labels_3d, class_names=class_names
        )
        drawn_boxes.append(prediction.bboxes_3d)

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
        lidar_to_camera=lidar_to_camera,
        image_from_camera=image_from_camera,
        points=point_batch.points if point_batch is not None else None,
        image=image,
        fused_image=fused_image,
        gt_status=gt_status,
        pred_status=pred_status,
        pred_score=pred_score,
        sample_name=sample_name,
        root_path="calibration_status" if predictions is not None else "dataset/calibration_status",
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


def _camera_metadata(image_samples: Sequence[ImageSample] | None) -> dict[str, dict[str, object]]:
    """Describe every camera of a sample by its image file and calibration."""
    return {
        image_sample.camera_name: {
            "img_path": image_sample.image_path,
            "cam2img": image_sample.camera_intrinsic,
            "lidar2cam": image_sample.lidar2cam,
        }
        for image_sample in image_samples or ()
    }


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
