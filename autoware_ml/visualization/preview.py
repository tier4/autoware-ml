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

"""Sample-at-a-time visualization preview helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from autoware_ml.datamodule.base import DataModule
from autoware_ml.datamodule.pipeline_context import PipelineContext
from autoware_ml.models.base import BaseModel
from autoware_ml.visualization.common import as_numpy
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.cameras import CameraBoxProjection, CameraPointProjection
from autoware_ml.visualization.detection3d import (
    DETECTION_PREDICTION_KEY_SETS,
    normalize_detection_predictions,
)
from autoware_ml.visualization.session import VisualizationSession

#: Per-point ground-truth label keys, most detailed first. Shared by the task
#: matcher and the label lookup so both stay in sync.
SEGMENTATION_LABEL_KEYS: tuple[str, ...] = (
    "origin_segment",
    "segment",
    "pts_semantic_mask",
)

PreviewSplit = Literal["train", "val", "test", "predict"]
PreviewMode = Literal["auto", "predictions", "data"]
PreviewTask = Literal["calibration_status", "segmentation3d", "detection3d", "multi"]


@dataclass(frozen=True)
class VisualizationPreviewConfig:
    """Configure one visualization preview run."""

    mode: PreviewMode = "auto"
    split: PreviewSplit = "test"
    sample_index: int = 0
    max_samples: int = 1
    prediction_frequency_hz: float = 10.0
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
        if self.split not in {"train", "val", "test", "predict"}:
            raise ValueError(f"Unknown visualization split: {self.split}")
        if self.sample_index < 0:
            raise ValueError("sample_index must not be negative")
        if self.max_samples <= 0:
            raise ValueError("max_samples must be greater than zero")
        if self.prediction_frequency_hz <= 0:
            raise ValueError("prediction_frequency_hz must be greater than zero")
        if self.point_color_mode not in {"semantic", "intensity", "solid"}:
            raise ValueError("point color mode must be 'semantic', 'intensity', or 'solid'")
        resolve_preview_device(self.device)


def resolve_preview_device(device: str) -> torch.device:
    """Resolve the preview execution device."""
    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def run_visualization_preview(
    model: BaseModel | None,
    datamodule: DataModule,
    config: VisualizationPreviewConfig,
) -> int:
    """Render one or more task samples through a visualization session."""
    if config.max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")
    if config.prediction_frequency_hz <= 0:
        raise ValueError("prediction_frequency_hz must be greater than zero")
    if config.mode not in {"auto", "predictions", "data"}:
        raise ValueError(f"Unknown visualization mode: {config.mode}")

    mode = _resolve_preview_mode(config.mode, model)
    if mode == "predictions" and model is None:
        raise ValueError("Model must be provided when visualization mode is 'predictions'.")

    device = resolve_preview_device(config.device)
    preview_dataloader, preview_indices = _build_preview_dataloader(
        datamodule=datamodule,
        split=config.split,
        sample_index=config.sample_index,
        max_samples=config.max_samples,
    )

    if model is not None:
        model.to(device)
        model.eval()
    session = VisualizationSession.from_config(config.session)
    visualized_count = 0

    dataset = getattr(datamodule, f"{config.split}_dataset")
    intermediate_provider = getattr(dataset, "get_intermediate_prediction_infos", None)
    use_intermediate_predictions = mode == "predictions" and callable(intermediate_provider)
    timeline_stride = max(1, int(round(config.prediction_frequency_hz)))
    with torch.no_grad():
        for dataset_index, batch in zip(preview_indices, preview_dataloader, strict=True):
            raw_info = dataset.get_data_info(dataset_index)
            keyframe_step = (
                dataset_index * timeline_stride if use_intermediate_predictions else dataset_index
            )
            _run_preview_frame(
                session=session,
                model=model,
                batch=batch,
                prediction_batch_index=visualized_count,
                timeline_step=keyframe_step,
                dataset_index=dataset_index,
                mode=mode,
                config=config,
                raw_info=raw_info,
                device=device,
            )
            visualized_count += 1

            if not use_intermediate_predictions:
                continue
            intermediate_infos = intermediate_provider(
                dataset_index,
                config.prediction_frequency_hz,
            )
            keyframe_timestamp = _get_sample_timestamp(raw_info)
            for fallback_offset, intermediate_info in enumerate(intermediate_infos, start=1):
                intermediate_timestamp = _get_sample_timestamp(intermediate_info)
                if keyframe_timestamp is not None and intermediate_timestamp is not None:
                    step_offset = max(
                        1,
                        int(
                            round(
                                (intermediate_timestamp - keyframe_timestamp)
                                * config.prediction_frequency_hz
                            )
                        ),
                    )
                else:
                    step_offset = fallback_offset
                intermediate_batch = _build_prediction_batch(
                    dataset,
                    datamodule,
                    intermediate_info,
                    dataset_index,
                )
                _run_preview_frame(
                    session=session,
                    model=model,
                    batch=intermediate_batch,
                    prediction_batch_index=visualized_count,
                    timeline_step=keyframe_step + step_offset,
                    dataset_index=dataset_index,
                    mode=mode,
                    config=config,
                    raw_info=intermediate_info,
                    device=device,
                )
                visualized_count += 1

    session.backend.wait_until_interrupted()
    return visualized_count


def _run_preview_frame(
    *,
    session: VisualizationSession,
    model: BaseModel | None,
    batch: dict[str, Any],
    prediction_batch_index: int,
    timeline_step: int,
    dataset_index: int,
    mode: PreviewMode,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any],
    device: torch.device,
) -> None:
    """Run and log one keyframe or prediction-only intermediate frame."""
    batch = _move_to_device(batch, device)
    if model is not None:
        batch = model.on_after_batch_transfer(batch, 0)
    predictions = (
        model.predict_step(batch, prediction_batch_index)
        if mode == "predictions" and model is not None
        else None
    )
    if mode == "predictions" and model is not None and _is_raw_multitask_outputs(predictions):
        build_eval_output = getattr(model, "build_eval_output", None)
        if build_eval_output is None:
            raise ValueError(
                "Multi-task prediction output requires the model to provide "
                "build_eval_output for visualization."
            )
        predictions = build_eval_output(batch, predictions)

    session.begin_frame(timeline_step, timestamp=_get_sample_timestamp(raw_info))
    _log_preview_sample(session, batch, predictions, dataset_index, mode, config, raw_info)


def _build_prediction_batch(
    dataset: Any,
    datamodule: DataModule,
    raw_info: dict[str, Any],
    dataset_index: int,
) -> dict[str, Any]:
    """Transform and collate an unlabeled intermediate record for inference."""
    context = PipelineContext(dataset=dataset, index=dataset_index)
    sample = dataset.apply_transforms(
        deepcopy(raw_info),
        datamodule.predict_transforms,
        context,
    )
    return datamodule.collate_fn([sample])


def _resolve_preview_mode(
    mode: PreviewMode, model: BaseModel | None
) -> Literal["predictions", "data"]:
    """Resolve automatic preview mode selection."""
    if mode == "auto":
        return "predictions" if model is not None else "data"
    return mode


def _build_preview_dataloader(
    datamodule: DataModule,
    split: PreviewSplit,
    sample_index: int,
    max_samples: int,
) -> tuple[DataLoader[Any], list[int]]:
    """Build a one-sample preview dataloader for the selected split."""
    stage_map: dict[PreviewSplit, str] = {
        "train": "fit",
        "val": "validate",
        "test": "test",
        "predict": "predict",
    }
    datamodule.setup(stage_map[split])
    dataset = getattr(datamodule, f"{split}_dataset")
    if dataset is None:
        raise ValueError(f"Split '{split}' is not available for visualization.")
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {sample_index} is out of range for split '{split}' "
            f"with {len(dataset)} samples."
        )

    preview_indices = list(range(sample_index, min(len(dataset), sample_index + max_samples)))
    subset = Subset(dataset, preview_indices)
    return (
        DataLoader(
            dataset=subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
            collate_fn=datamodule.collate_fn,
        ),
        preview_indices,
    )


def _move_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensor-like preview data to the target device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    return data


def _log_preview_sample(
    session: VisualizationSession,
    batch: dict[str, Any],
    predictions: Any,
    dataset_index: int,
    mode: PreviewMode,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any] | None = None,
) -> None:
    """Dispatch one preview sample to the task-appropriate visualization adapter."""
    _validate_preview_sample(batch)
    task = _infer_preview_task(batch, predictions)
    sample_name = _infer_sample_name(batch, dataset_index)

    if task == "calibration_status":
        _log_calibration_preview(session, batch, predictions, sample_name, mode)
        return
    if task == "segmentation3d":
        if mode == "data":
            _log_segmentation_data_preview(session, batch, sample_name, config, raw_info)
            return
        _log_segmentation_preview(session, batch, predictions, sample_name, config, raw_info)
        return
    if task == "detection3d":
        if mode == "data":
            _log_detection_data_preview(session, batch, sample_name, config, raw_info)
            return
        detection_predictions = _extract_single_detection_prediction(predictions)
        if detection_predictions is None:
            raise ValueError("Could not normalize detection predictions for visualization.")
        _log_detection_preview(session, batch, detection_predictions, sample_name, config, raw_info)
        return
    if task == "multi":
        _log_multitask_preview(session, batch, predictions, sample_name, config, raw_info, mode)
        return

    raise ValueError(f"Unsupported visualization task: {task}")


def _is_calibration_sample(batch: dict[str, Any]) -> bool:
    """Return whether the batch carries a calibration-status sample.

    The fused image and the status label are the model-facing keys, so a sample
    is recognized by them even when ``calibration_data`` was dropped during
    collation. Routing on them lets the adapter report the missing entry
    instead of the sample matching no task at all.
    """
    return bool({"calibration_data", "gt_calibration_status", "fused_img"} & batch.keys())


def _infer_preview_task(batch: dict[str, Any], predictions: Any) -> PreviewTask:
    """Infer which task adapter should handle the preview sample.

    Each task is matched on the full set of keys its adapter requires, so a
    sample that satisfies more than one contract is reported as ambiguous
    instead of being silently routed to whichever check happens to run first.
    """
    if _is_multitask_predictions(predictions) or (
        _has_segmentation_sample(batch) and _has_detection_sample(batch)
    ):
        return "multi"

    matches: list[PreviewTask] = []
    if _is_calibration_sample(batch):
        matches.append("calibration_status")
    if _has_segmentation_sample(batch) or _is_segmentation_predictions(predictions):
        matches.append("segmentation3d")
    if _has_detection_sample(batch) or _is_detection_predictions(predictions):
        matches.append("detection3d")

    if len(matches) == 1:
        return matches[0]

    observed_keys = ", ".join(sorted(batch)) or "<none>"
    if not matches:
        raise ValueError(
            "Could not infer a visualization task from the current batch and predictions. "
            f"Observed batch keys: {observed_keys}."
        )
    raise ValueError(
        f"Ambiguous visualization task: sample matches {', '.join(matches)}. "
        f"Observed batch keys: {observed_keys}."
    )


def _validate_preview_sample(batch: dict[str, Any]) -> None:
    """Reject incomplete task contracts before dispatch can hide them."""
    has_boxes = batch.get("gt_boxes") is not None
    has_labels = batch.get("gt_labels") is not None
    if has_boxes != has_labels:
        missing = "gt_labels" if has_boxes else "gt_boxes"
        raise ValueError(
            "Detection ground truth is incomplete: "
            f"{missing!r} is required when its paired field is present."
        )


def _resolve_class_names(
    config: VisualizationPreviewConfig,
    batch: dict[str, Any],
    raw_info: dict[str, Any] | None,
    *,
    task: Literal["segmentation3d", "detection3d"],
) -> Sequence[str] | None:
    """Resolve semantic class names for one preview sample.

    Class names decide whether the viewer legend and per-instance labels read as
    names or as bare integer ids, so they are looked up from every source that
    can carry them. Split transform pipelines drop ``class_names`` before
    collation, which leaves the batch empty for every task, so the raw dataset
    info is consulted as well; segmentation pipelines carry them in neither and
    depend on the configured value.
    """
    configured_names = (
        config.segmentation_class_names
        if task == "segmentation3d"
        else config.detection_class_names
    )
    for candidate in (
        configured_names,
        config.class_names,
        _unwrap_single_item(batch.get("class_names")),
        (raw_info or {}).get("class_names"),
    ):
        if candidate is not None and len(candidate) > 0:
            return candidate
    return None


def _has_segmentation_sample(batch: dict[str, Any]) -> bool:
    """Return whether the batch follows the segmentation3d schema.

    PTv3 pipelines drop raw ``points`` during grid sampling and carry positions
    in ``coord`` instead, so either key counts as a point source. Without this
    the transformed-data preview could not route any PTv3 segmentation config,
    because that path has no predictions to fall back on.
    """
    has_positions = batch.get("points") is not None or batch.get("coord") is not None
    has_labels = any(batch.get(key) is not None for key in SEGMENTATION_LABEL_KEYS)
    return has_positions and has_labels


def _has_detection_sample(batch: dict[str, Any]) -> bool:
    """Return whether the batch follows the detection3d schema."""
    return batch.get("gt_boxes") is not None and batch.get("gt_labels") is not None


def _is_segmentation_predictions(predictions: Any) -> bool:
    """Return whether prediction outputs match the segmentation contract."""
    return isinstance(predictions, dict) and "pred_labels" in predictions


def _is_detection_predictions(predictions: Any) -> bool:
    """Return whether prediction outputs carry a decoded 3D detection key set."""
    candidate = _extract_single_detection_prediction(predictions)
    if candidate is None:
        return False
    return any(key_set <= candidate.keys() for key_set in DETECTION_PREDICTION_KEY_SETS)


def _is_multitask_predictions(predictions: Any) -> bool:
    """Return whether outputs contain decoded multi-task branches."""
    if not isinstance(predictions, dict):
        return False
    return {"predictions", "seg_pred_labels"} <= predictions.keys() or {
        "seg_logits",
        "det_outputs",
    } <= predictions.keys()


def _is_raw_multitask_outputs(predictions: Any) -> bool:
    """Return whether a model exposed undecoded joint branch outputs."""
    return (
        isinstance(predictions, dict)
        and {
            "seg_logits",
            "det_outputs",
        }
        <= predictions.keys()
    )


def _log_calibration_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    predictions: Any,
    sample_name: str,
    mode: PreviewMode,
) -> None:
    """Render one calibration-status preview sample."""
    if "calibration_data" not in batch:
        raise ValueError(
            "A calibration-status preview needs the 'calibration_data' entry, but this "
            "batch does not carry it. collation_map is a whitelist and the shipped "
            "calibration configs list only the model inputs, so the entry the transforms "
            "produce is dropped. Carry it, the camera image, and the point cloud through, "
            "for example with '+datamodule.collation_map.calibration_data=list "
            "+datamodule.collation_map.img=list +datamodule.collation_map.points=list'."
        )
    calibration_data = _unwrap_single_item(batch["calibration_data"])
    pred_status: int | None = None
    root_path = "dataset/calibration_status" if mode == "data" else "calibration_status"
    if mode == "predictions":
        pred_probs = as_numpy(predictions)
        pred_probs = pred_probs[0] if pred_probs.ndim > 1 else pred_probs
        pred_status = int(pred_probs.argmax()) if pred_probs.size > 0 else None
        pred_score = float(pred_probs.max()) if pred_probs.size > 0 else None
    else:
        pred_score = None

    gt_status = batch.get("gt_calibration_status")
    if isinstance(gt_status, torch.Tensor):
        gt_status_value = int(gt_status.reshape(-1)[0].item())
    elif isinstance(gt_status, list) and gt_status:
        gt_status_value = int(gt_status[0])
    else:
        gt_status_value = None

    session.log_calibration_status(
        calibration_data,
        points=_unwrap_single_item(batch.get("points")),
        image=_unwrap_single_item(batch.get("img")),
        fused_image=_unwrap_single_item(batch.get("fused_img")),
        gt_status=gt_status_value,
        pred_status=pred_status,
        pred_score=pred_score,
        sample_name=sample_name,
        root_path=root_path,
    )


def _log_segmentation_data_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    sample_name: str,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any] | None = None,
    *,
    log_cameras: bool = True,
    log_extent: bool = True,
) -> None:
    """Render one transformed segmentation sample without predictions."""
    gt_labels = _get_segmentation_gt_labels(batch)
    points = _get_segmentation_points(batch, gt_labels)
    class_names = _resolve_class_names(config, batch, raw_info, task="segmentation3d")
    session.log_segmentation3d_data(
        points,
        _unwrap_single_item(gt_labels),
        class_names=class_names,
        point_labels=config.point_labels,
        sample_name=sample_name,
        point_color_mode=config.point_color_mode,
        root_path="scene",
    )
    if log_extent:
        session.log_scene_extent(points)
    if log_cameras:
        _log_camera_preview(
            session,
            raw_info,
            point_layers={
                "ground_truth/segmentation": CameraPointProjection(
                    points=points,
                    labels=_unwrap_single_item(gt_labels),
                    class_names=class_names,
                )
            },
        )


def _log_detection_data_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    sample_name: str,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any] | None = None,
    *,
    log_cameras: bool = True,
    log_extent: bool = True,
) -> None:
    """Render one transformed detection sample without predictions."""
    points = _unwrap_single_item(batch.get("points"))
    gt_boxes = _unwrap_single_item(batch.get("gt_boxes"))
    gt_labels = _unwrap_single_item(batch.get("gt_labels"))
    class_names = _resolve_class_names(config, batch, raw_info, task="detection3d")
    session.log_detection3d_data(
        points=points,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        class_names=class_names,
        sample_name=sample_name,
        point_color_mode=config.point_color_mode,
        root_path="scene",
    )
    if log_extent:
        session.log_scene_extent(points, boxes=(gt_boxes,))
    if log_cameras:
        _log_camera_preview(
            session,
            raw_info,
            point_layers=_lidar_camera_layer(points),
            box_layers={
                "ground_truth/detections": CameraBoxProjection(
                    boxes=gt_boxes,
                    labels=gt_labels,
                    class_names=class_names,
                )
            },
        )


def _log_segmentation_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    predictions: dict[str, Any],
    sample_name: str,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any] | None = None,
) -> None:
    """Render one 3D segmentation preview sample."""
    gt_labels = _get_segmentation_gt_labels(batch)
    pred_labels = predictions["pred_labels"]
    pred_logits = predictions.get("pred_logits")
    if pred_logits is None:
        raise ValueError(
            "Segmentation predictions must include 'pred_logits' for pointwise "
            "probability and entropy."
        )
    points = _get_segmentation_points(batch, pred_labels)
    class_names = _resolve_class_names(config, batch, raw_info, task="segmentation3d")
    session.log_segmentation3d(
        points,
        pred_labels,
        pred_probs=predictions.get("pred_probs"),
        pred_logits=pred_logits,
        gt_labels=_unwrap_single_item(gt_labels),
        class_names=class_names,
        point_labels=config.point_labels,
        sample_name=sample_name,
        point_color_mode=config.point_color_mode,
    )
    session.log_scene_extent(points)
    point_layers = {
        "prediction/segmentation": CameraPointProjection(
            points=points,
            labels=pred_labels,
            class_names=class_names,
        )
    }
    if gt_labels is not None:
        point_layers["ground_truth/segmentation"] = CameraPointProjection(
            points=points,
            labels=_unwrap_single_item(gt_labels),
            class_names=class_names,
        )
    _log_camera_preview(session, raw_info, point_layers=point_layers)


def _log_camera_preview(
    session: VisualizationSession,
    raw_info: dict[str, Any] | None,
    root_path: str = "scene/cameras",
    *,
    point_layers: dict[str, CameraPointProjection] | None = None,
    box_layers: dict[str, CameraBoxProjection] | None = None,
) -> None:
    """Log cameras and task overlays from raw, uncollated dataset metadata."""
    if raw_info is None:
        return
    if "images" not in raw_info:
        return
    images = raw_info.get("images")
    if not isinstance(images, Mapping):
        raise TypeError("Camera metadata 'images' must be a mapping")
    if not images:
        return
    session.log_cameras(
        dict(images),
        root_path=root_path,
        point_layers=point_layers,
        box_layers=box_layers,
    )


def _lidar_camera_layer(points: Any) -> dict[str, CameraPointProjection]:
    """Return a depth-colored camera layer when LiDAR points are available."""
    if points is None:
        return {}
    return {"lidar": CameraPointProjection(points=points)}


def _log_detection_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    predictions: dict[str, Any],
    sample_name: str,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any] | None = None,
) -> None:
    """Render one 3D detection preview sample."""
    points = _unwrap_single_item(batch.get("points"))
    gt_boxes = _unwrap_single_item(batch.get("gt_boxes"))
    gt_labels = _unwrap_single_item(batch.get("gt_labels"))
    class_names = _resolve_class_names(config, batch, raw_info, task="detection3d")
    normalized_predictions = normalize_detection_predictions(predictions)
    session.log_detection3d(
        predictions,
        points=points,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        class_names=class_names,
        sample_name=sample_name,
        point_color_mode=config.point_color_mode,
    )
    session.log_scene_extent(points, boxes=(normalized_predictions["boxes"], gt_boxes))
    box_layers = {
        "prediction/detections": CameraBoxProjection(
            boxes=normalized_predictions["boxes"],
            labels=normalized_predictions["labels"],
            class_names=class_names,
        )
    }
    if gt_boxes is not None and gt_labels is not None:
        box_layers["ground_truth/detections"] = CameraBoxProjection(
            boxes=gt_boxes,
            labels=gt_labels,
            class_names=class_names,
        )
    _log_camera_preview(
        session,
        raw_info,
        point_layers=_lidar_camera_layer(points),
        box_layers=box_layers,
    )


def _log_multitask_preview(
    session: VisualizationSession,
    batch: dict[str, Any],
    predictions: Any,
    sample_name: str,
    config: VisualizationPreviewConfig,
    raw_info: dict[str, Any] | None,
    mode: PreviewMode,
) -> None:
    """Render both branches of a combined detection/segmentation sample."""
    segmentation_class_names = _resolve_class_names(config, batch, raw_info, task="segmentation3d")
    detection_class_names = _resolve_class_names(config, batch, raw_info, task="detection3d")
    segmentation_ground_truth = _get_segmentation_gt_labels(batch)
    has_segmentation_ground_truth = _has_ground_truth(
        raw_info,
        "has_segmentation_ground_truth",
        segmentation_ground_truth,
    )
    detection_ground_truth_boxes = _unwrap_single_item(batch.get("gt_boxes"))
    detection_ground_truth_labels = _unwrap_single_item(batch.get("gt_labels"))
    has_detection_ground_truth = _has_ground_truth(
        raw_info,
        "has_detection_ground_truth",
        detection_ground_truth_boxes,
    )
    if mode == "data":
        if has_segmentation_ground_truth:
            _log_segmentation_data_preview(
                session,
                batch,
                sample_name,
                config,
                raw_info,
                log_cameras=False,
                log_extent=False,
            )
        if has_detection_ground_truth:
            _log_detection_data_preview(
                session,
                batch,
                sample_name,
                config,
                raw_info,
                log_cameras=False,
                log_extent=False,
            )
        # One union extent for both branches: two competing ones would leave
        # the sibling views with the bounds of whichever was logged last.
        session.log_scene_extent(
            _get_segmentation_points(batch, segmentation_ground_truth)
            if has_segmentation_ground_truth
            else _unwrap_single_item(batch.get("points")),
            boxes=(detection_ground_truth_boxes if has_detection_ground_truth else None,),
        )
        point_layers: dict[str, CameraPointProjection] = {}
        box_layers: dict[str, CameraBoxProjection] = {}
        if has_segmentation_ground_truth:
            gt_points = _get_segmentation_points(batch, segmentation_ground_truth)
            point_layers["ground_truth/segmentation"] = CameraPointProjection(
                points=gt_points,
                labels=_unwrap_single_item(segmentation_ground_truth),
                class_names=segmentation_class_names,
            )
        else:
            point_layers.update(_lidar_camera_layer(_unwrap_single_item(batch.get("points"))))
        if has_detection_ground_truth:
            box_layers["ground_truth/detections"] = CameraBoxProjection(
                boxes=detection_ground_truth_boxes,
                labels=detection_ground_truth_labels,
                class_names=detection_class_names,
            )
        _log_camera_preview(
            session,
            raw_info,
            point_layers=point_layers,
            box_layers=box_layers,
        )
        return
    if not isinstance(predictions, dict):
        raise TypeError("Multi-task predictions must be a decoded dictionary.")
    detection_predictions = predictions.get("predictions")
    if not isinstance(detection_predictions, (list, tuple)) or len(detection_predictions) != 1:
        raise ValueError("Multi-task detection predictions must contain exactly one sample.")
    segmentation_labels = predictions.get("seg_pred_labels")
    if segmentation_labels is None:
        raise ValueError("Multi-task predictions are missing segmentation outputs.")
    segmentation_logits = predictions.get("seg_pred_logits")
    if segmentation_logits is None:
        raise ValueError(
            "Multi-task predictions must include 'seg_pred_logits' for pointwise "
            "probability and entropy."
        )
    segmentation_points = _get_segmentation_points(batch, segmentation_labels)

    session.log_segmentation3d(
        segmentation_points,
        segmentation_labels,
        pred_logits=segmentation_logits,
        gt_labels=(
            _unwrap_single_item(segmentation_ground_truth)
            if has_segmentation_ground_truth
            else None
        ),
        class_names=segmentation_class_names,
        point_labels=config.point_labels,
        sample_name=sample_name,
        point_color_mode=config.point_color_mode,
        root_path="scene",
    )
    session.log_detection3d(
        detection_predictions[0],
        points=_unwrap_single_item(batch.get("points")),
        gt_boxes=(detection_ground_truth_boxes if has_detection_ground_truth else None),
        gt_labels=(detection_ground_truth_labels if has_detection_ground_truth else None),
        class_names=detection_class_names,
        sample_name=sample_name,
        point_color_mode=config.point_color_mode,
        root_path="scene",
    )
    normalized_predictions = normalize_detection_predictions(detection_predictions[0])
    session.log_scene_extent(
        segmentation_points,
        boxes=(
            normalized_predictions["boxes"],
            detection_ground_truth_boxes if has_detection_ground_truth else None,
        ),
    )
    point_layers = {
        "prediction/segmentation": CameraPointProjection(
            points=segmentation_points,
            labels=segmentation_labels,
            class_names=segmentation_class_names,
        )
    }
    if has_segmentation_ground_truth:
        point_layers["ground_truth/segmentation"] = CameraPointProjection(
            points=segmentation_points,
            labels=_unwrap_single_item(segmentation_ground_truth),
            class_names=segmentation_class_names,
        )
    box_layers = {
        "prediction/detections": CameraBoxProjection(
            boxes=normalized_predictions["boxes"],
            labels=normalized_predictions["labels"],
            class_names=detection_class_names,
        )
    }
    if has_detection_ground_truth:
        box_layers["ground_truth/detections"] = CameraBoxProjection(
            boxes=detection_ground_truth_boxes,
            labels=detection_ground_truth_labels,
            class_names=detection_class_names,
        )
    _log_camera_preview(
        session,
        raw_info,
        point_layers=point_layers,
        box_layers=box_layers,
    )


def _extract_single_detection_prediction(predictions: Any) -> dict[str, Any] | None:
    """Normalize one-sample detection predictions for visualization."""
    if isinstance(predictions, dict):
        return predictions
    if isinstance(predictions, (list, tuple)) and len(predictions) == 1:
        first_prediction = predictions[0]
        if isinstance(first_prediction, dict):
            return first_prediction
    return None


def _unwrap_single_item(value: Any) -> Any:
    """Unwrap the first (and only) item from a single-sample batch value.

    Handles both the old list-based collation format and the current
    tensor-based format where fixed-shape tensors are stacked along a new
    leading batch dimension.  Point-cloud tensors (ndim == 2) are left as-is
    since their leading dimension is the point count, not the batch size.
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    if isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


def _get_segmentation_gt_labels(batch: dict[str, Any]) -> Any:
    """Return the most detailed segmentation ground-truth labels available."""
    for key in SEGMENTATION_LABEL_KEYS:
        labels = batch.get(key)
        if labels is not None:
            return labels
    return None


def _get_segmentation_points(batch: dict[str, Any], labels: Any) -> Any:
    """Return point positions aligned with segmentation labels.

    PTv3 pipelines may drop raw ``points`` after formatting and grid sampling.
    In that case, ``inverse`` maps original points to sampled voxel
    representatives, so ``coord[inverse]`` yields visualization positions with
    the same length as point-level predictions and ``origin_segment`` labels.
    """
    point_count = _first_dimension(labels)
    points = _unwrap_single_item(batch.get("points"))
    if points is not None and _first_dimension(points) == point_count:
        return _append_aligned_intensity(points, batch, point_count)

    coord = _unwrap_single_item(batch.get("coord"))
    inverse = _unwrap_single_item(batch.get("inverse"))
    if coord is not None and inverse is not None and _first_dimension(inverse) == point_count:
        if isinstance(inverse, torch.Tensor):
            positions = coord[inverse.long()]
        else:
            positions = coord[inverse.astype(int)]
        return _append_aligned_intensity(
            positions,
            batch,
            point_count,
            inverse=inverse,
        )

    if coord is not None and _first_dimension(coord) == point_count:
        return _append_aligned_intensity(coord, batch, point_count)

    raise KeyError("Segmentation visualization requires 'points' or 'coord' aligned with labels.")


def _append_aligned_intensity(
    positions: Any,
    batch: dict[str, Any],
    point_count: int | None,
    *,
    inverse: Any | None = None,
) -> Any:
    """Append the PTv3 ``strength`` feature when positions only contain XYZ."""
    if point_count is None or positions.shape[1] >= 4:
        return positions

    strength = _unwrap_single_item(batch.get("origin_strength"))
    if strength is None:
        strength = _unwrap_single_item(batch.get("strength"))
    if strength is None:
        features = _unwrap_single_item(batch.get("feat"))
        coordinates = _unwrap_single_item(batch.get("coord"))
        if (
            features is not None
            and coordinates is not None
            and len(features.shape) == 2
            and len(coordinates.shape) == 2
            and features.shape[0] == coordinates.shape[0]
            and features.shape[1] == coordinates.shape[1] + 1
        ):
            # PTv3 BuildPointFeatures concatenates [coord, strength], while
            # the datamodule's explicit collation map retains only ``feat``.
            # Recover its final intensity channel for visualization.
            strength = features[:, -1]
    if strength is not None:
        if _first_dimension(strength) != point_count and inverse is not None:
            if isinstance(inverse, torch.Tensor):
                strength = strength[inverse.long()]
            else:
                strength = strength[inverse.astype(int)]
    if strength is None or _first_dimension(strength) != point_count:
        return positions

    if isinstance(positions, torch.Tensor):
        if not isinstance(strength, torch.Tensor):
            strength = torch.as_tensor(
                strength,
                dtype=positions.dtype,
                device=positions.device,
            )
        return torch.cat((positions, strength.reshape(-1, 1)), dim=1)

    positions_np = as_numpy(positions)
    strength_np = as_numpy(strength).reshape(-1, 1)
    return np.concatenate((positions_np, strength_np), axis=1)


def _has_ground_truth(
    raw_info: dict[str, Any] | None,
    flag_name: str,
    fallback_value: Any,
) -> bool:
    """Resolve explicit supervision provenance before falling back to data presence."""
    if raw_info is not None and flag_name in raw_info:
        return bool(raw_info[flag_name])
    return fallback_value is not None


def _get_sample_timestamp(raw_info: dict[str, Any] | None) -> float | None:
    """Return one numeric source timestamp when the dataset provides it."""
    if raw_info is None or raw_info.get("timestamp") is None:
        return None
    return float(raw_info["timestamp"])


def _first_dimension(value: Any) -> int | None:
    """Return the leading dimension for tensor-like preview values."""
    value = _unwrap_single_item(value)
    if value is None or not hasattr(value, "shape") or len(value.shape) == 0:
        return None
    return int(value.shape[0])


def _infer_sample_name(batch: dict[str, Any], dataset_index: int) -> str:
    """Build a readable sample name for visualization metadata."""
    for key in ("name", "sample_token", "scene_token", "img_path", "lidar_path"):
        value = _unwrap_single_item(batch.get(key))
        if isinstance(value, str) and value:
            return value

    metadata = _unwrap_single_item(batch.get("metadata"))
    if isinstance(metadata, dict):
        token = metadata.get("token")
        if isinstance(token, str) and token:
            return token
        image_path = metadata.get("image", {}).get("img_path")
        if isinstance(image_path, str) and image_path:
            return Path(image_path).name

    return f"sample-{dataset_index}"
