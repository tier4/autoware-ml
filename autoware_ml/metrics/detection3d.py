"""Dataset-level metrics for 3D object detection.

This module implements center-distance mean average precision (mAP) for lidar-based
3D detectors. Evaluation supports per-class distance caps, minimum point filters,
attribute-based GT exclusions, and arbitrary radial distance buckets for range-specific
metrics alongside an overall total metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection3DSample:
    """Prediction and ground-truth tensors for one detection frame.

    Attributes:
        pred_boxes: Predicted 3D boxes, shape ``(P, 7+)``.
        pred_scores: Predicted confidence scores, shape ``(P,)``.
        pred_labels: Predicted class indices, shape ``(P,)``.
        gt_boxes: Ground-truth 3D boxes, shape ``(G, 7+)``.
        gt_labels: Ground-truth class indices, shape ``(G,)``.
        gt_num_points: Number of LiDAR points per GT box, shape ``(G,)``.
            ``None`` when point counts are unavailable.
        gt_attributes: Per-box attribute tag lists, length ``G``.
            ``None`` when attributes are unavailable.
    """

    pred_boxes: torch.Tensor
    pred_scores: torch.Tensor
    pred_labels: torch.Tensor
    gt_boxes: torch.Tensor
    gt_labels: torch.Tensor
    gt_num_points: torch.Tensor | None = None
    gt_attributes: list[list[str]] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class DetectionRange:
    """Radial evaluation window in meters.

    Attributes:
        name: Human-readable range label used in metric keys.
        min_distance: Inclusive lower bound in meters.
        max_distance: Exclusive upper bound in meters, or ``None`` for unbounded.
    """

    name: str
    min_distance: float
    max_distance: float | None


def _to_cpu_tensor(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Move a tensor to CPU and optionally cast its dtype.

    Args:
        tensor: Source tensor on any device.
        dtype: Target dtype. ``None`` preserves the source dtype.

    Returns:
        CPU tensor with the requested dtype.
    """
    output = tensor.detach().to(device="cpu")
    return output.to(dtype=dtype) if dtype is not None else output


def _gather_samples(samples: list[Detection3DSample]) -> list[Detection3DSample]:
    """Gather detection samples from all distributed ranks.

    Args:
        samples: Per-rank list of accumulated detection samples.

    Returns:
        Flat list of samples collected from every rank. Returns the input
        unchanged when distributed training is not active.
    """
    if not dist.is_available() or not dist.is_initialized():
        return samples

    gathered: list[list[Detection3DSample]] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, samples)
    return [sample for rank_samples in gathered for sample in rank_samples]


def _metric_token(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def _range_metric_suffix(detection_range: DetectionRange) -> str:
    """Build the metric key suffix for a detection range.

    Args:
        detection_range: Range whose suffix to build.

    Returns:
        Suffix string of the form ``"0m_50m"`` or ``"50m_inf"``.
    """
    min_distance = _distance_metric_token(detection_range.min_distance)
    if detection_range.max_distance is None:
        return f"{min_distance}_inf"
    return f"{min_distance}_{_distance_metric_token(detection_range.max_distance)}"


def _distance_metric_token(distance: float) -> str:
    """Build a collision-free metric token for one distance bound."""
    token = f"{float(distance):g}".replace("-", "minus").replace(".", "p")
    return f"{token}m"


def _label_metric_name(label: int, class_names: tuple[str, ...] | None) -> str:
    """Return the metric name token for a class label.

    Args:
        label: Integer class index.
        class_names: Ordered class name tuple. ``None`` falls back to
            ``"class_{label}"``.

    Returns:
        Lowercase underscore-separated class name token.
    """
    if class_names is not None and 0 <= label < len(class_names):
        return _metric_token(class_names[label])
    return f"class_{label}"


def _supported_labels(samples: list[Detection3DSample]) -> list[int]:
    """Collect sorted non-negative GT label indices present in the samples.

    Args:
        samples: Accumulated detection samples.

    Returns:
        Sorted list of unique GT class indices.
    """
    labels = {
        int(label.item())
        for sample in samples
        for label in sample.gt_labels.reshape(-1)
        if int(label.item()) >= 0
    }
    return sorted(labels)


def _interpolated_average_precision(precision: torch.Tensor, recall: torch.Tensor) -> float:
    """Compute nuScenes-style 101-point interpolated average precision.

    Matches the AP formula used by T4MetricV2 / perception_eval:

    1. Linearly interpolate precision over 101 uniformly spaced recall points
       in ``[0, 1]`` and fill recall values to the right of the observed curve
       with zero precision.
    2. Discard the first 11 points (recall 0.0–0.10).
    3. Subtract a precision floor of 0.1 from the remaining 90 points and
       clamp negatives to zero.
    4. Divide the mean of those 90 values by 0.9 to re-normalize the scale.

    Args:
        precision: Precision values at each prediction threshold.
        recall: Recall values at each prediction threshold.

    Returns:
        Scalar average precision in ``[0, 1]``.
    """
    if precision.numel() == 0:
        return 0.0

    min_recall = 0.1
    min_precision = 0.1
    recall_grid = np.linspace(0.0, 1.0, 101)
    interpolated = torch.as_tensor(
        np.interp(
            recall_grid,
            recall.detach().cpu().numpy(),
            precision.detach().cpu().numpy(),
            right=0.0,
        ),
        dtype=precision.dtype,
    )
    first_index = int(round(100 * min_recall)) + 1
    clipped = (interpolated[first_index:] - min_precision).clamp(min=0.0)
    average_precision = float(clipped.mean().item()) / (1.0 - min_precision)
    return min(1.0, max(0.0, average_precision))


def _center_distance_ap(
    samples: list[Detection3DSample],
    label: int,
    threshold: float,
) -> float | None:
    """Compute AP for one class at one center-distance matching threshold.

    Args:
        samples: Accumulated detection samples.
        label: Class index to evaluate.
        threshold: Maximum BEV center distance in meters for a true positive.

    Returns:
        AP in ``[0, 1]``, or ``None`` when the class has no GT boxes.
    """
    gt_centers: list[torch.Tensor] = []
    matched: list[torch.Tensor] = []
    predictions: list[tuple[float, int, torch.Tensor]] = []
    total_gt = 0

    for sample_index, sample in enumerate(samples):
        gt_mask = sample.gt_labels == label
        sample_gt_centers = sample.gt_boxes[gt_mask, :2].to(dtype=torch.float32)
        gt_centers.append(sample_gt_centers)
        matched.append(torch.zeros(sample_gt_centers.shape[0], dtype=torch.bool))
        total_gt += int(sample_gt_centers.shape[0])

        pred_mask = sample.pred_labels == label
        sample_pred_boxes = sample.pred_boxes[pred_mask]
        sample_pred_scores = sample.pred_scores[pred_mask]
        for pred_box, pred_score in zip(sample_pred_boxes, sample_pred_scores, strict=True):
            predictions.append(
                (float(pred_score.item()), sample_index, pred_box[:2].to(dtype=torch.float32))
            )

    if total_gt == 0:
        return None if not predictions else 0.0
    if not predictions:
        return 0.0

    predictions.sort(key=lambda item: item[0], reverse=True)
    true_positive = torch.zeros(len(predictions), dtype=torch.float32)
    false_positive = torch.zeros(len(predictions), dtype=torch.float32)

    for pred_index, (_, sample_index, pred_center) in enumerate(predictions):
        sample_gt_centers = gt_centers[sample_index]
        if sample_gt_centers.numel() == 0:
            false_positive[pred_index] = 1.0
            continue

        distances = torch.linalg.vector_norm(
            sample_gt_centers - pred_center.unsqueeze(0),
            dim=1,
        )
        distances[matched[sample_index]] = torch.inf
        min_distance, min_index = distances.min(dim=0)
        if float(min_distance.item()) <= threshold:
            true_positive[pred_index] = 1.0
            matched[sample_index][int(min_index.item())] = True
        else:
            false_positive[pred_index] = 1.0

    cumulative_tp = torch.cumsum(true_positive, dim=0)
    cumulative_fp = torch.cumsum(false_positive, dim=0)
    recall = cumulative_tp / float(total_gt)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp_min(1.0)
    return _interpolated_average_precision(precision, recall)


def _distance_mask(
    boxes: torch.Tensor,
    min_distance: float,
    max_distance: float | None,
) -> torch.Tensor:
    """Build a boolean mask selecting boxes within a radial distance window.

    Args:
        boxes: Box tensor of shape ``(N, 7+)``. XY columns used for distance.
        min_distance: Inclusive lower bound in meters.
        max_distance: Exclusive upper bound in meters. ``None`` for unbounded.

    Returns:
        Boolean mask of shape ``(N,)``.
    """
    if boxes.numel() == 0:
        return torch.zeros((boxes.shape[0],), dtype=torch.bool)

    distances = torch.linalg.vector_norm(boxes[:, :2].to(dtype=torch.float32), dim=1)
    mask = distances >= min_distance
    if max_distance is not None:
        mask &= distances < max_distance
    return mask


def _slice_sample(
    sample: Detection3DSample,
    gt_keep: torch.Tensor,
    pred_keep: torch.Tensor | None = None,
) -> Detection3DSample:
    """Return a new sample with GT and predictions sliced by boolean masks or indices.

    Args:
        sample: Source detection sample.
        gt_keep: Boolean mask or long index tensor selecting GT boxes to keep.
        pred_keep: Boolean mask or long index tensor selecting predictions to
            keep. ``None`` leaves predictions untouched.

    Returns:
        New ``Detection3DSample`` with the selected subset of GT and predictions.
    """
    if pred_keep is None:
        pred_boxes = sample.pred_boxes
        pred_scores = sample.pred_scores
        pred_labels = sample.pred_labels
    else:
        pred_boxes = sample.pred_boxes[pred_keep]
        pred_scores = sample.pred_scores[pred_keep]
        pred_labels = sample.pred_labels[pred_keep]

    if sample.gt_attributes is not None:
        if gt_keep.dtype == torch.bool:
            gt_indices = gt_keep.nonzero(as_tuple=False).squeeze(1).tolist()
        else:
            gt_indices = gt_keep.tolist()
        gt_attributes = [sample.gt_attributes[i] for i in gt_indices]
    else:
        gt_attributes = None

    return Detection3DSample(
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_boxes=sample.gt_boxes[gt_keep],
        gt_labels=sample.gt_labels[gt_keep],
        gt_num_points=sample.gt_num_points[gt_keep] if sample.gt_num_points is not None else None,
        gt_attributes=gt_attributes,
    )


def _filter_gt(
    samples: list[Detection3DSample],
    class_names: tuple[str, ...],
    eval_class_range: dict[str, float] | None = None,
    min_point_numbers: int = 0,
    filter_attributes: list[tuple[str, str]] | None = None,
) -> list[Detection3DSample]:
    """Apply all GT-side filters in a single pass per sample.

    Combines per-class distance caps, minimum LiDAR point counts, and
    attribute-based exclusions into one composite boolean mask. Predictions
    are left untouched.

    Args:
        samples: Accumulated detection samples.
        class_names: Ordered class name tuple used for label look-ups.
        eval_class_range: Per-class maximum evaluation distance in meters.
            GT boxes beyond a class's cap are excluded. ``None`` disables
            this filter.
        min_point_numbers: Minimum number of LiDAR points required for a GT
            box to be included. Ignored when ``gt_num_points`` is ``None``.
        filter_attributes: List of ``(class_name, attribute)`` pairs whose
            matching GT boxes are excluded from evaluation. ``None`` disables
            this filter.

    Returns:
        Samples with GT boxes filtered according to all active criteria.
    """
    filter_set = _normalize_filter_attributes(filter_attributes)

    filtered: list[Detection3DSample] = []
    for sample in samples:
        n = sample.gt_boxes.shape[0]
        keep = torch.ones(n, dtype=torch.bool)

        if n > 0:
            if eval_class_range:
                distances = torch.linalg.vector_norm(
                    sample.gt_boxes[:, :2].to(dtype=torch.float32), dim=1
                )
                for box_idx in range(n):
                    label = int(sample.gt_labels[box_idx].item())
                    if 0 <= label < len(class_names):
                        max_dist = eval_class_range.get(class_names[label])
                        if max_dist is not None and float(distances[box_idx].item()) > max_dist:
                            keep[box_idx] = False

            if min_point_numbers > 0 and sample.gt_num_points is not None:
                keep &= sample.gt_num_points >= min_point_numbers

            if filter_set and sample.gt_attributes is not None:
                for box_idx in range(n):
                    if not keep[box_idx]:
                        continue
                    label = int(sample.gt_labels[box_idx].item())
                    class_name = class_names[label] if 0 <= label < len(class_names) else None
                    if class_name is not None:
                        attrs = sample.gt_attributes[box_idx]
                        if any((str(class_name), str(attr)) in filter_set for attr in attrs):
                            keep[box_idx] = False

        filtered.append(_slice_sample(sample, gt_keep=keep))
    return filtered


def _normalize_filter_attributes(
    filter_attributes: list[tuple[str, str]] | None,
) -> set[tuple[str, str]]:
    """Normalize configured class-attribute filters from Hydra/Python containers."""
    if not filter_attributes:
        return set()
    return {(str(class_name), str(attribute)) for class_name, attribute in filter_attributes}


def _clip_to_range(
    samples: list[Detection3DSample],
    detection_range: DetectionRange,
) -> list[Detection3DSample]:
    """Clip both GT and predictions to a radial distance window.

    Args:
        samples: Accumulated detection samples.
        detection_range: Distance window defining the evaluation bucket.

    Returns:
        Samples with GT and predictions outside the window removed.
    """
    return [
        _slice_sample(
            sample,
            gt_keep=_distance_mask(
                sample.gt_boxes, detection_range.min_distance, detection_range.max_distance
            ),
            pred_keep=_distance_mask(
                sample.pred_boxes, detection_range.min_distance, detection_range.max_distance
            ),
        )
        for sample in samples
    ]


def _compute_map(
    samples: list[Detection3DSample],
    thresholds: tuple[float, ...],
    class_names: tuple[str, ...] | None,
) -> dict[str, float]:
    """Compute mAP and per-class mAP over the provided samples.

    Args:
        samples: Detection samples to evaluate.
        thresholds: Center-distance matching thresholds in meters.
        class_names: Ordered class name tuple for metric key generation.
            ``None`` falls back to ``"class_{label}"`` keys.

    Returns:
        Dictionary with ``"mAP"`` and ``"mAP_{class}"`` entries for every
        class that has at least one GT box.
    """
    labels = _supported_labels(samples)
    if not labels:
        return {}

    class_aps: dict[int, list[float]] = {label: [] for label in labels}
    for threshold in thresholds:
        for label in labels:
            ap = _center_distance_ap(samples, label, threshold)
            if ap is not None:
                class_aps[label].append(ap)

    metrics: dict[str, float] = {}
    all_mean_aps: list[float] = []
    for label, aps in class_aps.items():
        if not aps:
            continue
        mean_ap = float(sum(aps) / len(aps))
        label_name = _label_metric_name(label, class_names)
        metrics[f"mAP_{label_name}"] = mean_ap
        all_mean_aps.append(mean_ap)

    metrics["mAP"] = float(sum(all_mean_aps) / len(all_mean_aps)) if all_mean_aps else 0.0
    return metrics


class CenterDistanceMeanAP:
    """Compute class-mean AP using BEV center-distance matching.

    Accumulates per-frame predictions and ground truth across batches, then
    computes mAP at epoch end. Supports distributed training via
    ``all_gather_object``.

    Evaluation proceeds in two stages:

    1. **GT filtering** - per-class distance caps, minimum point counts, and
       attribute exclusions narrow which GT boxes are evaluated at all.
    2. **Range bucketing** - both GT and predictions are clipped to each
       configured ``DetectionRange`` window. mAP is computed per bucket and
       for the full non-clipped set (total metric).

    The total metric intentionally applies ``eval_class_range`` to GT boxes
    only. Predictions are left non-clipped for the total metric, while configured
    range buckets clip both predictions and GT boxes.

    Args:
        thresholds: Center-distance matching thresholds in meters.
        class_names: Ordered class name tuple. ``None`` uses generic keys.
        ranges: Distance buckets for per-range metrics.
        eval_class_range: Per-class maximum evaluation distance in meters.
            GT boxes beyond a class cap are excluded before bucketing. When a
            class cap is smaller than a bucket's upper bound a warning is
            emitted at construction time.
        filter_attributes: ``(class_name, attribute)`` pairs whose matching
            GT boxes are excluded from evaluation.
        min_point_numbers: Minimum LiDAR point count for a GT box to be
            included. Requires ``gt_num_points`` to be provided in ``update``.
    """

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        class_names: tuple[str, ...] | None = None,
        ranges: tuple[DetectionRange, ...] = (
            DetectionRange("0-50m", 0.0, 50.0),
            DetectionRange("50-90m", 50.0, 90.0),
            DetectionRange("90-121m", 90.0, 121.0),
            DetectionRange("0-121m", 0.0, 121.0),
        ),
        eval_class_range: dict[str, float] | None = None,
        filter_attributes: list[tuple[str, str]] | None = None,
        min_point_numbers: int = 0,
    ) -> None:
        self.thresholds = tuple(float(t) for t in thresholds)
        self.class_names = class_names
        self.ranges = ranges
        self.eval_class_range = eval_class_range
        self.filter_attributes = filter_attributes
        self.min_point_numbers = int(min_point_numbers)
        self.samples: list[Detection3DSample] = []

        if (eval_class_range or filter_attributes) and not class_names:
            raise ValueError(
                "class_names must be provided when eval_class_range or filter_attributes are "
                "configured."
            )

        range_suffixes = [_range_metric_suffix(detection_range) for detection_range in ranges]
        duplicate_suffixes = sorted(
            suffix for suffix in set(range_suffixes) if range_suffixes.count(suffix) > 1
        )
        if duplicate_suffixes:
            raise ValueError(f"Detection range metric suffixes must be unique: {duplicate_suffixes}")

        if eval_class_range is not None:
            for class_name, max_dist in eval_class_range.items():
                for detection_range in ranges:
                    if (
                        detection_range.max_distance is not None
                        and max_dist < detection_range.max_distance
                    ):
                        logger.warning(
                            "eval_class_range['%s'] = %.1fm is smaller than bucket '%s' upper "
                            "bound %.1fm. GT boxes for '%s' beyond %.1fm are excluded, making "
                            "the '%s' bucket metrics misleading for this class.",
                            class_name,
                            max_dist,
                            detection_range.name,
                            detection_range.max_distance,
                            class_name,
                            max_dist,
                            detection_range.name,
                        )

    def reset(self) -> None:
        """Clear all accumulated samples."""
        self.samples.clear()

    def update(
        self,
        predictions: list[dict[str, torch.Tensor]],
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        gt_num_points: list[torch.Tensor] | None = None,
        gt_attributes: list[list[list[str]]] | None = None,
    ) -> None:
        """Accumulate predictions and ground truth for one batch.

        Args:
            predictions: Per-sample prediction dictionaries with keys
                ``"bboxes_3d"``, ``"scores_3d"``, and ``"labels_3d"``.
            gt_boxes: Per-sample GT box tensors of shape ``(G, 7+)``.
            gt_labels: Per-sample GT label tensors of shape ``(G,)``.
            gt_num_points: Per-sample LiDAR point count tensors of shape
                ``(G,)``. ``None`` when point counts are unavailable.
            gt_attributes: Per-sample per-box attribute tag lists.
                ``None`` when attributes are unavailable.

        Raises:
            ValueError: When ``predictions``, ``gt_boxes``, and ``gt_labels``
                have different lengths.
        """
        if len(predictions) != len(gt_boxes) or len(predictions) != len(gt_labels):
            raise ValueError(
                "Detection metric expects equal numbers of predictions, gt_boxes, and gt_labels."
            )

        for i, (prediction, boxes, labels) in enumerate(
            zip(predictions, gt_boxes, gt_labels, strict=True)
        ):
            sample_num_points = (
                _to_cpu_tensor(gt_num_points[i], dtype=torch.long)
                if gt_num_points is not None
                else None
            )
            sample_attributes = gt_attributes[i] if gt_attributes is not None else None
            self.samples.append(
                Detection3DSample(
                    pred_boxes=_to_cpu_tensor(prediction["bboxes_3d"], dtype=torch.float32),
                    pred_scores=_to_cpu_tensor(prediction["scores_3d"], dtype=torch.float32),
                    pred_labels=_to_cpu_tensor(prediction["labels_3d"], dtype=torch.long),
                    gt_boxes=_to_cpu_tensor(boxes, dtype=torch.float32),
                    gt_labels=_to_cpu_tensor(labels, dtype=torch.long),
                    gt_num_points=sample_num_points,
                    gt_attributes=sample_attributes,
                )
            )

    def compute(self) -> dict[str, float]:
        """Compute mAP metrics over all accumulated samples.

        Gathers samples across distributed ranks, applies GT filters, then
        computes metrics for the total unclipped set and for each configured
        distance bucket.

        Returns:
            Dictionary of scalar metrics. Keys follow the pattern
            ``"mAP"``, ``"mAP_{class}"`` for total metrics and
            ``"mAP_{range}"``, ``"mAP_{class}_{range}"`` for per-bucket
            metrics, where ``{range}`` is e.g. ``"0m_50m"``.
        """
        samples = _gather_samples(self.samples)
        samples = _filter_gt(
            samples,
            class_names=self.class_names or (),
            eval_class_range=self.eval_class_range,
            min_point_numbers=self.min_point_numbers,
            filter_attributes=self.filter_attributes,
        )

        metrics = _compute_map(samples, self.thresholds, self.class_names)

        for detection_range in self.ranges:
            range_samples = _clip_to_range(samples, detection_range)
            suffix = _range_metric_suffix(detection_range)
            metrics.update(
                {
                    f"{k}_{suffix}": v
                    for k, v in _compute_map(
                        range_samples, self.thresholds, self.class_names
                    ).items()
                }
            )

        return metrics


def metrics_to_tensors(metrics: dict[str, float], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert scalar metric values to single-element tensors.

    Args:
        metrics: Dictionary of scalar metric values.
        device: Target device for the output tensors.

    Returns:
        Dictionary with the same keys and float32 scalar tensors as values.
    """
    return {
        name: torch.tensor(value, device=device, dtype=torch.float32)
        for name, value in metrics.items()
    }
