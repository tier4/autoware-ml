"""Center-distance matching math shared by the detection metrics.

Pure helpers only. The metrics orchestrate these: they ask a ``DetectionState``
for match curves (which it memoizes) and turn them into AP, APH, NDS, or TP
errors. Matching is nuScenes-style BEV center distance with score-ordered
precision/recall curves.
"""

from __future__ import annotations

from math import pi

import numpy as np
import torch

from autoware_ml.metrics.base import MetricRange
from autoware_ml.metrics.detection3d.structures import (
    CurveMetrics,
    Detection3DSample,
    ERROR_NAMES,
    MatchCurve,
    PredictionRecord,
    SelectedTpErrors,
)


def to_cpu_tensor(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Move a tensor to CPU and optionally cast its dtype."""
    output = tensor.detach().to(device="cpu")
    return output.to(dtype=dtype) if dtype is not None else output


def _distance_mask(
    boxes: torch.Tensor,
    min_distance: float,
    max_distance: float | None,
) -> torch.Tensor:
    """Boolean mask selecting boxes within a radial distance window (XY norm)."""
    if boxes.numel() == 0:
        return torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)

    distances = torch.linalg.vector_norm(boxes[:, :2].to(dtype=torch.float32), dim=1)
    mask = distances >= min_distance
    if max_distance is not None:
        mask &= distances < max_distance
    return mask


def normalize_filter_attributes(
    filter_attributes: list[tuple[str, str]] | None,
) -> set[tuple[str, str]]:
    """Normalize configured class-attribute filters from Hydra/Python containers."""
    if not filter_attributes:
        return set()
    return {(str(class_name), str(attribute)) for class_name, attribute in filter_attributes}


def gt_keep_mask(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_num_points: torch.Tensor | None,
    gt_attributes: list[list[str]] | None,
    class_names: tuple[str, ...],
    eval_class_range: dict[str, float] | None,
    min_point_numbers: int,
    filter_set: set[tuple[str, str]],
) -> torch.Tensor:
    """Build the per-frame GT keep mask applied at accumulation time.

    Combines per-class distance caps, the minimum LiDAR point count, and
    attribute exclusions into one boolean mask, so the suite stores only kept GT
    and never gathers point counts or string attributes across ranks.
    """
    n = gt_boxes.shape[0]
    keep = torch.ones(n, dtype=torch.bool, device=gt_boxes.device)
    if n == 0:
        return keep

    if eval_class_range:
        distances = torch.linalg.vector_norm(gt_boxes[:, :2].to(dtype=torch.float32), dim=1)
        for box_idx in range(n):
            label = int(gt_labels[box_idx].item())
            if 0 <= label < len(class_names):
                max_dist = eval_class_range.get(class_names[label])
                if max_dist is not None and float(distances[box_idx].item()) > max_dist:
                    keep[box_idx] = False

    if min_point_numbers > 0 and gt_num_points is not None:
        keep &= gt_num_points >= min_point_numbers

    if filter_set and gt_attributes is not None:
        for box_idx in range(n):
            if not keep[box_idx]:
                continue
            label = int(gt_labels[box_idx].item())
            class_name = class_names[label] if 0 <= label < len(class_names) else None
            if class_name is not None:
                attrs = gt_attributes[box_idx]
                if any((str(class_name), str(attr)) in filter_set for attr in attrs):
                    keep[box_idx] = False

    return keep


def _slice_sample(
    sample: Detection3DSample,
    gt_keep: torch.Tensor,
    pred_keep: torch.Tensor,
) -> Detection3DSample:
    """Return a new sample with GT and predictions sliced by boolean masks."""
    return Detection3DSample(
        pred_boxes=sample.pred_boxes[pred_keep],
        pred_scores=sample.pred_scores[pred_keep],
        pred_labels=sample.pred_labels[pred_keep],
        gt_boxes=sample.gt_boxes[gt_keep],
        gt_labels=sample.gt_labels[gt_keep],
    )


def clip_to_range(
    samples: list[Detection3DSample],
    metric_range: MetricRange,
) -> list[Detection3DSample]:
    """Clip both GT and predictions to a radial distance window."""
    return [
        _slice_sample(
            sample,
            gt_keep=_distance_mask(
                sample.gt_boxes, metric_range.min_distance, metric_range.max_distance
            ),
            pred_keep=_distance_mask(
                sample.pred_boxes, metric_range.min_distance, metric_range.max_distance
            ),
        )
        for sample in samples
    ]


def labels_to_evaluate(
    samples: list[Detection3DSample],
    class_names: tuple[str, ...] | None,
) -> list[int]:
    """All class indices when ``class_names`` is given, else only present labels."""
    if class_names is not None:
        return list(range(len(class_names)))
    labels = {
        int(label.item())
        for sample in samples
        for label in sample.gt_labels.reshape(-1)
        if int(label.item()) >= 0
    }
    return sorted(labels)


def match_center_distance(
    samples: list[Detection3DSample],
    label: int,
    threshold: float,
) -> MatchCurve:
    """Greedy score-ordered nuScenes-style matching for one class and threshold."""
    gt_boxes_by_sample: list[np.ndarray] = []
    matched_by_sample: list[np.ndarray] = []
    predictions: list[PredictionRecord] = []
    total_gt = 0

    for sample_index, sample in enumerate(samples):
        _validate_box_tensor(sample.gt_boxes, "gt_boxes")
        _validate_box_tensor(sample.pred_boxes, "pred_boxes")

        gt_mask = sample.gt_labels == label
        sample_gt_boxes = sample.gt_boxes[gt_mask].numpy()
        gt_boxes_by_sample.append(sample_gt_boxes)
        matched_by_sample.append(np.zeros(sample_gt_boxes.shape[0], dtype=bool))
        total_gt += int(sample_gt_boxes.shape[0])

        pred_mask = sample.pred_labels == label
        sample_pred_boxes = sample.pred_boxes[pred_mask].numpy()
        sample_pred_scores = sample.pred_scores[pred_mask].numpy()
        predictions.extend(
            PredictionRecord(float(score), sample_index, box)
            for score, box in zip(sample_pred_scores, sample_pred_boxes, strict=True)
        )

    predictions.sort(key=lambda item: item.score, reverse=True)

    count = len(predictions)
    scores = np.asarray([item.score for item in predictions], dtype=np.float64)
    true_positive = np.zeros(count, dtype=np.float64)
    false_positive = np.zeros(count, dtype=np.float64)
    heading_score = np.zeros(count, dtype=np.float64)
    error_values = {name: np.full(count, np.nan, dtype=np.float64) for name in ERROR_NAMES}

    for pred_index, prediction in enumerate(predictions):
        sample_gt_boxes = gt_boxes_by_sample[prediction.sample_index]
        if sample_gt_boxes.shape[0] == 0:
            false_positive[pred_index] = 1.0
            continue

        distances = np.linalg.norm(sample_gt_boxes[:, :2] - prediction.box[:2], axis=1)
        distances[matched_by_sample[prediction.sample_index]] = np.inf
        gt_index = int(np.argmin(distances))
        if float(distances[gt_index]) <= threshold:
            gt_box = sample_gt_boxes[gt_index]
            true_positive[pred_index] = 1.0
            matched_by_sample[prediction.sample_index][gt_index] = True
            error_values["ATE"][pred_index] = _translation_error_bev(prediction.box, gt_box)
            error_values["AOE"][pred_index] = _orientation_error(prediction.box, gt_box)
            error_values["ASE"][pred_index] = _scale_error(prediction.box, gt_box)
            error_values["AVE"][pred_index] = _velocity_error(prediction.box, gt_box)
            error_values["AAE"][pred_index] = 1.0
            heading_score[pred_index] = _heading_score(error_values["AOE"][pred_index])
        else:
            false_positive[pred_index] = 1.0

    return MatchCurve(
        total_gt=total_gt,
        scores=scores,
        true_positive=true_positive,
        false_positive=false_positive,
        heading_score=heading_score,
        translation_error=error_values["ATE"],
        orientation_error=error_values["AOE"],
        scale_error=error_values["ASE"],
        velocity_error=error_values["AVE"],
        attribute_error=error_values["AAE"],
    )


def _validate_box_tensor(boxes: torch.Tensor, name: str) -> None:
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"{name} must have shape (N, 7+) but got {tuple(boxes.shape)}.")


def curve_metrics(curve: MatchCurve) -> CurveMetrics:
    """AP, APH, max-F1 and the optimal-confidence operating point for a curve."""
    precision, recall = _precision_recall(curve.cumulative_tp, curve.cumulative_fp, curve.total_gt)
    ap = _interpolated_ap(precision, recall, curve.total_gt, curve.num_predictions)

    heading_precision, heading_recall = _precision_recall(
        curve.cumulative_heading_tp, curve.cumulative_fp, curve.total_gt
    )
    f1_scores = _f1_scores(precision, recall)
    optimal_index = _max_f1_index(f1_scores)

    if optimal_index >= 0:
        max_f1 = float(f1_scores[optimal_index])
        optimal_conf = float(curve.scores[optimal_index])
        optimal_recall = float(recall[optimal_index])
        optimal_precision = float(precision[optimal_index])
    else:
        max_f1 = optimal_conf = optimal_recall = optimal_precision = np.nan

    return CurveMetrics(
        ap=ap,
        aph=_interpolated_ap(
            heading_precision, heading_recall, curve.total_gt, curve.num_predictions
        ),
        max_f1=max_f1,
        optimal_conf=optimal_conf,
        optimal_index=optimal_index,
        optimal_recall=optimal_recall,
        optimal_precision=optimal_precision,
    )


def _precision_recall(
    cumulative_tp: np.ndarray,
    cumulative_fp: np.ndarray,
    total_gt: int,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = cumulative_tp + cumulative_fp
    precision = np.divide(
        cumulative_tp,
        denominator,
        out=np.zeros_like(cumulative_tp, dtype=np.float64),
        where=denominator != 0.0,
    )
    recall = cumulative_tp / float(total_gt) if total_gt > 0 else np.zeros_like(cumulative_tp)
    return precision, recall


def _interpolated_ap(
    precision: np.ndarray,
    recall: np.ndarray,
    total_gt: int,
    num_predictions: int,
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> float:
    if total_gt == 0 and num_predictions == 0:
        return np.nan
    if precision.shape[0] == 0:
        return 0.0

    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_grid = np.linspace(0.0, 1.0, 101)
    precision_interp = np.interp(recall_grid, recall, precision_envelope, right=0.0)
    first_index = int(round(100 * min_recall)) + 1
    filtered_precision = precision_interp[first_index:] - min_precision
    filtered_precision[filtered_precision < 0.0] = 0.0
    return float(np.mean(filtered_precision)) / (1.0 - min_precision)


def _f1_scores(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    denominator = precision + recall
    return np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float64),
        where=denominator != 0.0,
    )


def _max_f1_index(f1_scores: np.ndarray) -> int:
    if f1_scores.shape[0] == 0 or np.all(np.isnan(f1_scores)):
        return -1
    return int(np.nanargmax(f1_scores))


def select_recall_tp_errors(curve: MatchCurve, recall_target: float) -> SelectedTpErrors:
    """Mean TP errors over the matches up to a recall target."""
    effective_recall = (int(round(100 * recall_target)) + 1) / 100.0
    target_matches = int(np.floor(curve.total_gt * effective_recall))
    tp_indices = np.flatnonzero(curve.true_positive == 1.0)[:target_matches]
    return _selected_error_values(curve, tp_indices)


def select_optimal_tp_errors(curve: MatchCurve, optimal_index: int) -> SelectedTpErrors:
    """Mean TP errors over the matches up to the optimal-F1 operating point."""
    if optimal_index < 0:
        return _selected_error_values(curve, np.asarray([], dtype=np.int64))
    prefix_true_positive = curve.true_positive[: optimal_index + 1]
    tp_indices = np.flatnonzero(prefix_true_positive == 1.0)
    return _selected_error_values(curve, tp_indices)


def _selected_error_values(curve: MatchCurve, tp_indices: np.ndarray) -> SelectedTpErrors:
    errors = {
        "ATE": _mean_or_one(curve.translation_error[tp_indices]),
        "AOE": _mean_or_one(curve.orientation_error[tp_indices]),
        "ASE": _mean_or_one(curve.scale_error[tp_indices]),
        "AVE": _mean_or_one(curve.velocity_error[tp_indices]),
        "AAE": _mean_or_one(curve.attribute_error[tp_indices]),
    }
    return SelectedTpErrors(count=int(tp_indices.shape[0]), errors=errors)


def mean_tp_errors(error_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each error name across the given per-class/threshold error dicts."""
    return {
        error_name: _mean_valid([errors[error_name] for errors in error_dicts])
        for error_name in ERROR_NAMES
    }


def nds(mean_ap: float, errors: dict[str, float]) -> float:
    """nuScenes detection score from mean AP and the mean TP errors."""
    error_score = sum(max(0.0, 1.0 - errors[name]) for name in ERROR_NAMES)
    return (5.0 * mean_ap + error_score) / 10.0


def _translation_error_bev(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    return float(np.linalg.norm(pred_box[:2] - gt_box[:2]))


def _orientation_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    diff = abs(float(pred_box[6] - gt_box[6]))
    diff = (diff + pi) % (2.0 * pi) - pi
    return abs(diff)


def _heading_score(orientation_error: float) -> float:
    return round(max(0.0, min(1.0, 1.0 - orientation_error / pi)), 10)


def _scale_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    pred_dims = np.maximum(pred_box[3:6], 0.0)
    gt_dims = np.maximum(gt_box[3:6], 0.0)
    intersection = float(np.prod(np.minimum(pred_dims, gt_dims)))
    pred_volume = float(np.prod(pred_dims))
    gt_volume = float(np.prod(gt_dims))
    union = pred_volume + gt_volume - intersection
    if union <= 0.0:
        return 1.0
    return 1.0 - intersection / union


def _velocity_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    if pred_box.shape[0] < 9 or gt_box.shape[0] < 9:
        return 1.0
    return float(np.linalg.norm(pred_box[7:9] - gt_box[7:9]))


def _mean_or_one(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.shape[0] == 0:
        return 1.0
    return float(np.mean(valid))


def _mean_valid(values: list[float] | tuple[float, ...]) -> float:
    valid_values = [float(value) for value in values if not np.isnan(float(value))]
    if not valid_values:
        return np.nan
    return float(sum(valid_values) / len(valid_values))


# Public alias: metrics compute class-means and threshold-means with this.
mean_valid = _mean_valid
