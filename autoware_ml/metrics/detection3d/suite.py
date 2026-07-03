"""The 3D detection metric suite: a task state-engine.

``Detection3DMetricSuite`` owns the per-frame prediction and ground-truth tensors
as list states (so torchmetrics handles cross-GPU sync) and applies the GT
filters at ``update`` time. It knows nothing about which metrics run: it builds
a ``DetectionState`` (overall and per range) and hands it to the injected metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from autoware_ml.metrics.base import Metric, MetricRange, MetricSuite
from autoware_ml.metrics.detection3d.matching import (
    clip_to_range,
    gt_keep_mask,
)
from autoware_ml.metrics.detection3d.structures import Detection3DSample, DetectionState

logger = logging.getLogger(__name__)


class Detection3DMetricSuite(MetricSuite[DetectionState]):
    """Center-distance 3D detection suite. Accumulates per-frame samples, applies
    GT filters at update, and exposes a ``DetectionState`` (clipped per range) to
    the injected metrics.
    """

    prefix = "det3d"
    _required_keys = ("predictions", "gt_boxes", "gt_labels")

    def __init__(
        self,
        components: list[Metric[DetectionState]],
        class_names: tuple[str, ...] | None = None,
        thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        ranges: tuple[MetricRange, ...] = (
            MetricRange("0-50m", 0.0, 50.0),
            MetricRange("50-90m", 50.0, 90.0),
            MetricRange("90-121m", 90.0, 121.0),
            MetricRange("0-121m", 0.0, 121.0),
        ),
        eval_class_range: dict[str, float] | None = None,
        min_num_points: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(components=components, ranges=ranges, **kwargs)
        self.class_names = tuple(class_names) if class_names is not None else None
        self.thresholds = tuple(float(threshold) for threshold in thresholds)
        self.eval_class_range = eval_class_range
        self.min_num_points = int(min_num_points)

        if eval_class_range and not self.class_names:
            raise ValueError("class_names must be provided when eval_class_range is configured.")
        self._warn_on_range_class_caps()

        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        self.add_state("pred_scores", default=[], dist_reduce_fx=None)
        self.add_state("pred_labels", default=[], dist_reduce_fx=None)
        self.add_state("gt_boxes", default=[], dist_reduce_fx=None)
        self.add_state("gt_labels", default=[], dist_reduce_fx=None)

    def _warn_on_range_class_caps(self) -> None:
        if self.eval_class_range is None:
            return
        for class_name, max_dist in self.eval_class_range.items():
            for metric_range in self.ranges:
                if metric_range.max_distance is not None and max_dist < metric_range.max_distance:
                    logger.warning(
                        "eval_class_range['%s'] = %.1fm is smaller than bucket '%s' upper "
                        "bound %.1fm, so the '%s' bucket metrics are misleading for this class.",
                        class_name,
                        max_dist,
                        metric_range.name,
                        metric_range.max_distance,
                        metric_range.name,
                    )

    def update(self, eval_out: dict[str, Any]) -> None:
        """Accumulate one batch, applying every GT filter per frame."""
        predictions = eval_out["predictions"]
        gt_boxes = eval_out["gt_boxes"]
        gt_labels = eval_out["gt_labels"]
        gt_num_points = eval_out.get("gt_num_points")
        if len(predictions) != len(gt_boxes) or len(predictions) != len(gt_labels):
            raise ValueError(
                "Detection metric expects equal numbers of predictions, gt_boxes, and gt_labels."
            )

        for i, (prediction, boxes, labels) in enumerate(
            zip(predictions, gt_boxes, gt_labels, strict=True)
        ):
            frame_boxes = boxes.detach().to(dtype=torch.float32)
            frame_labels = labels.detach().to(dtype=torch.long)
            num_points = (
                gt_num_points[i].detach().to(dtype=torch.long, device=frame_boxes.device)
                if gt_num_points is not None
                else None
            )
            keep = gt_keep_mask(
                frame_boxes,
                frame_labels,
                num_points,
                self.class_names or (),
                self.eval_class_range,
                self.min_num_points,
            )

            self.pred_boxes.append(prediction["bboxes_3d"].detach().to(dtype=torch.float32))
            self.pred_scores.append(prediction["scores_3d"].detach().to(dtype=torch.float32))
            self.pred_labels.append(prediction["labels_3d"].detach().to(dtype=torch.long))
            self.gt_boxes.append(frame_boxes[keep])
            self.gt_labels.append(frame_labels[keep])

    def state_for(self, metric_range: MetricRange | None) -> DetectionState:
        """Build the detection state for the requested metric range.

        Args:
            metric_range: Optional radial range used to clip predictions and
                ground truth before metric evaluation.

        Returns:
            Detection state consumed by the configured metric components.
        """
        samples = [
            Detection3DSample(
                pred_boxes=pred_boxes.cpu(),
                pred_scores=pred_scores.cpu(),
                pred_labels=pred_labels.cpu(),
                gt_boxes=gt_boxes.cpu(),
                gt_labels=gt_labels.cpu(),
            )
            for pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels in zip(
                self.pred_boxes,
                self.pred_scores,
                self.pred_labels,
                self.gt_boxes,
                self.gt_labels,
                strict=True,
            )
        ]
        if metric_range is not None:
            samples = clip_to_range(samples, metric_range)
        return DetectionState(
            samples=samples, class_names=self.class_names, thresholds=self.thresholds
        )
