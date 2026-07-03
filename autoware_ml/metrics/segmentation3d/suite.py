"""The 3D semantic segmentation metric suite: a task state-engine.

``Segmentation3DMetricSuite`` accumulates one fixed-size confusion matrix per
bucket in a single stacked ``(R+1, C, C)`` ``"sum"`` state (slice 0 is the
overall matrix, slices 1..R are the ranges), so memory never grows with the
dataset and torchmetrics sums the matrices across ranks exactly. It knows nothing
about which metrics run: it hands each a ``ConfusionState``.
"""

from __future__ import annotations

from typing import Any

import torch

from autoware_ml.metrics.base import Metric, MetricRange, MetricSuite
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState


class Segmentation3DMetricSuite(MetricSuite[ConfusionState]):
    """Confusion-matrix segmentation suite. Buckets points by range at update into
    one stacked confusion state, and exposes a ``ConfusionState`` per bucket to the
    injected metrics.
    """

    prefix = "seg3d"
    _required_keys = ("seg_pred_labels", "seg_target_labels", "seg_coord")

    def __init__(
        self,
        components: list[Metric[ConfusionState]],
        num_classes: int,
        ignore_index: int = -1,
        class_names: tuple[str, ...] | None = None,
        ranges: tuple[MetricRange, ...] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(components=components, ranges=ranges, **kwargs)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = tuple(class_names) if class_names is not None else None
        # One matrix per bucket: slice 0 overall, slices 1..R the configured ranges.
        self.add_state(
            "confusion",
            default=torch.zeros(len(self.ranges) + 1, num_classes, num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def _counts(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        indices = targets * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        return counts.reshape(self.num_classes, self.num_classes).long()

    def update(self, eval_out: dict[str, torch.Tensor]) -> None:
        """Fold one batch of point predictions and targets into the matrices."""
        predictions = eval_out["seg_pred_labels"].detach().reshape(-1)
        targets = eval_out["seg_target_labels"].detach().reshape(-1)
        coord = eval_out["seg_coord"].detach().to(dtype=torch.float32)
        valid = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
            & (predictions >= 0)
            & (predictions < self.num_classes)
        )
        targets = targets[valid].long()
        predictions = predictions[valid].long()
        coord = coord[valid]

        device = self.confusion.device
        self.confusion[0] += self._counts(predictions, targets).to(device)
        if not self.ranges:
            return
        distance = torch.linalg.vector_norm(coord[:, :2], dim=1)
        for index, metric_range in enumerate(self.ranges):
            in_range = distance >= metric_range.min_distance
            if metric_range.max_distance is not None:
                in_range &= distance < metric_range.max_distance
            self.confusion[index + 1] += self._counts(predictions[in_range], targets[in_range]).to(
                device
            )

    def state_for(self, metric_range: MetricRange | None) -> ConfusionState:
        """Build the confusion state for the requested metric range.

        Args:
            metric_range: Optional radial range selecting one confusion-matrix
                bucket; ``None`` selects the overall bucket.

        Returns:
            Confusion state consumed by the configured metric components.
        """
        index = 0 if metric_range is None else self.ranges.index(metric_range) + 1
        return ConfusionState(
            confusion=self.confusion[index],
            class_names=self.class_names,
            num_classes=self.num_classes,
        )
