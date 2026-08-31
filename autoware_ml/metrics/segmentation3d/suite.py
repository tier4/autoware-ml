"""The 3D semantic segmentation confusion-matrix suite: a task state-engine.

``Segmentation3DConfusionMatrixMetricSuite`` accumulates one fixed-size confusion
matrix per (filter, range) bucket in a single stacked ``(F+1, R+1, C, C)``
``"sum"`` state, where slice ``[0, 0]`` is whole-scene, filter slices 1..F are the
configured evaluation filters (region / corridor / collision) and range slices 1..R the
radial windows, so memory never grows with the dataset and torchmetrics sums the
matrices across ranks exactly. Per-point filter membership is resolved at
``update`` from each ``seg_frames`` entry (where the ego pose lives), which is
what lets a bounded matrix support the filter axis at all. It knows nothing about
which metrics run: it hands each a ``ConfusionState``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from autoware_ml.metrics.base import IDENTITY, Metric, MetricFilter, MetricRange, MetricSuite
from autoware_ml.metrics.class_groups import fold_confusion, resolve_class_groups
from autoware_ml.metrics.segmentation3d.confusion import ConfusionState


class Segmentation3DConfusionMatrixMetricSuite(MetricSuite[ConfusionState]):
    """Confusion-matrix segmentation suite over (filter, range) buckets."""

    prefix = "seg3d"
    _required_keys = ("seg_frames",)

    def __init__(
        self,
        components: list[Metric[ConfusionState]],
        num_classes: int,
        ignore_index: int = -1,
        class_names: tuple[str, ...] | None = None,
        ranges: tuple[MetricRange, ...] = (),
        class_groups: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register the stacked confusion state for every (filter, range) bucket.

        Args:
            components: Injected confusion-reading metrics, as in :class:`MetricSuite`.
            num_classes: Number of trained classes, the matrix is ``C x C``.
            ignore_index: Target label excluded from the accumulation.
            class_names: Ordered class names for metric keys, or ``None``.
            ranges: Radial windows every key is also emitted for.
            class_groups: Group name to member class names. When set, the accumulated matrix
                is folded onto the behaviour taxonomy at compute.
            **kwargs: Forwarded to :class:`MetricSuite`.
        """
        super().__init__(components=components, ranges=ranges, **kwargs)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = tuple(class_names) if class_names is not None else None
        # When class_groups is set the suite folds the accumulated matrix onto the
        # behaviour taxonomy at compute. The synced state is always the trained
        # classes, so a per-class and a grouped suite share the same accumulation.
        self._group_lut, self._grouped_names = (
            resolve_class_groups(self.class_names, class_groups) if class_groups else (None, None)
        )

        # Distinct evaluation filters used by the components. Bucket 0 is the
        # unfiltered (whole-scene) slice, filter buckets follow in registration
        # order.
        self._register_component_filters()

        # One matrix per (filter, range) bucket. Accumulation always uses the
        # trained classes and the behaviour-group fold is a compute-time view, so
        # the synced state is identical with or without grouping.
        self.add_state(
            "confusion",
            default=torch.zeros(
                len(self._component_filters) + 1,
                len(self.ranges) + 1,
                num_classes,
                num_classes,
                dtype=torch.long,
            ),
            dist_reduce_fx="sum",
        )

    def required_keys(self) -> tuple[str, ...]:
        """``seg_frames`` plus any component-declared eval-output keys.

        Filter context keys (ego pose, scene token) live *inside* each frame
        entry, not at the eval-output top level, so they are validated per frame in
        ``update`` instead. Component-level keys stay in the batch-0 check.

        Returns:
            The required eval-output key names.
        """
        extra = tuple(
            key for component in self.active_components() for key in component.required_eval_keys
        )
        return tuple(dict.fromkeys(type(self)._required_keys + extra))

    def _counts(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        indices = targets * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        return counts.reshape(self.num_classes, self.num_classes).long()

    def update(self, eval_out: dict[str, Any]) -> None:
        """Fold one batch of per-frame point predictions into the matrices.

        Args:
            eval_out: Flat eval-output dict with the per-frame ``seg_frames``.
        """
        active = self._stage_filter_keys()
        for frame in eval_out["seg_frames"]:
            predictions = frame["pred"].detach().reshape(-1)
            targets = frame["target"].detach().reshape(-1)
            coord = frame["coord"].detach().to(dtype=torch.float32)
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

            self._accumulate(0, predictions, targets, coord)
            if not active:
                # No region filter is active at this stage (they are test-only):
                # skip the device-to-host copy entirely on validation epochs.
                continue
            coord_np = coord[:, :3].cpu().numpy()  # xyz only, filters convert dtype themselves
            for index, metric_filter in enumerate(self._component_filters):
                if metric_filter.cache_key not in active:
                    continue
                missing = [key for key in metric_filter.required_eval_keys if key not in frame]
                if missing:
                    raise ValueError(
                        f"Filter {metric_filter.name!r} needs {missing} inside each "
                        "seg_frames entry, the model's build_eval_output must add them."
                    )
                context = {key: frame[key] for key in metric_filter.required_eval_keys}
                is_available = metric_filter.available(context)
                self._note_frame_coverage(index, is_available)
                if not is_available:
                    continue  # scene without a lanelet map, excluded from this slice
                keep = torch.from_numpy(
                    np.asarray(metric_filter.keep(coord_np, context), dtype=bool)
                ).to(predictions.device)
                self._accumulate(index + 1, predictions[keep], targets[keep], coord[keep])

    def _accumulate(
        self,
        filter_bucket: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        coord: torch.Tensor,
    ) -> None:
        device = self.confusion.device
        self.confusion[filter_bucket, 0] += self._counts(predictions, targets).to(device)
        if not self.ranges:
            return
        distance = torch.linalg.vector_norm(coord[:, :2], dim=1)
        for index, metric_range in enumerate(self.ranges):
            in_range = distance >= metric_range.min_distance
            if metric_range.max_distance is not None:
                in_range &= distance < metric_range.max_distance
            self.confusion[filter_bucket, index + 1] += self._counts(
                predictions[in_range], targets[in_range]
            ).to(device)

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter = IDENTITY
    ) -> ConfusionState:
        """Build the confusion state for the requested (filter, range) bucket.

        Args:
            metric_range: Radial window in meters, or ``None`` for no clipping.
            metric_filter: Spatial filter selecting the bucket's points.

        Returns:
            The confusion state for the bucket.
        """
        index = self._filter_index(metric_filter)
        filter_bucket = 0 if index is None else index + 1
        range_bucket = 0 if metric_range is None else self.ranges.index(metric_range) + 1
        confusion = self.confusion[filter_bucket, range_bucket]
        if self._group_lut is not None:
            num_grouped = len(self._grouped_names)
            return ConfusionState(
                confusion=fold_confusion(confusion, self._group_lut, num_grouped),
                class_names=self._grouped_names,
                num_classes=num_grouped,
            )
        return ConfusionState(
            confusion=confusion,
            class_names=self.class_names,
            num_classes=self.num_classes,
        )
