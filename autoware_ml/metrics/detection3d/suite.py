"""The 3D detection metric suite: a task state-engine.

``Detection3DMetricSuite`` owns the per-frame prediction and ground-truth tensors
as list states (so torchmetrics handles cross-GPU sync) and applies the GT
filters at ``update`` time. It knows nothing about which metrics run: it builds
a ``DetectionState`` (overall and per range/filter) and hands it to the injected
metrics. Region and collision filters need per-frame metadata (ego pose, scene token) in the
eval output, so their keep-masks are precomputed per frame at update and nothing needs the map
at compute time, DDP gathers only tensors.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from autoware_ml.metrics.base import (
    IDENTITY,
    Metric,
    MetricFilter,
    MetricRange,
    MetricSuite,
    rank_zero,
)
from autoware_ml.metrics.class_groups import fold_labels, resolve_class_groups
from autoware_ml.metrics.detection3d.matching import (
    DetectionState,
    clip_to_range,
    gt_keep_mask,
    resolve_match_cost,
)
from autoware_ml.metrics.detection3d.naming import metric_token
from autoware_ml.metrics.detection3d.structures import Detection3DSample

logger = logging.getLogger(__name__)


def _validate_name_tokens(names: tuple[str, ...] | None) -> None:
    """Reject class-name sets whose metric-key tokens collide."""
    if not names:
        return
    tokens = [metric_token(name) for name in names]
    duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
    if duplicates:
        raise ValueError(
            f"class names collapse to duplicate metric-key tokens {duplicates}, "
            "per-class values would silently overwrite each other."
        )


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
        ranges: tuple[MetricRange, ...] = (
            MetricRange("0-50m", 0.0, 50.0),
            MetricRange("50-90m", 50.0, 90.0),
            MetricRange("90-121m", 90.0, 121.0),
            MetricRange("0-121m", 0.0, 121.0),
        ),
        eval_class_range: dict[str, float] | None = None,
        min_num_points: int = 0,
        match_cost: str = "center",
        class_groups: dict[str, list[str]] | None = None,
        collision: Any = None,
        **kwargs: Any,
    ) -> None:
        """Validate the class setup and register the accumulation states.

        Args:
            components: Injected metrics, as in :class:`MetricSuite`.
            class_names: Ordered class names for metric keys, or ``None`` to report only
                classes present in the ground truth.
            ranges: Radial windows every key is also emitted for.
            eval_class_range: Class name to maximum GT distance in meters, applied as a GT
                filter at accumulation.
            min_num_points: Minimum LiDAR points a GT box needs to count, 0 disables the
                filter.
            match_cost: Matching cost name, ``center`` or ``corner``.
            class_groups: Group name to member class names. When set, labels are folded onto
                the group taxonomy at state build, so matching happens between grouped classes.
            collision: Optional collision TTC provider. When set, the per-box
                time-to-collision to ego is computed once per frame at update and carried on
                each sample for the criticality metrics.
            **kwargs: Forwarded to :class:`MetricSuite`.
        """
        super().__init__(components=components, ranges=ranges, **kwargs)
        self.class_names = tuple(class_names) if class_names is not None else None
        _validate_name_tokens(self.class_names)
        self.eval_class_range = eval_class_range
        self.min_num_points = int(min_num_points)
        self.match_cost = str(match_cost)
        resolve_match_cost(self.match_cost)  # a config typo must fail at build, not after the epoch
        self.collision = collision
        # When class_groups is set, GT/pred labels are folded onto the behaviour
        # taxonomy at state build (after the per-class GT filters), so matching
        # happens between grouped classes: a bus predicted as a truck is a true
        # positive for grouped_vehicle. The synced boxes stay the trained classes.
        self._group_lut, self._grouped_names = (
            resolve_class_groups(self.class_names, class_groups) if class_groups else (None, None)
        )
        # The grouped names are what label_metric_name tokenizes in a grouped
        # suite, so they need the same duplicate-token guard.
        _validate_name_tokens(self._grouped_names)

        if eval_class_range and not self.class_names:
            raise ValueError("class_names must be provided when eval_class_range is configured.")
        self._warn_on_range_class_caps()

        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        self.add_state("pred_scores", default=[], dist_reduce_fx=None)
        self.add_state("pred_labels", default=[], dist_reduce_fx=None)
        self.add_state("gt_boxes", default=[], dist_reduce_fx=None)
        self.add_state("gt_labels", default=[], dist_reduce_fx=None)
        if self.collision is not None:
            self.add_state("gt_ttc", default=[], dist_reduce_fx=None)
            self.add_state("pred_ttc", default=[], dist_reduce_fx=None)
            # Per-frame flag: does this frame's scene have a lanelet map (and the
            # batch an ego pose)? Uncovered frames carry all-inf TTC and are
            # excluded from the criticality metrics' denominators, and the counters
            # log the coverage once per epoch, like the region filters do.
            self.add_state("ttc_covered", default=[], dist_reduce_fx=None)
            self.add_state(
                "ttc_frames_seen", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum"
            )
            self.add_state(
                "ttc_frames_covered",
                default=torch.zeros((), dtype=torch.long),
                dist_reduce_fx="sum",
            )

        # Distinct non-identity filters used by the components. Their keep-masks
        # are precomputed per frame at update (where the ego pose and map are
        # available) and stored as boolean tensors, so nothing needs the scene
        # token or map at compute time and DDP gathers only tensors.
        self._register_component_filters()
        if self._component_filters:
            self.add_state("gt_region_masks", default=[], dist_reduce_fx=None)
            self.add_state("pred_region_masks", default=[], dist_reduce_fx=None)
            self.add_state("region_available", default=[], dist_reduce_fx=None)

    def compute(self) -> dict[str, float]:
        """Log TTC frame coverage once per epoch, then run the metric groups.

        Returns:
            Metric keys mapped to values.
        """
        self._log_ttc_coverage()
        return super().compute()

    def _log_ttc_coverage(self) -> None:
        """Log how many frames the collision provider covered (map + ego pose).

        Uncovered frames are excluded from the criticality metrics' denominators
        (the whole-scene metrics keep them), so a partial-map dataset is never
        silently deflated.
        """
        if self.collision is None or not rank_zero():
            return
        seen = int(self.ttc_frames_seen.item())
        covered = int(self.ttc_frames_covered.item())
        if not seen:
            return
        if covered < seen:
            logger.warning(
                "Collision TTC covered %d/%d frames, %d frame(s) in scenes without a "
                "lanelet map are excluded from the criticality metrics (whole-scene "
                "metrics keep them).",
                covered,
                seen,
                seen - covered,
            )
        else:
            logger.info("Collision TTC covered %d/%d frames (full map coverage).", covered, seen)

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

    def required_keys(self) -> tuple[str, ...]:
        """Fold in the collision provider's frame-context keys.

        The collision TTC needs the ego pose and scene token exactly like the
        region filters do. Declaring them here makes a missing key fail loud at
        the first batch instead of silently reporting zero coverage.

        Returns:
            The required eval-output key names.
        """
        keys = super().required_keys()
        if self.collision is not None:
            keys = tuple(dict.fromkeys(keys + ("ego2global", "scene_token")))
        if self.min_num_points > 0:
            keys = tuple(dict.fromkeys(keys + ("gt_num_points",)))
        return keys

    def update(self, eval_out: dict[str, Any]) -> None:
        """Accumulate one batch, applying every GT filter per frame.

        Args:
            eval_out: Flat eval-output dict built by the model for one batch.
        """
        predictions = eval_out["predictions"]
        gt_boxes = eval_out["gt_boxes"]
        gt_labels = eval_out["gt_labels"]
        gt_num_points = eval_out.get("gt_num_points")
        if len(predictions) != len(gt_boxes) or len(predictions) != len(gt_labels):
            raise ValueError(
                "Detection metric expects equal numbers of predictions, gt_boxes, and gt_labels."
            )
        if self.min_num_points > 0 and gt_num_points is None:
            raise ValueError(
                f"min_num_points={self.min_num_points} is configured but the eval output "
                "carries no gt_num_points, the GT filter cannot be applied."
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

            pred_boxes = prediction["bboxes_3d"].detach().to(dtype=torch.float32)
            self.pred_boxes.append(pred_boxes)
            self.pred_scores.append(prediction["scores_3d"].detach().to(dtype=torch.float32))
            self.pred_labels.append(prediction["labels_3d"].detach().to(dtype=torch.long))
            self.gt_boxes.append(frame_boxes[keep])
            self.gt_labels.append(frame_labels[keep])

            if self.collision is not None:
                covered = self._ttc_frame_covered(eval_out, i)
                self.ttc_covered.append(torch.tensor([covered], device=frame_boxes.device))
                self.ttc_frames_seen += 1
                self.ttc_frames_covered += int(covered)
                self.gt_ttc.append(
                    self._frame_ttc(frame_boxes[keep], frame_labels[keep], eval_out, i, covered)
                )
                self.pred_ttc.append(
                    self._frame_ttc(
                        pred_boxes,
                        prediction["labels_3d"].detach().to(dtype=torch.long),
                        eval_out,
                        i,
                        covered,
                    )
                )

            if self._component_filters and self._stage_filter_keys():
                available = self._frame_region_availability(eval_out, i)
                self.region_available.append(
                    torch.tensor(available, dtype=torch.bool, device=frame_boxes.device)
                )
                self.gt_region_masks.append(
                    self._region_columns(frame_boxes[keep], eval_out, i, available)
                )
                self.pred_region_masks.append(
                    self._region_columns(pred_boxes, eval_out, i, available)
                )

    def bind_stage(self, stage: Any) -> None:
        """Drop the collision provider on stages where no metric reads TTC.

        The TTC engine is the expensive part of update, and a validation clone whose
        criticality metrics are test-only must not pay for it every epoch.

        Args:
            stage: Evaluation stage this instance accumulates and reports for.
        """
        super().bind_stage(stage)
        if self.collision is not None and not any(
            component.needs_ttc for component in self.active_components()
        ):
            self.collision = None

    def _ttc_frame_covered(self, eval_out: dict[str, Any], frame_index: int) -> bool:
        """Whether this frame's scene has a lanelet map for TTC.

        The ego pose and scene token themselves are guaranteed by
        :meth:`required_keys` at the first batch.
        """
        return self.collision.available(eval_out["scene_token"][frame_index])

    def _frame_ttc(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        eval_out: dict[str, Any],
        frame_index: int,
        covered: bool,
    ) -> torch.Tensor:
        """Per-box collision TTC for one frame, all-``inf`` when uncovered.

        An uncovered frame (map-less scene) is flagged via ``ttc_covered`` so the
        criticality metrics exclude it from their denominators. The result lives
        on the boxes' device: every list state must be device-consistent for the
        DDP gather.
        """
        if not covered:
            return torch.full((int(boxes.shape[0]),), float("inf"), device=boxes.device)
        ttc = self.collision.per_box_ttc(
            boxes.detach().cpu().numpy().astype(np.float64),
            labels.detach().cpu().numpy().astype(np.int64),
            eval_out["ego2global"][frame_index],
            eval_out["scene_token"][frame_index],
        )
        return torch.from_numpy(np.asarray(ttc, dtype=np.float32)).to(boxes.device)

    def _frame_region_availability(self, eval_out: dict[str, Any], frame_index: int) -> list[bool]:
        """Per region filter: active at this stage *and* its map present this frame.

        Tallies the per-filter coverage counters (a scene missing its lanelet map
        is excluded from that filter's slice) and returns whether each filter
        should contribute this frame. Inactive filters are ``False`` throughout.
        """
        active = self._stage_filter_keys()
        available = []
        for index, metric_filter in enumerate(self._component_filters):
            is_available = metric_filter.cache_key in active and metric_filter.available(
                {key: eval_out[key][frame_index] for key in metric_filter.required_eval_keys}
            )
            if metric_filter.cache_key in active:
                self._note_frame_coverage(index, is_available)
            available.append(is_available)
        return available

    def _region_columns(
        self,
        boxes: torch.Tensor,
        eval_out: dict[str, Any],
        frame_index: int,
        available: list[bool],
    ) -> torch.Tensor:
        """Per-box keep-mask for each region filter, as an ``(N, F)`` bool tensor.

        Only filters that are active at this stage *and* whose map is available for
        this frame are evaluated. The other columns stay ``False`` and are never
        read (their groups are skipped at compute, or the scene has no map), so a
        validation clone never touches test-only context keys and a map-less scene
        contributes nothing to the slice. The full box rows are passed (not just
        centers) so region filters can test footprint overlap.
        """
        elements = boxes.detach().cpu().numpy().astype(np.float64)
        columns = []
        for index, metric_filter in enumerate(self._component_filters):
            if available[index]:
                context = {
                    key: eval_out[key][frame_index] for key in metric_filter.required_eval_keys
                }
                columns.append(metric_filter.keep(elements, context))
            else:
                columns.append(np.zeros(elements.shape[0], dtype=bool))
        stacked = (
            np.stack(columns, axis=1)
            if elements.shape[0]
            else np.zeros((0, len(self._component_filters)), dtype=bool)
        )
        # On the boxes' device: list states must be device-consistent for DDP.
        return torch.from_numpy(np.asarray(stacked, dtype=bool)).to(boxes.device)

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter = IDENTITY
    ) -> DetectionState:
        """Build the detection state for the requested range and region filter.

        Args:
            metric_range: Optional radial range used to clip predictions and
                ground truth before metric evaluation.
            metric_filter: Optional region filter, boxes are pre-selected by the
                keep-mask computed for it at update time. Frames where the
                filter was unavailable (e.g. a scene without a lanelet map) are
                omitted from the slice entirely, so per-frame denominators only
                count covered frames.

        Returns:
            Detection state consumed by the configured metric components.
        """
        column = self._filter_index(metric_filter)

        samples = []
        for index in range(len(self.gt_boxes)):
            if column is not None and not bool(self.region_available[index][column]):
                continue
            pred_boxes = self.pred_boxes[index].cpu()
            pred_scores = self.pred_scores[index].cpu()
            pred_labels = self.pred_labels[index].cpu()
            gt_boxes = self.gt_boxes[index].cpu()
            gt_labels = self.gt_labels[index].cpu()
            gt_ttc = self.gt_ttc[index].cpu() if self.collision is not None else None
            pred_ttc = self.pred_ttc[index].cpu() if self.collision is not None else None
            ttc_covered = (
                bool(self.ttc_covered[index].item()) if self.collision is not None else True
            )
            if column is not None:
                gt_mask = self.gt_region_masks[index][:, column].cpu()
                pred_mask = self.pred_region_masks[index][:, column].cpu()
                gt_boxes, gt_labels = gt_boxes[gt_mask], gt_labels[gt_mask]
                pred_boxes = pred_boxes[pred_mask]
                pred_scores = pred_scores[pred_mask]
                pred_labels = pred_labels[pred_mask]
                if gt_ttc is not None:
                    gt_ttc = gt_ttc[gt_mask]
                if pred_ttc is not None:
                    pred_ttc = pred_ttc[pred_mask]
            samples.append(
                Detection3DSample(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=self._fold(pred_labels),
                    gt_boxes=gt_boxes,
                    gt_labels=self._fold(gt_labels),
                    gt_ttc=gt_ttc,
                    pred_ttc=pred_ttc,
                    ttc_covered=ttc_covered,
                )
            )
        if metric_range is not None:
            samples = clip_to_range(samples, metric_range)
        return DetectionState(
            samples=samples,
            class_names=self._grouped_names if self._group_lut is not None else self.class_names,
            match_cost=self.match_cost,
        )

    def _fold(self, labels: torch.Tensor) -> torch.Tensor:
        """Fold trained-class labels onto the behaviour taxonomy (identity if unset)."""
        if self._group_lut is None:
            return labels
        folded = fold_labels(labels.numpy(), self._group_lut)
        return torch.from_numpy(folded).to(dtype=torch.long)
