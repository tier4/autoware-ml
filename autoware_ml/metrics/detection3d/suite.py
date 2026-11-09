"""The 3D detection metric suite: a task state-engine.

``Detection3DMetricSuite`` owns the per-frame prediction and ground-truth tensors
as list states (so torchmetrics handles cross-GPU sync) and applies the GT
filters at ``update`` time. It knows nothing about which metrics run: it builds
a ``Detection3DState`` (overall and per range/filter) and hands it to the injected
metrics. Region and collision filters need the per-frame metadata (ego pose, lanelet map) the
eval frames carry, so their keep-masks are precomputed per frame at update and nothing needs
the map at compute time, DDP gathers only tensors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
    Detection3DState,
    clip_to_range,
    gt_keep_mask,
)
from autoware_ml.metrics.detection3d.naming import metric_token
from autoware_ml.metrics.detection3d.structures import Detection3DSample
from autoware_ml.metrics.eval_output import EvalOutput, MetricRequirements
from autoware_ml.metrics.report import MetricReport
from autoware_ml.types.metrics import MatchCost

if TYPE_CHECKING:
    from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample

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


class Detection3DMetricSuite(MetricSuite[Detection3DState]):
    """Center-distance 3D detection suite. Accumulates per-frame samples, applies
    GT filters at update, and exposes a ``Detection3DState`` (clipped per range) to
    the injected metrics.
    """

    prefix = "det3d"

    def __init__(
        self,
        components: list[Metric[Detection3DState]],
        class_names: tuple[str, ...] | None = None,
        ranges: tuple[MetricRange, ...] = (
            MetricRange("0-50m", 0.0, 50.0),
            MetricRange("50-90m", 50.0, 90.0),
            MetricRange("90-121m", 90.0, 121.0),
            MetricRange("0-121m", 0.0, 121.0),
        ),
        max_distances: dict[str, float] | None = None,
        min_num_points: int = 0,
        match_cost: MatchCost | str = MatchCost.CENTER,
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
            max_distances: Class name to maximum GT distance in meters, applied as a GT
                filter at accumulation.
            min_num_points: Minimum LiDAR points a GT box needs to count, 0 disables the
                filter.
            match_cost: Cost the matching pairs boxes by, or its name (``center`` or
                ``corner``) as a config spells it.
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
        self.max_distances = max_distances
        self.min_num_points = int(min_num_points)
        # A config typo must fail at build, not after the epoch.
        self.match_cost = MatchCost(match_cost)
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

        if max_distances and not self.class_names:
            raise ValueError("class_names must be provided when max_distances is configured.")
        self._warn_on_range_class_caps()

        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        self.add_state("pred_scores", default=[], dist_reduce_fx=None)
        self.add_state("pred_labels", default=[], dist_reduce_fx=None)
        self.add_state("gt_boxes", default=[], dist_reduce_fx=None)
        self.add_state("gt_labels", default=[], dist_reduce_fx=None)
        # Per-frame flag: does the frame only pad a streaming lane? It is gathered like every
        # other frame, so the ranks hold as many entries, and left out of every metric.
        self.add_state("padding", default=[], dist_reduce_fx=None)
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

    def compute(self) -> MetricReport:
        """Log TTC frame coverage once per epoch, then run the metric groups.

        Returns:
            The suite's report.
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
        if self.max_distances is None:
            return
        for class_name, max_dist in self.max_distances.items():
            for metric_range in self.ranges:
                if metric_range.max_distance is not None and max_dist < metric_range.max_distance:
                    logger.warning(
                        "max_distances['%s'] = %.1fm is smaller than bucket '%s' upper "
                        "bound %.1fm, so the '%s' bucket metrics are misleading for this class.",
                        class_name,
                        max_dist,
                        metric_range.name,
                        metric_range.max_distance,
                        metric_range.name,
                    )

    def requirements(self) -> MetricRequirements:
        """The detection frames, plus the frame metadata and point counts the config reads.

        The collision TTC needs the ego pose and lanelet map path exactly like the
        region filters do, and the ground truth point filter needs the point count of
        every box. Declaring them makes a missing part fail loud at the first batch
        instead of silently reporting zero coverage.

        Returns:
            The requirements at this suite instance's stage.
        """
        return MetricRequirements(
            detection3d=True,
            frame_meta=self.collision is not None or self._filters_need_frame_meta(),
            gt_num_points=self.min_num_points > 0,
        )

    def update(self, eval_output: EvalOutput) -> None:
        """Accumulate one batch, applying every GT filter per frame.

        Args:
            eval_output: The typed eval output the model built for one batch.
        """
        output = eval_output.detection3d
        if output is None:
            raise ValueError(
                "The detection suite needs the detection3d frames, the eval output carries none."
            )
        if self.min_num_points > 0 and not output.check_all_gt_num_points:
            raise ValueError(
                f"min_num_points={self.min_num_points} is configured but the eval output "
                "carries no gt_num_points, the GT filter cannot be applied."
            )

        for i, frame in enumerate(output.frames):
            prediction = frame.predictions
            frame_boxes = frame.gt_boxes.detach().to(dtype=torch.float32)
            frame_labels = frame.gt_labels.detach().to(dtype=torch.long)
            if self.class_names and bool(
                ((frame_labels < 0) | (frame_labels >= len(self.class_names))).any()
            ):
                raise ValueError(
                    f"Frame {i} carries ground truth labels outside the {len(self.class_names)} "
                    "classes, the evaluation pipeline has to drop the boxes of ignored classes."
                )
            num_points = (
                frame.gt_num_points.detach().to(dtype=torch.long, device=frame_boxes.device)
                if frame.gt_num_points is not None
                else None
            )
            keep = gt_keep_mask(
                frame_boxes,
                frame_labels,
                num_points,
                self.class_names or (),
                self.max_distances,
                self.min_num_points,
            )

            self._note_frame()
            pred_boxes = prediction.bboxes_3d.detach().to(dtype=torch.float32)
            pred_labels = prediction.labels_3d.detach().to(dtype=torch.long)
            self.pred_boxes.append(pred_boxes)
            self.pred_scores.append(prediction.scores_3d.detach().to(dtype=torch.float32))
            self.pred_labels.append(pred_labels)
            self.gt_boxes.append(frame_boxes[keep])
            self.gt_labels.append(frame_labels[keep])
            self.padding.append(torch.tensor([frame.is_padding], device=frame_boxes.device))

            if self.collision is not None:
                frame_meta = self._collision_frame_meta(frame.frame_meta)
                covered = self.collision.available(frame_meta.lanelet_map_path)
                self.ttc_covered.append(torch.tensor([covered], device=frame_boxes.device))
                if not frame.is_padding:
                    self.ttc_frames_seen += 1
                    self.ttc_frames_covered += int(covered)
                self.gt_ttc.append(
                    self._frame_ttc(frame_boxes[keep], frame_labels[keep], frame_meta, covered)
                )
                self.pred_ttc.append(
                    self._frame_ttc(
                        pred_boxes,
                        pred_labels,
                        frame_meta,
                        covered,
                        scores=prediction.scores_3d.detach(),
                    )
                )

            if self._component_filters and self._stage_filter_keys():
                available = self._frame_region_availability(frame.frame_meta)
                self.region_available.append(
                    torch.tensor(available, dtype=torch.bool, device=frame_boxes.device)
                )
                self.gt_region_masks.append(
                    self._region_columns(frame_boxes[keep], frame.frame_meta, available)
                )
                self.pred_region_masks.append(
                    self._region_columns(pred_boxes, frame.frame_meta, available)
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

    @staticmethod
    def _collision_frame_meta(frame_meta: FrameMetaSample | None) -> FrameMetaSample:
        """The frame metadata the collision provider reads, guaranteed by the requirements.

        Args:
            frame_meta: Metadata of the frame, ``None`` when the dataset attached none.

        Returns:
            The metadata.

        Raises:
            ValueError: If the frame carries no metadata.
        """
        if frame_meta is None:
            raise ValueError(
                "The collision provider reads the ego pose and lanelet map path of every "
                "frame, the model's build_eval_output must attach the frame metadata."
            )
        return frame_meta

    def _frame_ttc(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        frame_meta: FrameMetaSample,
        covered: bool,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-box collision TTC for one frame, all-``inf`` when uncovered.

        An uncovered frame (map-less scene) is flagged via ``ttc_covered`` so the
        criticality metrics exclude it from their denominators. Predictions carry
        their scores so the propagation can skip the tail no active metric reads.
        The result lives on the boxes' device: every list state must be
        device-consistent for the DDP gather.
        """
        if not covered:
            return torch.full((int(boxes.shape[0]),), float("inf"), device=boxes.device)
        ttc = self.collision.per_box_ttc(
            boxes.detach().cpu().numpy().astype(np.float64),
            labels.detach().cpu().numpy().astype(np.int64),
            frame_meta.ego2global,
            frame_meta.lanelet_map_path,
            scores=None if scores is None else scores.cpu().numpy().astype(np.float64),
            score_floor=self._ttc_score_floor(),
        )
        return torch.from_numpy(np.asarray(ttc, dtype=np.float32)).to(boxes.device)

    def _ttc_score_floor(self) -> float:
        """Lowest prediction score whose TTC any active metric reads.

        The minimum over the metrics that read TTC at all, so one metric that
        integrates the whole score range keeps every box propagated.
        """
        return min(
            component.ttc_score_floor
            for component in self.active_components()
            if component.needs_ttc
        )

    def _frame_region_availability(self, frame_meta: FrameMetaSample | None) -> list[bool]:
        """Per region filter: active at this stage *and* its map present this frame.

        Tallies the per-filter coverage counters (a scene missing its lanelet map
        is excluded from that filter's slice) and returns whether each filter
        should contribute this frame. Inactive filters are ``False`` throughout.
        """
        active = self._stage_filter_keys()
        available = []
        for index, metric_filter in enumerate(self._component_filters):
            is_available = metric_filter.cache_key in active and metric_filter.available(
                self._filter_context(metric_filter, frame_meta)
            )
            if metric_filter.cache_key in active:
                self._note_frame_coverage(index, is_available)
            available.append(is_available)
        return available

    def _region_columns(
        self,
        boxes: torch.Tensor,
        frame_meta: FrameMetaSample | None,
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
                context = self._filter_context(metric_filter, frame_meta)
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
    ) -> Detection3DState:
        """Build the detection state for the requested range and region filter.

        Args:
            metric_range: Optional radial range used to clip predictions and
                ground truth before metric evaluation.
            metric_filter: Optional region filter, boxes are pre-selected by the
                keep-mask computed for it at update time. Frames where the
                filter was unavailable (e.g. a scene without a lanelet map) are
                omitted from the slice entirely, so per-frame denominators only
                count covered frames. Frames that only pad a streaming lane are
                always omitted.

        Returns:
            Detection state consumed by the configured metric components.
        """
        column = self._filter_index(metric_filter)

        samples = []
        for index in range(len(self.gt_boxes)):
            if bool(self.padding[index].item()):
                continue
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
        return Detection3DState(
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
