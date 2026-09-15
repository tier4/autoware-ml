"""Test-only point-cache segmentation suite for spatial metrics.

The confusion matrix is enough for IoU, accuracy and precision-recall, but not for the
driving-aware metrics: they need where the errors are (spatial clusters, neighbourhood
tolerance, points inside a detection box) and how confident the model was.

The suite caches per point, per frame: coordinates, predicted and target labels as int16,
the probability of the reported prediction and the normalized entropy, plus the frame's
detection GT boxes. The two uncertainty scalars are reduced from the softmax at update, the
full ``(N, C)`` distribution never reaches the cache. Every metric on top is a stateless
:class:`~autoware_ml.metrics.base.Metric` reading these frames.

A grouped view folds the cached labels onto the behaviour taxonomy at state build and reads
its own uncertainty scalars, reduced at update from the group-marginalized distribution,
under a ``grouped/`` key prefix.

Test only and single process, the cache is CPU held. Validation uses the confusion suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from autoware_ml.metrics.base import IDENTITY, Metric, MetricFilter, MetricRange, MetricSuite
from autoware_ml.metrics.class_groups import fold_labels, fold_labels_tensor, resolve_class_groups


def _reduce_scores(scores: torch.Tensor, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-point confidence and normalized entropy, on the device of the scores.

    Args:
        scores: ``(N, C)`` softmax probabilities per point. Logits would read as
            probabilities here and both outputs would be wrong, so ``update``
            rejects rows that do not sum to one.
        pred: ``(N,)`` reported class per point, indexing the score columns.

    Returns:
        The probability the distribution gives the reported class, and the
        entropy normalized by ``log`` of the distribution's width.
    """
    probs = scores.clamp(1e-12, 1.0)
    entropy = -(probs * probs.log()).sum(dim=1) / float(np.log(scores.shape[1]))
    confidence = (
        scores.gather(1, pred.unsqueeze(1)).squeeze(1) if pred.shape[0] else scores.new_zeros(0)
    )
    return confidence, entropy


@dataclass
class FramePoints:
    """Per-point data for one frame, plus that frame's detection GT boxes.

    ``confidence`` and ``entropy`` are the per-point probability of the reported
    prediction and the normalized predictive entropy, reduced from the full
    distribution at update.
    """

    coord: np.ndarray  # (N, 3) float
    pred: np.ndarray  # (N,) int predicted class
    target: np.ndarray  # (N,) int ground-truth class
    confidence: np.ndarray  # (N,) float probability of the reported prediction
    entropy: np.ndarray  # (N,) float normalized entropy in [0, 1]
    gt_boxes: np.ndarray  # (M, 7+) float detection GT boxes (may be empty)
    gt_box_labels: np.ndarray  # (M,) int detection GT labels (may be empty)


@dataclass
class PointCloudSegState:
    """Synced per-frame point cache handed to each point-level metric.

    In a grouped suite the cached per-point labels and uncertainty scalars are
    already folded onto the behaviour taxonomy, so ``class_names`` are the
    grouped names and metrics read the frames unchanged.
    """

    frames: list[FramePoints]
    num_classes: int
    ignore_index: int
    class_names: tuple[str, ...] | None


class Segmentation3DPointCloudMetricSuite(MetricSuite[PointCloudSegState]):
    """Caches raw per-point segmentation data at test and exposes it per range.

    Point-level metrics (error clusters, neighbourhood-tolerant error, partial
    detection, calibration, entropy) are injected as stateless components.
    """

    prefix = "seg3d_pt"
    _required_keys = ("seg_frames",)

    _FRAME_FIELDS = (
        "coord",
        "pred",
        "target",
        "confidence",
        "entropy",
        "gt_boxes",
        "gt_box_labels",
    )

    def __init__(
        self,
        components: list[Metric[PointCloudSegState]],
        num_classes: int,
        ignore_index: int = -1,
        class_names: tuple[str, ...] | None = None,
        ranges: tuple[MetricRange, ...] = (
            MetricRange("0-50m", 0.0, 50.0),
            MetricRange("50-90m", 50.0, 90.0),
            MetricRange("90-121m", 90.0, 121.0),
            MetricRange("0-121m", 0.0, 121.0),
        ),
        class_groups: dict[str, list[str]] | None = None,
        grouped_components: list[Metric[PointCloudSegState]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Validate the taxonomy setup and register the point-cache states.

        Args:
            components: Injected point-level metrics, as in :class:`MetricSuite`.
            num_classes: Number of trained classes the score columns carry.
            ignore_index: Target label excluded from every metric.
            class_names: Ordered class names for metric keys, or ``None``.
            ranges: Radial windows every key is also emitted for.
            class_groups: Group name to member class names defining the folded view. Comes
                together with ``grouped_components``.
            grouped_components: Metrics reading the behaviour-group view under the
                ``grouped/`` key prefix.
            **kwargs: Forwarded to :class:`MetricSuite`.
        """
        super().__init__(components=components, ranges=ranges, **kwargs)
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2 (entropy is normalized by log C).")
        self.ignore_index = int(ignore_index)
        self.class_names = tuple(class_names) if class_names is not None else None
        # The behaviour-group view is served from the same accumulation: labels
        # are folded at state build, and only the grouped uncertainty scalars
        # (which need the full softmax) are reduced at update alongside the
        # per-class ones. grouped_components read that view under grouped/.
        self._grouped_components = list(grouped_components) if grouped_components else []
        if (class_groups is None) != (not self._grouped_components):
            raise ValueError(
                "class_groups and grouped_components come together: the groups define "
                "the folded view, the components read it."
            )
        self._group_lut, self._grouped_names = (
            resolve_class_groups(self.class_names, class_groups) if class_groups else (None, None)
        )
        if self._group_lut is not None:
            if len(self._grouped_names) < 2:
                raise ValueError("class_groups must keep >= 2 groups (entropy normalization).")
            # Column-folding matrix marginalizing scores over grouped members.
            fold = np.zeros((self.num_classes, len(self._grouped_names)), dtype=np.float32)
            fold[np.arange(self.num_classes), self._group_lut] = 1.0
            self._score_fold = fold
            self.add_state("grouped_confidence", default=[], dist_reduce_fx=None)
            self.add_state("grouped_entropy", default=[], dist_reduce_fx=None)
        for name in self._FRAME_FIELDS:
            self.add_state(name, default=[], dist_reduce_fx=None)

        # Filters, resolved to per-point keep-masks at update time (see the
        # detection suite for the same DDP-safe pattern).
        self._register_component_filters()
        if self._component_filters:
            self.add_state("region_masks", default=[], dist_reduce_fx=None)
            self.add_state("box_region_masks", default=[], dist_reduce_fx=None)

    def _filterable_components(self) -> list[Metric[PointCloudSegState]]:
        """Both views' components: the grouped ones read the same masks."""
        return [*self.components, *self._grouped_components]

    def _active_grouped_components(self) -> list[Metric[PointCloudSegState]]:
        return [
            component for component in self._grouped_components if self._stage in component.stages
        ]

    def _needs_boxes(self) -> bool:
        """Whether a component reporting at this stage reads the detection GT.

        Stage-scoped like the filter keys: a test-only box reader must not make
        another stage demand annotations it never reports on.
        """
        return any(
            component.needs_boxes
            for component in (*self.active_components(), *self._active_grouped_components())
        )

    def runs_at(self, stage) -> bool:
        """Whether any component, per-class or grouped, reports at ``stage``.

        Args:
            stage: Evaluation stage to probe.

        Returns:
            Whether at least one component reports at ``stage``.
        """
        return super().runs_at(stage) or any(
            stage in component.stages for component in self._grouped_components
        )

    def required_keys(self) -> tuple[str, ...]:
        """``seg_frames`` plus any component-declared eval-output keys.

        Filter context keys (ego pose, scene token) live *inside* each frame
        entry, not at the eval-output top level, so they are validated per frame
        in ``update`` instead. Component-level keys stay in the batch-0 check.

        Returns:
            The required eval-output key names.
        """
        extra = tuple(
            key
            for component in (*self.active_components(), *self._active_grouped_components())
            for key in component.required_eval_keys
        )
        return tuple(dict.fromkeys(type(self)._required_keys + extra))

    @staticmethod
    def _require_single_process() -> None:
        """The raw point cache is CPU-held and single-process by design.

        Its list states would need one collective per frame and would replicate
        the full multi-gigabyte cache onto every rank. Run the test stage on one
        device instead of silently paying (or crashing on) that sync.
        """
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise RuntimeError(
                "Segmentation3DPointCloudMetricSuite caches raw per-point data and is "
                "single-process by design, run the test stage on a single device."
            )

    def compute(self) -> dict[str, float]:
        """Per-class components on the trained view, grouped ones under ``grouped/``.

        Also guards the sync path: a rank that saw no batches never ran update.

        Returns:
            Metric keys mapped to values.
        """
        self._require_single_process()
        report = super().compute()
        grouped_active = self._active_grouped_components()
        if grouped_active:
            for key, value in self._compute_report(grouped_active, self._grouped_state_for).items():
                grouped_key = f"grouped/{key}"
                if grouped_key in report:
                    raise ValueError(f"Grouped view emits a colliding key {grouped_key!r}.")
                report[grouped_key] = value
        return report

    def update(self, eval_out: dict[str, Any]) -> None:
        """Append one batch of per-frame point data to the cache.

        The softmax is reduced on its own device, so only the per-point scalars cross to
        the CPU.

        Args:
            eval_out: Flat eval-output dict with the per-frame ``seg_frames``.
        """
        self._require_single_process()
        active = self._stage_filter_keys()
        needs_boxes = self._needs_boxes()
        for frame in eval_out["seg_frames"]:
            missing = [key for key in ("coord", "pred", "target", "scores") if key not in frame]
            if missing:
                raise ValueError(
                    f"seg_frames entries need {missing}, the model's build_eval_output "
                    "must supply them."
                )
            coord = frame["coord"].detach().to(dtype=torch.float32)
            pred = frame["pred"].detach().reshape(-1).to(dtype=torch.long)
            target = frame["target"].detach().reshape(-1).to(dtype=torch.long)
            scores = frame["scores"].detach().to(dtype=torch.float32)
            num_points = coord.shape[0]
            if not (pred.shape[0] == target.shape[0] == scores.shape[0] == num_points):
                raise ValueError(
                    f"per-point arrays disagree: coord {num_points}, pred {pred.shape[0]}, "
                    f"target {target.shape[0]}, scores {scores.shape[0]}."
                )
            if num_points and scores.shape[1] != self.num_classes:
                raise ValueError(
                    f"scores carry {scores.shape[1]} columns for {self.num_classes} "
                    "classes, entropy normalization and the cache would silently drift."
                )
            if num_points:
                row_sums = scores.sum(dim=1)
                if float(scores.min()) < 0.0 or not torch.allclose(
                    row_sums, torch.ones_like(row_sums), atol=1e-3
                ):
                    raise ValueError(
                        "scores must be softmax probabilities, the confidence and the "
                        "grouped fold read them as such."
                    )
            if num_points:
                pred_min, pred_max = (int(v) for v in torch.aminmax(pred))
                if pred_min < 0 or pred_max >= scores.shape[1]:
                    raise ValueError("predicted labels must index the score columns.")
                target_min, target_max = (int(v) for v in torch.aminmax(target))
                if target_min < -(2**15) or target_max >= 2**15:
                    raise ValueError("target labels exceed the int16 cache range.")
            confidence_values, entropy = _reduce_scores(scores, pred)
            coord_cpu = coord.cpu()
            self.coord.append(coord_cpu)
            self.pred.append(pred.to(torch.int16).cpu())
            self.target.append(target.to(torch.int16).cpu())
            self.confidence.append(confidence_values.cpu())
            self.entropy.append(entropy.to(torch.float32).cpu())
            if self._group_lut is not None:
                folded_scores = scores @ torch.as_tensor(self._score_fold, device=scores.device)
                folded_pred = fold_labels_tensor(pred, self._group_lut)
                grouped_confidence, grouped_entropy = _reduce_scores(folded_scores, folded_pred)
                self.grouped_confidence.append(grouped_confidence.cpu())
                self.grouped_entropy.append(grouped_entropy.to(torch.float32).cpu())
            boxes = frame.get("gt_boxes")
            labels = frame.get("gt_box_labels")
            if needs_boxes and (boxes is None or labels is None):
                raise ValueError(
                    "a configured component reads detection ground truth, seg_frames "
                    "entries must carry gt_boxes and gt_box_labels (empty tensors for "
                    "a frame without objects)."
                )
            self.gt_boxes.append(
                torch.zeros((0, 7), dtype=torch.float32)
                if boxes is None
                else boxes.detach().to(dtype=torch.float32).cpu()
            )
            self.gt_box_labels.append(
                torch.zeros((0,), dtype=torch.long)
                if labels is None
                else labels.detach().to(dtype=torch.long).cpu()
            )
            if self._component_filters and active:
                # filters convert the coordinates to float64 themselves
                point_masks, box_masks = self._filter_masks(
                    frame, coord_cpu.numpy(), self.gt_boxes[-1].numpy(), active
                )
                self.region_masks.append(point_masks)
                self.box_region_masks.append(box_masks)

    def _build_frames(
        self,
        confidences: list[torch.Tensor],
        entropies: list[torch.Tensor],
        metric_range: MetricRange | None,
        metric_filter: MetricFilter,
        fold_lut: np.ndarray | None = None,
    ) -> list[FramePoints]:
        """Slice every cached frame for one bucket, on one view of the taxonomy.

        Args:
            confidences: Per-frame confidence vectors of the view being built.
            entropies: Per-frame entropy vectors of the view being built.
            metric_range: Radial window in meters, or ``None`` for no clipping.
            metric_filter: Spatial filter selecting the bucket's points.
            fold_lut: Trained-index to grouped-index LUT, or ``None`` per class.

        Returns:
            One sliced :class:`FramePoints` per cached frame.
        """
        column = self._filter_index(metric_filter)
        return [
            self._frame(
                self.coord[index],
                self.pred[index],
                self.target[index],
                confidences[index],
                entropies[index],
                self.gt_boxes[index],
                self.gt_box_labels[index],
                metric_range,
                None if column is None else self.region_masks[index][:, column].cpu().numpy(),
                None if column is None else self.box_region_masks[index][:, column].cpu().numpy(),
                fold_lut=fold_lut,
            )
            for index in range(len(self.coord))
        ]

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter = IDENTITY
    ) -> PointCloudSegState:
        """Build the point-cache state, clipping points/boxes to range and region.

        Args:
            metric_range: Radial window in meters, or ``None`` for no clipping.
            metric_filter: Spatial filter selecting the bucket's points.

        Returns:
            The point-cache state for the bucket.
        """
        return PointCloudSegState(
            frames=self._build_frames(self.confidence, self.entropy, metric_range, metric_filter),
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            class_names=self.class_names,
        )

    def _grouped_state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter = IDENTITY
    ) -> PointCloudSegState:
        """The behaviour-group view of the same cache.

        Labels are folded through the group LUT after slicing, and the uncertainty
        scalars come from the group-marginalized softmax reduced at update.
        """
        return PointCloudSegState(
            frames=self._build_frames(
                self.grouped_confidence,
                self.grouped_entropy,
                metric_range,
                metric_filter,
                fold_lut=self._group_lut,
            ),
            num_classes=len(self._grouped_names),
            ignore_index=self.ignore_index,
            class_names=self._grouped_names,
        )

    def _frame(
        self,
        coord: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence_values: torch.Tensor,
        entropy_values: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        metric_range: MetricRange | None,
        region_mask: np.ndarray | None,
        box_region_mask: np.ndarray | None,
        fold_lut: np.ndarray | None = None,
    ) -> FramePoints:
        """Slice the cached frame. ``fold_lut`` maps labels onto the grouped view."""
        coord_np = coord.numpy()
        boxes_np = boxes.numpy()
        labels_np = labels.numpy()

        keep = np.ones(coord_np.shape[0], dtype=bool) if region_mask is None else region_mask
        if metric_range is not None:
            keep = keep & self._range_mask(coord_np[:, :2], metric_range)
        # Boxes are clipped by the same filter (footprint membership resolved at
        # update) and by the radial range on their centers.
        if boxes_np.shape[0]:
            box_keep = (
                np.ones(boxes_np.shape[0], dtype=bool)
                if box_region_mask is None
                else box_region_mask
            )
            if metric_range is not None:
                box_keep = box_keep & self._range_mask(boxes_np[:, :2], metric_range)
            boxes_np, labels_np = boxes_np[box_keep], labels_np[box_keep]

        # Nothing sliced away (e.g. the whole-scene, no-range state) shares the cached
        # arrays instead of copying them, so those are frozen below: an accidental
        # in-place edit in a component must not corrupt the epoch's cache.
        whole = bool(keep.all())
        sliced = [
            array if whole else array[keep]
            for array in (
                coord_np,
                pred.numpy(),
                target.numpy(),
                confidence_values.numpy(),
                entropy_values.numpy(),
            )
        ]
        if fold_lut is not None:
            sliced[1] = fold_labels(sliced[1], fold_lut)
            sliced[2] = fold_labels(sliced[2], fold_lut)
        frame = (*sliced, boxes_np, labels_np)
        if whole:
            for array in frame:
                array.flags.writeable = False
        return FramePoints(*frame)

    def _filter_masks(
        self,
        frame: dict[str, Any],
        coord_np: np.ndarray,
        boxes_np: np.ndarray,
        active: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve every configured filter to one keep-mask column, for points and boxes.

        A filter the stage does not read, and a scene the filter has no map for, keep
        nothing, so the frame drops out of that slice instead of leaking into it.

        Args:
            frame: One ``seg_frames`` entry, the source of the filter context.
            coord_np: ``(N, 3+)`` point coordinates of the frame.
            boxes_np: ``(M, 7+)`` detection GT boxes of the frame.
            active: Cache keys of the filters this stage reads.

        Returns:
            The ``(N, F)`` point mask and the ``(M, F)`` box mask, one column per filter.
        """
        columns = []
        box_columns = []
        for index, metric_filter in enumerate(self._component_filters):
            context = (
                self._frame_filter_context(index, metric_filter, frame)
                if metric_filter.cache_key in active
                else None
            )
            if context is None:
                columns.append(np.zeros(coord_np.shape[0], dtype=bool))
                box_columns.append(np.zeros(boxes_np.shape[0], dtype=bool))
                continue
            # xyz only: a wide per-point array must never trip the filters' box-vs-point
            # dispatch. Boxes go through the same filter as full rows, footprint-overlap
            # membership.
            columns.append(metric_filter.keep(coord_np[:, :3], context))
            box_columns.append(metric_filter.keep(boxes_np, context))
        return (
            self._stack_masks(columns, coord_np.shape[0]),
            self._stack_masks(box_columns, boxes_np.shape[0]),
        )

    def _stack_masks(self, columns: list[np.ndarray], num_rows: int) -> torch.Tensor:
        """Stack one keep-mask column per filter into a ``(num_rows, F)`` tensor.

        Args:
            columns: One boolean column per configured filter.
            num_rows: Rows the masks cover, points or boxes of the frame.

        Returns:
            The stacked mask.
        """
        stacked = (
            np.stack(columns, axis=1)
            if num_rows
            else np.zeros((0, len(self._component_filters)), dtype=bool)
        )
        return torch.from_numpy(np.asarray(stacked, dtype=bool))

    @staticmethod
    def _range_mask(xy: np.ndarray, metric_range: MetricRange) -> np.ndarray:
        distance = np.linalg.norm(xy, axis=1)
        keep = distance >= metric_range.min_distance
        if metric_range.max_distance is not None:
            keep &= distance < metric_range.max_distance
        return keep


def valid_point_mask(frame: FramePoints, num_classes: int, ignore_index: int) -> np.ndarray:
    """Mask of points with an in-range target and prediction (ignore excluded).

    Args:
        frame: Cached frame points.
        num_classes: Number of trained classes.
        ignore_index: Target value excluded from evaluation.

    Returns:
        Boolean mask over the frame's points.
    """
    return (
        (frame.target != ignore_index)
        & (frame.target >= 0)
        & (frame.target < num_classes)
        & (frame.pred >= 0)
        & (frame.pred < num_classes)
    )


def class_token(index: int, class_names: tuple[str, ...] | None) -> str:
    """Per-class key token, falling back to ``class_{index}`` when names are absent.

    Args:
        index: Class index.
        class_names: Class names in index order, or ``None``.

    Returns:
        The class token.
    """
    if class_names is not None and index < len(class_names):
        return class_names[index]
    return f"class_{index}"


def normalized_entropy(scores: np.ndarray) -> np.ndarray:
    """Per-point Shannon entropy of the softmax distribution, normalized to [0, 1].

    Args:
        scores: Softmax scores ``(N, C)``.

    Returns:
        Per-point normalized entropy ``(N,)``.
    """
    num_classes = scores.shape[1]
    probs = np.clip(scores, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return entropy / np.log(num_classes)


def confidence(scores: np.ndarray) -> np.ndarray:
    """Per-point max softmax probability.

    Matches the cached :attr:`FramePoints.confidence` for an argmax prediction,
    which is what a reference implementation of the cache reduction needs.

    Args:
        scores: Softmax scores ``(N, C)``.

    Returns:
        Per-point maximum probability ``(N,)``.
    """
    return scores.max(axis=1)
