"""Small-object partial-detection score (diagnostic).

For a pedestrian or cone, classifying even a few points correctly is far better
than none, and mIoU, point-averaged and dominated by large classes, cannot see it.
The score groups segmentation points inside each small-object detection box and rewards
partial hits with a saturating credit: the first correct point earns about half
(existence), further points add diminishing refinement up to 1 at all-correct,
and zero correct points score exactly 0. Instances below ``min_points`` are
excluded and reported as ``pd_skipped_low_point_boxes``. Diagnostic only, never
a release gate.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import mean_valid
from autoware_ml.metrics.detection3d.naming import label_metric_name
from autoware_ml.metrics.segmentation3d.point_cloud import (
    PointCloudSegState,
    valid_point_mask,
)


def _points_in_box(coord: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Boolean mask of the given points inside the oriented box.

    The height bound matches the annotation point counter: ``cz`` is the box's
    gravity center, so ground from a lower level or a return from an object
    above cannot be scored as this object's point.
    """
    center = box[:3]
    half = box[3:6] / 2.0
    yaw = float(box[6])
    offset = coord - center
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    local_x = offset[:, 0] * cos_yaw + offset[:, 1] * sin_yaw
    local_y = -offset[:, 0] * sin_yaw + offset[:, 1] * cos_yaw
    return (
        (np.abs(local_x) <= half[0])
        & (np.abs(local_y) <= half[1])
        & (np.abs(offset[:, 2]) <= half[2])
    )


class PartialDetectionScore(Metric[PointCloudSegState]):
    """Saturating credit for partial segmentation of small-object detection boxes.

    The metric refuses to run on a state whose class set differs from
    ``seg_class_names``: a grouped suite folds labels into another index space,
    where the box to class mapping would silently score the wrong class.
    """

    needs_boxes = True

    def __init__(
        self,
        box_label_to_seg_class: dict[int, int],
        seg_class_names: tuple[str, ...] | list[str],
        class_names: tuple[str, ...] | None = None,
        half_saturation: float = 1.0,
        min_points: int = 1,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the box to class mapping against the label spaces.

        Args:
            box_label_to_seg_class: Detection class index to the segmentation class its
                interior points should carry, in the label space named by ``seg_class_names``.
            seg_class_names: Ordered segmentation class names the suite state must match.
            class_names: Ordered detection class names for the reported keys, or ``None`` for
                index-based keys.
            half_saturation: Correct-point count earning half the existence credit.
            min_points: Smallest interior point count a box needs to be scored.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.box_label_to_seg_class = {int(k): int(v) for k, v in box_label_to_seg_class.items()}
        self.seg_class_names = tuple(seg_class_names)
        self.class_names = tuple(class_names) if class_names is not None else None
        self.half_saturation = float(half_saturation)
        self.min_points = int(min_points)
        if self.half_saturation <= 0.0:
            raise ValueError("half_saturation must be > 0 (it is the credit's saturation point).")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1.")
        out_of_range = sorted(
            seg_class
            for seg_class in self.box_label_to_seg_class.values()
            if not 0 <= seg_class < len(self.seg_class_names)
        )
        if out_of_range:
            raise ValueError(
                f"box_label_to_seg_class values {out_of_range} are outside "
                f"seg_class_names ({len(self.seg_class_names)} classes)."
            )
        if self.class_names is not None:
            bad_keys = sorted(
                label
                for label in self.box_label_to_seg_class
                if not 0 <= label < len(self.class_names)
            )
            if bad_keys:
                raise ValueError(
                    f"box_label_to_seg_class keys {bad_keys} are outside class_names "
                    f"({len(self.class_names)} detection classes)."
                )

    def _credit(self, k: int, n: int) -> float:
        h = self.half_saturation
        saturate = lambda x: x / (x + h)  # noqa: E731
        return saturate(k) / saturate(n)

    def evaluate(self, state: PointCloudSegState, stage: EvalStage) -> dict[str, float]:
        """Mean per-instance credit per detection class, over small-object boxes.

        Args:
            state: Point-cache state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        if state.class_names != self.seg_class_names:
            raise ValueError(
                "PartialDetectionScore was configured for segmentation classes "
                f"{self.seg_class_names} but the suite provides {state.class_names}, "
                "the box to class mapping would score the wrong index space."
            )
        credits: dict[int, list[float]] = {label: [] for label in self.box_label_to_seg_class}
        skipped = 0
        for frame in state.frames:
            boxes = [
                (box, int(box_label))
                for box, box_label in zip(frame.gt_boxes, frame.gt_box_labels, strict=True)
                if int(box_label) in self.box_label_to_seg_class
            ]
            if not boxes:
                continue
            valid = valid_point_mask(frame, state.num_classes, state.ignore_index)
            coord = frame.coord[valid][:, :3].astype(np.float64)
            pred = frame.pred[valid]
            if coord.shape[0] == 0:
                skipped += len(boxes)
                continue
            for box, label in boxes:
                # Rectangular prefilter at the footprint's circumradius, then the
                # exact yaw-aware test on the candidates only. A vectorized bound
                # check beats building a spatial index of the whole frame for a
                # handful of small boxes. Tiny epsilon: box is float32, the
                # coordinates float64, so a corner point can otherwise land about
                # 1e-7 outside the circumradius (the exact test still gates).
                radius = float(np.linalg.norm(box[3:5])) / 2.0 * (1.0 + 1e-9)
                center = box[:2].astype(np.float64)
                near = (np.abs(coord[:, 0] - center[0]) <= radius) & (
                    np.abs(coord[:, 1] - center[1]) <= radius
                )
                candidates = np.flatnonzero(near)
                if candidates.shape[0]:
                    inside = candidates[_points_in_box(coord[candidates], box)]
                else:
                    inside = candidates
                n = int(inside.shape[0])
                if n < self.min_points:
                    skipped += 1
                    continue
                correct = int(np.sum(pred[inside] == self.box_label_to_seg_class[label]))
                credits[label].append(self._credit(correct, n))

        report: dict[str, float] = {"pd_skipped_low_point_boxes": float(skipped)}
        per_class_means: list[float] = []
        for label, values in credits.items():
            name = label_metric_name(label, self.class_names)
            if not values:
                report[f"pd_score_{name}"] = float("nan")
                continue
            mean_credit = float(np.mean(values))
            report[f"pd_score_{name}"] = mean_credit
            per_class_means.append(mean_credit)
        report["mpd_score"] = mean_valid(per_class_means)
        return report
