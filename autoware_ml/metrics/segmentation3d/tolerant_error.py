"""Neighbourhood-tolerant error rate.

Point-wise ground-truth labels are not perfect, and for driving a handful of
flipped points do not matter as long as the right class is present right there. A
misclassified point counts as an error only when it is wrong *and* the model
predicted its true class on no point within ``radius``, a genuinely wrong region,
not annotation jitter at a class boundary. The tolerance only ever removes errors,
so this rate is always at most the strict error rate, and it equals it at
``radius = 0``.

Reported whole-scene and per class (over points whose ground truth is that
class). Run in a grouped suite (labels folded onto the behaviour taxonomy), the
per-class values become per-group and intra-group confusion counts as correct.
"""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.segmentation3d.point_cloud import (
    PointCloudSegState,
    class_token,
    valid_point_mask,
)
from autoware_ml.metrics.segmentation3d.spatial import tolerant_error_mask


class NeighbourhoodTolerantErrorRate(Metric[PointCloudSegState]):
    """Fraction of valid points that are wrong with no correct-class neighbour."""

    def __init__(
        self,
        radius: float = 0.2,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Store the rescue radius.

        Args:
            radius: Neighbourhood radius in meters within which a correct-class prediction
                rescues a wrong point. Zero disables the tolerance.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.radius = float(radius)

    def evaluate(self, state: PointCloudSegState, stage: EvalStage) -> dict[str, float]:
        """Report the tolerant error rate, whole-scene and per class.

        Args:
            state: Point-cache state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        num_classes = state.num_classes
        error_total = valid_total = 0
        per_error = [0] * num_classes
        per_valid = [0] * num_classes
        for frame in state.frames:
            valid = valid_point_mask(frame, num_classes, state.ignore_index)
            if not valid.any():
                continue
            coord, pred, target = frame.coord[valid], frame.pred[valid], frame.target[valid]
            # The neighbourhood rescue uses every valid point's prediction, so the
            # mask is computed once over the whole frame and then sliced per class.
            errors = tolerant_error_mask(coord, pred, target, self.radius)
            valid_total += int(valid.sum())
            error_total += int(errors.sum())
            for class_index in range(num_classes):
                is_class = target == class_index
                per_valid[class_index] += int(is_class.sum())
                per_error[class_index] += int((errors & is_class).sum())

        report = {
            "tolerant_error_rate": error_total / valid_total if valid_total else float("nan"),
            "tolerant_error_count": float(error_total),
        }
        for class_index in range(num_classes):
            name = class_token(class_index, state.class_names)
            report[f"tolerant_error_rate_{name}"] = (
                per_error[class_index] / per_valid[class_index]
                if per_valid[class_index]
                else float("nan")
            )
            report[f"tolerant_error_count_{name}"] = float(per_error[class_index])
        return report
