"""nuScenes detection score metric (test only by default)."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import (
    curve_metrics,
    mean_tp_errors,
    mean_valid,
    nds,
    select_recall_tp_errors,
)
from autoware_ml.metrics.detection3d.structures import DetectionState


class Nds(Metric[DetectionState]):
    """Combines mean AP (and mean APH) with the TP errors at a recall target into
    the NDS-style summary scores. Self-contained: it computes its own matching and
    errors rather than reading other metrics' outputs.
    """

    def __init__(
        self,
        recall_target: float = 0.10,
        stages: tuple[str, ...] | list[str] = ("test",),
    ) -> None:
        super().__init__(stages)
        self.recall_target = float(recall_target)

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        labels = state.labels(full=True)
        per_class_ap: list[float] = []
        per_class_aph: list[float] = []
        error_dicts: list[dict[str, float]] = []
        for label in labels:
            aps: list[float] = []
            aphs: list[float] = []
            for threshold in state.thresholds:
                curve = state.match_curve(label, threshold)
                metrics = curve_metrics(curve)
                aps.append(metrics.ap)
                aphs.append(metrics.aph)
                error_dicts.append(select_recall_tp_errors(curve, self.recall_target).errors)
            per_class_ap.append(mean_valid(aps))
            per_class_aph.append(mean_valid(aphs))

        mean_ap = mean_valid(per_class_ap)
        mean_aph = mean_valid(per_class_aph)
        errors = mean_tp_errors(error_dicts)
        return {
            "map_based_nds": nds(mean_ap, errors),
            "mapH_based_nds": nds(mean_aph, errors),
        }
