"""nuScenes detection score metric (test only by default)."""

from __future__ import annotations

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.detection3d.matching import (
    DetectionState,
    mean_tp_errors,
    mean_valid,
    nds,
    select_recall_tp_errors,
)
from autoware_ml.metrics.detection3d.structures import (
    DEFAULT_MATCH_THRESHOLDS,
    DEFAULT_TP_THRESHOLD,
    ERROR_NAMES,
)


class Nds(Metric[DetectionState]):
    """Combines mean AP (and mean APH) with the TP errors at a recall target into
    the NDS-style summary scores. Self-contained: it computes its own matching and
    errors rather than reading other metrics' outputs.
    """

    def __init__(
        self,
        recall_target: float = 0.10,
        thresholds: tuple[float, ...] = DEFAULT_MATCH_THRESHOLDS,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the recall target and the operating-point threshold.

        Args:
            recall_target: Recall level in [0, 1] the TP errors are selected up to.
            thresholds: Center-distance match thresholds in meters for the AP part.
            tp_threshold: The single threshold the TP errors are computed at, must be one of
                ``thresholds``.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.recall_target = float(recall_target)
        if not 0.0 <= self.recall_target <= 1.0:
            raise ValueError("recall_target must be in [0, 1].")
        self.thresholds = tuple(float(threshold) for threshold in thresholds)
        if not self.thresholds:
            raise ValueError("thresholds must not be empty.")
        # nuScenes computes the TP errors at dist_th_tp only. Averaging them over
        # all four match thresholds would weight easy classes 4x against hard ones.
        self.tp_threshold = float(tp_threshold)
        if self.tp_threshold not in self.thresholds:
            raise ValueError(
                f"tp_threshold={self.tp_threshold} is not one of thresholds={self.thresholds}, "
                "the aggregate operating point must be an explicitly configured threshold."
            )

    def evaluate(self, state: DetectionState, stage: EvalStage) -> dict[str, float]:
        """Compute NDS-style summary scores from detection match curves.

        Args:
            state: Detection state with cached match curves.
            stage: Evaluation stage requesting the metrics.

        Returns:
            Mapping containing mAP-based and heading-aware NDS values.
        """
        labels = state.labels(full=True)
        per_class_ap: list[float] = []
        per_class_aph: list[float] = []
        error_dicts: list[dict[str, float]] = []
        for label in labels:
            aps: list[float] = []
            aphs: list[float] = []
            for threshold in self.thresholds:
                metrics = state.curve_metrics(label, threshold)
                aps.append(metrics.ap)
                aphs.append(metrics.aph)
            per_class_ap.append(mean_valid(aps))
            per_class_aph.append(mean_valid(aphs))
            # TP errors at the single nuScenes operating point, one entry per
            # class. Only classes with selected true positives contribute, a
            # class absent in this range has no error to measure and must not
            # drag the mean to the worst case.
            selected = select_recall_tp_errors(
                state.match_curve(label, self.tp_threshold), self.recall_target
            )
            if selected.count > 0:
                error_dicts.append(selected.errors)

        mean_ap = mean_valid(per_class_ap)
        mean_aph = mean_valid(per_class_aph)
        errors = (
            mean_tp_errors(error_dicts)
            if error_dicts
            else {name: float("nan") for name in ERROR_NAMES}
        )
        return {
            "map_based_nds": nds(mean_ap, errors),
            "mapH_based_nds": nds(mean_aph, errors),
        }
