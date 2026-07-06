"""Tests for 3D detection AP metrics."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.detection3d.heading_ap import HeadingAP
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.nds import Nds
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.detection3d.tp_errors import TpErrors
from autoware_ml.metrics.eval_mixin import MetricEvalMixin


def _suite(recall_targets=None, **kwargs):
    """Build a detection suite with the full standard metric set injected."""
    return Detection3DMetricSuite(
        components=[MeanAP(), HeadingAP(), Nds(), TpErrors(recall_targets=recall_targets)],
        **kwargs,
    )


def _prediction(
    centers: list[tuple[float, float]],
    scores: list[float],
    labels: list[int],
) -> dict[str, torch.Tensor]:
    boxes = torch.zeros((len(centers), 7), dtype=torch.float32)
    if centers:
        boxes[:, :2] = torch.tensor(centers, dtype=torch.float32)
    return {
        "bboxes_3d": boxes,
        "scores_3d": torch.tensor(scores, dtype=torch.float32),
        "labels_3d": torch.tensor(labels, dtype=torch.long),
    }


def _boxes(centers: list[tuple[float, float]]) -> torch.Tensor:
    boxes = torch.zeros((len(centers), 7), dtype=torch.float32)
    if centers:
        boxes[:, :2] = torch.tensor(centers, dtype=torch.float32)
    return boxes


def _detailed_prediction(
    boxes: list[tuple[float, float, float, float, float, float, float, float, float]],
    scores: list[float],
    labels: list[int],
) -> dict[str, torch.Tensor]:
    return {
        "bboxes_3d": torch.tensor(boxes, dtype=torch.float32),
        "scores_3d": torch.tensor(scores, dtype=torch.float32),
        "labels_3d": torch.tensor(labels, dtype=torch.long),
    }


def _detailed_boxes(
    boxes: list[tuple[float, float, float, float, float, float, float, float, float]],
) -> torch.Tensor:
    return torch.tensor(boxes, dtype=torch.float32)


def _update(
    metric: Detection3DMetricSuite,
    predictions: list[dict[str, torch.Tensor]],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    gt_num_points: list[torch.Tensor] | None = None,
) -> None:
    eval_out: dict[str, Any] = {
        "predictions": predictions,
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels,
    }
    if gt_num_points is not None:
        eval_out["gt_num_points"] = gt_num_points
    metric.update(eval_out)


class _DummyDetectionModel(MetricEvalMixin):
    """Minimal model exercising the metric lifecycle through the mixin."""

    def __init__(self, metrics: list[Detection3DMetricSuite]) -> None:
        super().__init__(metrics=metrics)
        self.logged_metrics: list[dict[str, float]] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def build_eval_output(self, batch: dict[str, Any], outputs: Any) -> dict[str, Any]:
        return {
            "predictions": outputs,
            "gt_boxes": batch["gt_boxes"],
            "gt_labels": batch["gt_labels"],
        }

    def log_dict(self, values: dict[str, float], **kwargs: object) -> None:
        del kwargs
        self.logged_metrics.append(values)


def test_center_distance_map_is_one_for_perfect_predictions() -> None:
    metric = _suite(thresholds=(0.5, 1.0))
    _update(
        metric,
        predictions=[_prediction([(0.1, 0.0), (10.0, 10.0)], [0.9, 0.8], [0, 1])],
        gt_boxes=[_boxes([(0.0, 0.0), (10.2, 10.0)])],
        gt_labels=[torch.tensor([0, 1], dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.VAL)

    assert torch.isclose(torch.tensor(metrics["mAP"]), torch.tensor(1.0))


def test_center_distance_ap_penalizes_tight_threshold_misses() -> None:
    # At threshold 0.5m the prediction (0.75m away) misses; at 1.0m it hits.
    metric_tight = _suite(thresholds=(0.5,))
    _update(
        metric_tight,
        predictions=[_prediction([(0.75, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )
    assert torch.isclose(torch.tensor(metric_tight.result(EvalStage.VAL)["mAP"]), torch.tensor(0.0))

    metric_loose = _suite(thresholds=(1.0,))
    _update(
        metric_loose,
        predictions=[_prediction([(0.75, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )
    assert torch.isclose(torch.tensor(metric_loose.result(EvalStage.VAL)["mAP"]), torch.tensor(1.0))


def test_center_distance_ap_ignores_classes_without_ground_truth() -> None:
    metric = _suite(thresholds=(0.5,))
    _update(
        metric,
        predictions=[_prediction([(100.0, 100.0), (0.0, 0.0)], [0.99, 0.9], [1, 0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.VAL)

    assert torch.isclose(torch.tensor(metrics["mAP"]), torch.tensor(1.0))


def test_center_distance_map_reports_class_and_range_breakdowns() -> None:
    metric = _suite(thresholds=(0.5,), class_names=("car", "pedestrian"))
    _update(
        metric,
        predictions=[
            _prediction([(10.0, 0.0), (60.0, 0.0), (100.0, 0.0)], [0.9, 0.8, 0.7], [0, 1, 0]),
        ],
        gt_boxes=[_boxes([(10.0, 0.0), (60.0, 0.0), (100.0, 0.0)])],
        gt_labels=[torch.tensor([0, 1, 0], dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.VAL)

    assert torch.isclose(torch.tensor(metrics["mAP_car"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["mAP_pedestrian"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["mAP_car_0m_50m"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["mAP_pedestrian_50m_90m"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["mAP_car_90m_121m"]), torch.tensor(1.0))


def test_eval_class_range_filters_ground_truth_only_for_total_metric() -> None:
    metric = _suite(
        thresholds=(0.5,),
        class_names=("car",),
        ranges=(MetricRange("0-200m", 0.0, 200.0),),
        eval_class_range={"car": 50.0},
    )
    _update(
        metric,
        predictions=[_prediction([(100.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(100.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )

    assert metric.result(EvalStage.VAL) == {}


def test_eval_class_range_requires_class_names() -> None:
    with pytest.raises(ValueError, match="class_names"):
        _suite(eval_class_range={"car": 50.0})


def test_min_num_points_filters_gt_num_points() -> None:
    metric = _suite(thresholds=(0.5,), min_num_points=2)
    _update(
        metric,
        predictions=[_prediction([(10.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0), (10.0, 0.0)])],
        gt_labels=[torch.tensor([0, 0], dtype=torch.long)],
        gt_num_points=[torch.tensor([1, 5], dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.VAL)

    assert torch.isclose(torch.tensor(metrics["mAP"]), torch.tensor(1.0))


def test_accumulates_predictions_across_multiple_update_calls() -> None:
    metric = _suite(thresholds=(0.5,))
    _update(
        metric,
        predictions=[_prediction([(0.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )
    _update(
        metric,
        predictions=[_prediction([(1.0, 0.0)], [0.8], [0])],
        gt_boxes=[_boxes([(1.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.VAL)

    assert torch.isclose(torch.tensor(metrics["mAP"]), torch.tensor(1.0))


def test_center_distance_ap_reports_imperfect_precision_recall_curve() -> None:
    metric = _suite(thresholds=(0.5,))
    _update(
        metric,
        predictions=[
            _prediction([(0.0, 0.0), (20.0, 0.0), (10.0, 0.0)], [0.9, 0.8, 0.7], [0, 0, 0])
        ],
        gt_boxes=[_boxes([(0.0, 0.0), (10.0, 0.0)])],
        gt_labels=[torch.tensor([0, 0], dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.VAL)

    assert 0.0 < metrics["mAP"] < 1.0


def test_non_integer_detection_range_suffixes_do_not_collide() -> None:
    metric = _suite(
        thresholds=(0.5,),
        ranges=(
            MetricRange("0-50m", 0.0, 50.0),
            MetricRange("0-50.5m", 0.0, 50.5),
        ),
    )
    _update(
        metric,
        predictions=[_prediction([(0.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.VAL)

    assert "mAP_0m_50m" in metrics
    assert "mAP_0m_50p5m" in metrics


def test_metric_eval_mixin_logs_test_map() -> None:
    model = _DummyDetectionModel(metrics=[_suite(thresholds=(0.5,))])
    model.on_test_epoch_start()
    model.on_test_batch_end(
        {"model_outputs": [_prediction([(0.0, 0.0)], [0.9], [0])]},
        {
            "gt_boxes": [_boxes([(0.0, 0.0)])],
            "gt_labels": [torch.tensor([0], dtype=torch.long)],
        },
        batch_idx=0,
    )
    model.on_test_epoch_end()

    assert model.logged_metrics[-1]["test/det3d/mAP"] == pytest.approx(1.0)


def test_metric_eval_mixin_logs_val_map() -> None:
    model = _DummyDetectionModel(metrics=[_suite(thresholds=(0.5,))])
    model.on_validation_epoch_start()
    model.on_validation_batch_end(
        {"model_outputs": [_prediction([(0.0, 0.0)], [0.9], [0])]},
        {
            "gt_boxes": [_boxes([(0.0, 0.0)])],
            "gt_labels": [torch.tensor([0], dtype=torch.long)],
        },
        batch_idx=0,
    )
    model.on_validation_epoch_end()

    assert model.logged_metrics[-1]["val/det3d/mAP"] == pytest.approx(1.0)


def test_metric_eval_mixin_rejects_missing_required_keys() -> None:
    class _BadModel(_DummyDetectionModel):
        def build_eval_output(self, batch: dict[str, Any], outputs: Any) -> dict[str, Any]:
            return {"gt_boxes": batch["gt_boxes"]}  # missing predictions / gt_labels

    model = _BadModel(metrics=[_suite(thresholds=(0.5,))])
    model.on_validation_epoch_start()
    with pytest.raises(ValueError, match="predictions"):
        model.on_validation_batch_end(
            {"model_outputs": [_prediction([(0.0, 0.0)], [0.9], [0])]},
            {"gt_boxes": [_boxes([(0.0, 0.0)])], "gt_labels": [torch.tensor([0])]},
            batch_idx=0,
        )


def test_detailed_center_distance_metrics_report_ap_aph_and_nds() -> None:
    metric = _suite(
        thresholds=(0.5,),
        class_names=("car",),
        ranges=(MetricRange("0-50m", 0.0, 50.0),),
    )
    boxes = [(float(i), 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 1.0, 0.0) for i in range(10)]
    _update(
        metric,
        predictions=[_detailed_prediction(boxes, [1.0 - 0.01 * i for i in range(10)], [0] * 10)],
        gt_boxes=[_detailed_boxes(boxes)],
        gt_labels=[torch.zeros(10, dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.TEST)

    assert torch.isclose(torch.tensor(metrics["mAP"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["mAPH"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["AP_car_0p5m"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["APH_car_0p5m"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["num_match_car_0p5m"]), torch.tensor(10.0))
    assert torch.isclose(torch.tensor(metrics["max_f1_car_0p5m"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["optimal_conf_car_0p5m"]), torch.tensor(0.91))
    assert torch.isclose(torch.tensor(metrics["mATE_default"]), torch.tensor(0.0))
    assert torch.isclose(torch.tensor(metrics["mAOE_default"]), torch.tensor(0.0))
    assert torch.isclose(torch.tensor(metrics["mASE_default"]), torch.tensor(0.0))
    assert torch.isclose(torch.tensor(metrics["mAVE_default"]), torch.tensor(0.0))
    assert torch.isclose(torch.tensor(metrics["mAAE_default"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["map_based_nds"]), torch.tensor(0.9))
    assert torch.isclose(torch.tensor(metrics["mAP_car_0m_50m"]), torch.tensor(1.0))


def test_detailed_center_distance_metrics_penalize_heading_for_maph() -> None:
    metric = _suite(thresholds=(0.5,), class_names=("car",), ranges=())
    pred_boxes = [(float(i), 0.0, 0.0, 4.0, 2.0, 1.5, torch.pi / 2, 0.0, 0.0) for i in range(10)]
    gt_boxes = [(float(i), 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0) for i in range(10)]
    _update(
        metric,
        predictions=[
            _detailed_prediction(pred_boxes, [1.0 - 0.01 * i for i in range(10)], [0] * 10)
        ],
        gt_boxes=[_detailed_boxes(gt_boxes)],
        gt_labels=[torch.zeros(10, dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.TEST)

    assert torch.isclose(torch.tensor(metrics["mAP"]), torch.tensor(1.0))
    assert 0.0 < metrics["mAPH"] < metrics["mAP"]
    assert torch.isclose(torch.tensor(metrics["mAOE_default"]), torch.tensor(torch.pi / 2))


def test_detailed_center_distance_metrics_report_optimal_error_subset() -> None:
    metric = _suite(
        thresholds=(0.5,),
        class_names=("car",),
        ranges=(),
        recall_targets={"default": 0.10},
    )
    _update(
        metric,
        predictions=[
            _detailed_prediction(
                [
                    (0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0),
                    (10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0),
                    (20.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0),
                ],
                [0.9, 0.8, 0.7],
                [0, 0, 0],
            )
        ],
        gt_boxes=[
            _detailed_boxes(
                [
                    (0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0),
                    (20.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0),
                ]
            )
        ],
        gt_labels=[torch.tensor([0, 0], dtype=torch.long)],
    )

    metrics = metric.result(EvalStage.TEST)

    assert torch.isclose(torch.tensor(metrics["num_match_car_0p5m"]), torch.tensor(2.0))
    assert torch.isclose(
        torch.tensor(metrics["tp_error_num_match_car_optimal_0p5m"]), torch.tensor(2.0)
    )
    assert torch.isclose(torch.tensor(metrics["optimal_conf_car_0p5m"]), torch.tensor(0.7))
