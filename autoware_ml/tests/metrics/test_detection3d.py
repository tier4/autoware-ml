"""Tests for 3D detection AP metrics."""

from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

from autoware_ml.metrics.detection3d import CenterDistanceMeanAP, DetectionRange, _gather_samples
from autoware_ml.models.detection3d.base import Detection3DBaseModel


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


class _DummyDetectionModel(Detection3DBaseModel):
    def __init__(self) -> None:
        self.logged_metrics: list[dict[str, torch.Tensor]] = []
        super().__init__()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, **kwargs):
        pass

    def compute_metrics(self, batch_inputs_dict, outputs):
        return {"loss": torch.tensor(0.0)}

    def decode_detection_predictions(
        self,
        outputs: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        return outputs

    def log_dict(self, values: dict[str, torch.Tensor], **kwargs: object) -> None:
        del kwargs
        self.logged_metrics.append(values)


def test_center_distance_map_is_one_for_perfect_predictions() -> None:
    metric = CenterDistanceMeanAP(thresholds=(0.5, 1.0))
    metric.update(
        predictions=[
            _prediction([(0.1, 0.0), (10.0, 10.0)], [0.9, 0.8], [0, 1]),
        ],
        gt_boxes=[_boxes([(0.0, 0.0), (10.2, 10.0)])],
        gt_labels=[torch.tensor([0, 1], dtype=torch.long)],
    )

    metrics = metric.compute()

    assert torch.isclose(torch.tensor(metrics["mAP"]), torch.tensor(1.0))


def test_center_distance_ap_penalizes_tight_threshold_misses() -> None:
    # At threshold 0.5m the prediction (0.75m away) misses; at 1.0m it hits.
    metric_tight = CenterDistanceMeanAP(thresholds=(0.5,))
    metric_tight.update(
        predictions=[_prediction([(0.75, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )
    assert torch.isclose(torch.tensor(metric_tight.compute()["mAP"]), torch.tensor(0.0))

    metric_loose = CenterDistanceMeanAP(thresholds=(1.0,))
    metric_loose.update(
        predictions=[_prediction([(0.75, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )
    assert torch.isclose(torch.tensor(metric_loose.compute()["mAP"]), torch.tensor(1.0))


def test_center_distance_ap_ignores_classes_without_ground_truth() -> None:
    metric = CenterDistanceMeanAP(thresholds=(0.5,))
    metric.update(
        predictions=[_prediction([(100.0, 100.0), (0.0, 0.0)], [0.99, 0.9], [1, 0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )

    metrics = metric.compute()

    assert torch.isclose(torch.tensor(metrics["mAP"]), torch.tensor(1.0))


def test_center_distance_map_reports_class_and_range_breakdowns() -> None:
    metric = CenterDistanceMeanAP(thresholds=(0.5,), class_names=("car", "pedestrian"))
    metric.update(
        predictions=[
            _prediction(
                [(10.0, 0.0), (60.0, 0.0), (100.0, 0.0)],
                [0.9, 0.8, 0.7],
                [0, 1, 0],
            ),
        ],
        gt_boxes=[
            _boxes([(10.0, 0.0), (60.0, 0.0), (100.0, 0.0)]),
        ],
        gt_labels=[torch.tensor([0, 1, 0], dtype=torch.long)],
    )

    metrics = metric.compute()

    assert torch.isclose(torch.tensor(metrics["mAP_car"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["mAP_pedestrian"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["mAP_car_0m_50m"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["mAP_pedestrian_50m_90m"]), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics["mAP_car_90m_121m"]), torch.tensor(1.0))


def test_eval_class_range_filters_ground_truth_only_for_total_metric() -> None:
    metric = CenterDistanceMeanAP(
        thresholds=(0.5,),
        class_names=("car",),
        ranges=(DetectionRange("0-200m", 0.0, 200.0),),
        eval_class_range={"car": 50.0},
    )
    metric.update(
        predictions=[_prediction([(100.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(100.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )

    assert metric.compute() == {}


def test_filter_attributes_excludes_matching_ground_truth() -> None:
    metric = CenterDistanceMeanAP(
        thresholds=(0.5,),
        class_names=("bicycle",),
        filter_attributes=[("bicycle", "cycle_state.without_rider")],
    )
    metric.update(
        predictions=[_prediction([(0.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
        gt_attributes=[[["cycle_state.without_rider"]]],
    )

    assert metric.compute() == {}


def test_filter_attributes_excludes_omegaconf_list_config_ground_truth() -> None:
    metric = CenterDistanceMeanAP(
        thresholds=(0.5,),
        class_names=("bicycle",),
        filter_attributes=OmegaConf.create([["bicycle", "cycle_state.without_rider"]]),
    )
    metric.update(
        predictions=[_prediction([(0.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
        gt_attributes=[[["cycle_state.without_rider"]]],
    )

    assert metric.compute() == {}


def test_class_based_filters_require_class_names() -> None:
    with pytest.raises(ValueError, match="class_names"):
        CenterDistanceMeanAP(filter_attributes=[("bicycle", "cycle_state.without_rider")])


def test_min_point_numbers_filters_gt_num_points() -> None:
    metric = CenterDistanceMeanAP(thresholds=(0.5,), min_point_numbers=2)
    metric.update(
        predictions=[_prediction([(10.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0), (10.0, 0.0)])],
        gt_labels=[torch.tensor([0, 0], dtype=torch.long)],
        gt_num_points=[torch.tensor([1, 5], dtype=torch.long)],
    )

    metrics = metric.compute()

    assert torch.isclose(torch.tensor(metrics["mAP"]), torch.tensor(1.0))


def test_gather_samples_uses_distributed_all_gather(monkeypatch) -> None:
    local_metric = CenterDistanceMeanAP()
    local_metric.update(
        predictions=[_prediction([(0.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )
    remote_metric = CenterDistanceMeanAP()
    remote_metric.update(
        predictions=[_prediction([(1.0, 0.0)], [0.8], [0])],
        gt_boxes=[_boxes([(1.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )

    monkeypatch.setattr("autoware_ml.metrics.detection3d.dist.is_available", lambda: True)
    monkeypatch.setattr("autoware_ml.metrics.detection3d.dist.is_initialized", lambda: True)
    monkeypatch.setattr("autoware_ml.metrics.detection3d.dist.get_world_size", lambda: 2)

    def all_gather_object(output, samples):
        output[0] = samples
        output[1] = remote_metric.samples

    monkeypatch.setattr("autoware_ml.metrics.detection3d.dist.all_gather_object", all_gather_object)

    gathered = _gather_samples(local_metric.samples)

    assert len(gathered) == 2


def test_center_distance_ap_reports_imperfect_precision_recall_curve() -> None:
    metric = CenterDistanceMeanAP(thresholds=(0.5,))
    metric.update(
        predictions=[
            _prediction(
                [(0.0, 0.0), (20.0, 0.0), (10.0, 0.0)],
                [0.9, 0.8, 0.7],
                [0, 0, 0],
            )
        ],
        gt_boxes=[_boxes([(0.0, 0.0), (10.0, 0.0)])],
        gt_labels=[torch.tensor([0, 0], dtype=torch.long)],
    )

    metrics = metric.compute()

    assert 0.0 < metrics["mAP"] < 1.0


def test_noninteger_detection_range_suffixes_do_not_collide() -> None:
    metric = CenterDistanceMeanAP(
        thresholds=(0.5,),
        ranges=(
            DetectionRange("0-50m", 0.0, 50.0),
            DetectionRange("0-50.5m", 0.0, 50.5),
        ),
    )
    metric.update(
        predictions=[_prediction([(0.0, 0.0)], [0.9], [0])],
        gt_boxes=[_boxes([(0.0, 0.0)])],
        gt_labels=[torch.tensor([0], dtype=torch.long)],
    )

    metrics = metric.compute()

    assert "mAP_0m_50m" in metrics
    assert "mAP_0m_50p5m" in metrics


def test_detection_epoch_metrics_logs_test_map() -> None:
    model = _DummyDetectionModel()
    model.on_test_epoch_start()
    model._accumulate_detection_map(
        "test",
        {"model_outputs": [_prediction([(0.0, 0.0)], [0.9], [0])]},
        {
            "gt_boxes": [_boxes([(0.0, 0.0)])],
            "gt_labels": [torch.tensor([0], dtype=torch.long)],
        },
    )
    model.on_test_epoch_end()

    assert torch.isclose(model.logged_metrics[-1]["test/mAP"], torch.tensor(1.0))


def test_detection_validation_batch_hook_accumulates_map() -> None:
    model = _DummyDetectionModel()
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

    assert torch.isclose(model.logged_metrics[-1]["val/mAP"], torch.tensor(1.0))
