"""Tests for shared detection2d model wrappers."""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from autoware_ml.models.detection2d.rtdetrv4.model import RTDETRv4DetectionModel
from autoware_ml.models.detection2d.rtdetrv4.postprocessor import PostProcessor


class _DummyBackbone(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images


class _DummyEncoder(nn.Module):
    def forward(self, backbone_out: torch.Tensor) -> torch.Tensor:
        return backbone_out


class _DummyDecoder(nn.Module):
    def forward(
        self,
        encoded: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        del encoded, targets
        return {
            "pred_logits": torch.zeros(1, 3, 2),
            "pred_boxes": torch.full((1, 3, 4), 0.5),
        }


class _DummyCriterion(nn.Module):
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        del outputs, targets
        return {"loss_bbox": torch.tensor(1.5), "loss_giou": torch.tensor(0.5)}


class _DummyPostprocessor(nn.Module):
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        orig_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        del outputs, orig_sizes
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
        ]

    def deploy(self) -> "_DummyPostprocessor":
        return self


def test_rtdetr_wrapper_computes_total_loss_and_predictions() -> None:
    model = RTDETRv4DetectionModel(
        backbone=_DummyBackbone(),
        encoder=_DummyEncoder(),
        decoder=_DummyDecoder(),
        criterion=_DummyCriterion(),
        postprocessor=_DummyPostprocessor(),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
    )

    outputs = model(images=torch.rand(1, 3, 16, 16), targets=[])
    metrics = model.compute_metrics(outputs, targets=[])
    predictions = model.postprocess_predictions(outputs, torch.tensor([[16, 16]]))

    assert set(outputs) == {"pred_logits", "pred_boxes"}
    assert torch.isclose(metrics["loss"], torch.tensor(2.0))
    assert predictions[0]["labels"].tolist() == [1]


def test_rtdetr_export_spec_exposes_images_and_orig_sizes() -> None:
    model = RTDETRv4DetectionModel(
        backbone=_DummyBackbone(),
        encoder=_DummyEncoder(),
        decoder=_DummyDecoder(),
        criterion=_DummyCriterion(),
        postprocessor=_DummyPostprocessor(),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
    )

    spec = model.build_export_spec(
        {
            "images": torch.rand(1, 3, 16, 16),
            "orig_sizes": torch.tensor([[16, 16]], dtype=torch.int64),
        }
    )

    assert spec.input_param_names == ["images", "orig_sizes"]
    assert spec.output_names == ["pred_labels", "pred_boxes", "pred_scores"]


def test_postprocessor_scales_xy_coordinates_with_image_width() -> None:
    postprocessor = PostProcessor(num_classes=2, use_focal_loss=True, num_top_queries=1)
    outputs = {
        "pred_logits": torch.tensor([[[10.0, -10.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
    }
    predictions = postprocessor(outputs, torch.tensor([[10, 20]], dtype=torch.int64))

    assert predictions[0]["boxes"].tolist() == [[5.0, 2.5, 15.0, 7.5]]


def test_rtdetr_wrapper_can_initialize_detector_from_raw_checkpoint() -> None:
    with patch(
        "autoware_ml.models.detection2d.rtdetrv4.model.load_model_from_raw_checkpoint"
    ) as load_mock:
        model = RTDETRv4DetectionModel(
            backbone=_DummyBackbone(),
            encoder=_DummyEncoder(),
            decoder=_DummyDecoder(),
            criterion=_DummyCriterion(),
            postprocessor=_DummyPostprocessor(),
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            init_checkpoint_path="/tmp/upstream_rtdetr.pth",
            init_checkpoint_state_key="ema.module",
            init_checkpoint_filter_mismatched_shapes=True,
            init_checkpoint_strict=False,
        )

    assert model.detector is not None
    load_mock.assert_called_once()
    args, kwargs = load_mock.call_args
    assert args[0] is model.detector
    assert str(args[1]) == "/tmp/upstream_rtdetr.pth"
    assert kwargs["state_key"] == "ema.module"
    assert kwargs["filter_mismatched_shapes"] is True
    assert kwargs["strict"] is False


def test_rtdetr_wrapper_skips_dataset_metrics_on_validation_when_disabled() -> None:
    model = RTDETRv4DetectionModel(
        backbone=_DummyBackbone(),
        encoder=_DummyEncoder(),
        decoder=_DummyDecoder(),
        criterion=_DummyCriterion(),
        postprocessor=_DummyPostprocessor(),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        compute_coco_metrics_on_val=False,
        compute_localization_metrics_on_val=False,
    )
    dataset = SimpleNamespace(
        label_to_category_id={0: 1},
        get_coco_api=lambda: object(),
    )
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(val_dataset=dataset))
    model._val_predictions = [
        {
            "image_id": 1,
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]

    with (
        patch("autoware_ml.models.detection2d.base.evaluate_coco_predictions") as coco_mock,
        patch(
            "autoware_ml.models.detection2d.base.evaluate_class_agnostic_localization"
        ) as localization_mock,
        patch.object(model, "log_dict") as log_mock,
    ):
        model.on_validation_epoch_end()

    coco_mock.assert_not_called()
    localization_mock.assert_not_called()
    log_mock.assert_not_called()


def test_rtdetr_wrapper_keeps_dataset_metrics_on_test_by_default() -> None:
    model = RTDETRv4DetectionModel(
        backbone=_DummyBackbone(),
        encoder=_DummyEncoder(),
        decoder=_DummyDecoder(),
        criterion=_DummyCriterion(),
        postprocessor=_DummyPostprocessor(),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        compute_coco_metrics_on_val=False,
        compute_localization_metrics_on_val=False,
    )
    dataset = SimpleNamespace(
        label_to_category_id={0: 1},
        get_coco_api=lambda: object(),
    )
    model._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(test_dataset=dataset),
        global_rank=0,
    )
    model._test_predictions = [
        {
            "image_id": 1,
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]

    with (
        patch(
            "autoware_ml.models.detection2d.base.evaluate_coco_predictions",
            return_value={"mAP": 0.5},
        ) as coco_mock,
        patch(
            "autoware_ml.models.detection2d.base.evaluate_class_agnostic_localization",
            return_value={"gt_recall_0p5": 0.25},
        ) as localization_mock,
        patch.object(model, "log_dict") as log_mock,
    ):
        model.on_test_epoch_end()

    coco_mock.assert_called_once()
    localization_mock.assert_called_once()
    log_mock.assert_called_once()
