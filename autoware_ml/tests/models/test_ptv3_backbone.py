"""Unit tests for PTv3 backbone components."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from autoware_ml.losses.segmentation3d.lovasz import LovaszLoss
from autoware_ml.models.segmentation3d.backbones.ptv3 import (
    Point,
    PointTransformerV3Backbone,
    SerializedAttention,
)
from autoware_ml.models.segmentation3d.ptv3 import PTv3SegmentationModel
from autoware_ml.utils.point_cloud.serialization.default import decode, encode
import autoware_ml.utils.point_cloud.structures as point_structures


def test_serialized_attention_requires_supported_flash_configuration() -> None:
    with pytest.raises(ValueError, match="relative positional encoding"):
        SerializedAttention(
            channels=32,
            num_heads=4,
            patch_size=8,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            order_index=0,
            enable_rpe=True,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
        )


def test_serialized_attention_uses_flash_module_when_enabled() -> None:
    flash_module = SimpleNamespace(
        flash_attn_varlen_qkvpacked_func=lambda qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale: (
            qkv[:, 2]
        )
    )
    point = Point(
        {
            "feat": torch.randn(8, 32),
            "grid_coord": torch.randint(0, 8, (8, 3), dtype=torch.int32),
            "serialized_order": torch.arange(8).reshape(1, 8),
            "serialized_inverse": torch.arange(8).reshape(1, 8),
            "offset": torch.tensor([8], dtype=torch.long),
        }
    )

    with patch(
        "autoware_ml.models.segmentation3d.backbones.ptv3.load_flash_attn_module",
        return_value=flash_module,
    ):
        attention = SerializedAttention(
            channels=32,
            num_heads=4,
            patch_size=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            order_index=0,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
        )
        output = attention(point)

    assert output.feat.shape == (8, 32)
    assert attention.enable_flash is True
    assert attention.flash_attn is flash_module


def test_build_export_module_disables_flash_attention_without_mutating_live_backbone() -> None:
    flash_module = SimpleNamespace(
        flash_attn_varlen_qkvpacked_func=lambda qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale: (
            qkv[:, 2]
        )
    )
    with patch(
        "autoware_ml.models.segmentation3d.backbones.ptv3.load_flash_attn_module",
        return_value=flash_module,
    ):
        attention = SerializedAttention(
            channels=32,
            num_heads=4,
            patch_size=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            order_index=0,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
        )

    class _BackboneForExport(torch.nn.Module):
        def __init__(self, attention_module: SerializedAttention) -> None:
            super().__init__()
            self.order = ["hilbert"]
            self.shuffle_orders = True
            self.attention = attention_module

        def set_serialization_order(self, order: tuple[str, ...]) -> None:
            self.order = list(order)

        def prepare_export_copy(self, order: tuple[str, ...]) -> torch.nn.Module:
            return PointTransformerV3Backbone.prepare_export_copy(self, order)

    model = PTv3SegmentationModel.__new__(PTv3SegmentationModel)
    torch.nn.Module.__init__(model)
    model.seg_head = nn.Linear(4, 2)
    model.backbone = _BackboneForExport(attention)

    with patch(
        "autoware_ml.models.segmentation3d.backbones.ptv3.replace_submconv3d_for_export",
        return_value=None,
    ):
        export_module = PTv3SegmentationModel._build_export_module(
            model,
            sparse_shape=torch.tensor([64, 64, 64], dtype=torch.long),
        )

    export_attention = export_module.backbone.attention
    assert attention.enable_flash is True
    assert attention.patch_size == 4
    assert export_attention.enable_flash is False
    assert export_attention.flash_attn is None
    assert export_attention.patch_size == 0
    assert model.backbone.shuffle_orders is True
    assert export_module.backbone.shuffle_orders is False


def test_compute_metrics_reports_losses_and_point_accuracy() -> None:
    model = PTv3SegmentationModel.__new__(PTv3SegmentationModel)
    torch.nn.Module.__init__(model)
    model.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
    model.lovasz = LovaszLoss(ignore_index=-1, loss_weight=1.0)
    model.ignore_index = -1
    model.num_classes = 3

    outputs = torch.tensor(
        [
            [3.0, 0.1, 0.2],
            [0.2, 2.5, 0.1],
            [0.1, 0.3, 4.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([0, 1, -1], dtype=torch.long)

    metrics = PTv3SegmentationModel.compute_metrics(model, outputs, target)

    assert "loss" in metrics
    assert "loss_ce" in metrics
    assert "loss_lovasz" in metrics
    assert "point_accuracy" in metrics
    assert "mean_iou" in metrics
    assert "mean_f1" in metrics
    assert torch.isclose(metrics["point_accuracy"], torch.tensor(1.0))
    assert metrics["loss"] > 0


def test_get_log_batch_size_uses_offset_length() -> None:
    model = PTv3SegmentationModel.__new__(PTv3SegmentationModel)
    torch.nn.Module.__init__(model)

    batch = {"offset": torch.tensor([128, 256, 320], dtype=torch.int32)}

    assert model.get_log_batch_size(batch) == 3


def test_serialization_helpers_reject_unsupported_orders() -> None:
    grid_coord = torch.tensor([[0, 0, 0]], dtype=torch.int64)

    with pytest.raises(ValueError, match="Unsupported serialization order"):
        encode(grid_coord, depth=4, order="invalid")

    with pytest.raises(ValueError, match="Unsupported serialization order"):
        decode(torch.tensor([0], dtype=torch.int64), depth=4, order="invalid")


def test_point_sparsify_requires_spconv_at_runtime(monkeypatch) -> None:
    monkeypatch.setattr(point_structures, "IS_SPCONV_AVAILABLE", False)
    point = Point(
        {
            "coord": torch.randn(2, 3),
            "grid_coord": torch.randint(0, 4, (2, 3), dtype=torch.int32),
            "feat": torch.randn(2, 4),
            "offset": torch.tensor([2], dtype=torch.long),
        }
    )

    with pytest.raises(ModuleNotFoundError, match="spconv is required"):
        point.sparsify()
