"""Unit tests for PTv3 encoder components."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import autoware_ml.utils.point_cloud.structures as point_structures
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Point,
    PointSequential,
    PointTransformerV3Encoder,
    SerializedAttention,
    SerializedPooling,
    build_serialized_pooling_meta,
)
from autoware_ml.models.segmentation3d.ptv3 import (
    PTv3SegmentationModel,
    _PTv3SegmentationExportModule,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    build_ptv3_encoder_dynamic_axes,
    validate_serialization_geometry,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.tests.models.ptv3_detection_fixtures import (
    build_inputs,
    build_ptv3_encoder,
    build_seg_head,
    move_batch_to_device,
)


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
        flash_attn_varlen_qkvpacked_func=lambda qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        deterministic=False: (qkv[:, 2])
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
        "autoware_ml.models.segmentation3d.encoders.ptv3.load_flash_attn_module",
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


def test_build_export_module_disables_flash_attention_without_mutating_live_encoder() -> None:
    flash_module = SimpleNamespace(
        flash_attn_varlen_qkvpacked_func=lambda qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        deterministic=False: (qkv[:, 2])
    )
    with patch(
        "autoware_ml.models.segmentation3d.encoders.ptv3.load_flash_attn_module",
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

    class _EncoderForExport(torch.nn.Module):
        def __init__(self, attention_module: SerializedAttention) -> None:
            super().__init__()
            self.order = ["hilbert"]
            self.shuffle_orders = True
            self.attention = attention_module

        def set_serialization_order(self, order: tuple[str, ...]) -> None:
            self.order = list(order)

        def prepare_for_export(self, order: tuple[str, ...]) -> torch.nn.Module:
            return PointTransformerV3Encoder.prepare_for_export(self, order)

    encoder = _EncoderForExport(attention)

    with patch(
        "autoware_ml.models.segmentation3d.encoders.ptv3.replace_submconv3d_for_export",
        return_value=None,
    ):
        export_module = _PTv3SegmentationExportModule(
            encoder=encoder.prepare_for_export(("z", "z-trans")),
            seg3d_head=nn.Linear(4, 2),
            sparse_shape=torch.tensor([64, 64, 64], dtype=torch.long),
            serialized_depth=torch.tensor(6, dtype=torch.long),
        )

    export_attention = export_module.encoder.attention
    assert attention.enable_flash is True
    assert attention.patch_size == 4
    assert export_attention.enable_flash is False
    assert export_attention.flash_attn is None
    assert export_attention.patch_size == 0
    assert encoder.shuffle_orders is True
    assert export_module.encoder.shuffle_orders is False


def test_prepare_for_export_handles_loaded_flash_attention_module() -> None:
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
    attention.flash_attn = math

    class _EncoderForExport(torch.nn.Module):
        def __init__(self, attention_module: SerializedAttention) -> None:
            super().__init__()
            self.order = ["hilbert"]
            self.shuffle_orders = True
            self.attention = attention_module

        def set_serialization_order(self, order: tuple[str, ...]) -> None:
            self.order = list(order)

        def prepare_for_export(self, order: tuple[str, ...]) -> torch.nn.Module:
            return PointTransformerV3Encoder.prepare_for_export(self, order)

    encoder = _EncoderForExport(attention)

    with patch(
        "autoware_ml.models.segmentation3d.encoders.ptv3.replace_submconv3d_for_export",
        return_value=None,
    ):
        export_encoder = PointTransformerV3Encoder.prepare_for_export(encoder, ("z", "z-trans"))

    assert attention.flash_attn is math
    assert export_encoder.attention.flash_attn is None
    assert export_encoder.attention.enable_flash is False


def test_serialized_attention_export_mode_keeps_a_static_window_below_capacity() -> None:
    """A sample smaller than one window exports without shrinking the window."""
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
    attention.disable_flash()
    attention.export_mode = True
    point = Point(
        {
            "feat": torch.randn(3, 32),
            "grid_coord": torch.randint(0, 8, (3, 3), dtype=torch.int32),
            "serialized_order": torch.arange(3).reshape(1, 3),
            "serialized_inverse": torch.arange(3).reshape(1, 3),
            "offset": torch.tensor([3], dtype=torch.long),
        }
    )

    output = attention(point)

    assert output.feat.shape == (3, 32)
    # The window stays at its configured size; only the fill adapts.
    assert attention.patch_size == 4
    assert torch.isfinite(output.feat).all()


def test_serialized_attention_non_export_mode_adapts_patch_size() -> None:
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
    attention.disable_flash()
    point = Point(
        {
            "feat": torch.randn(3, 32),
            "grid_coord": torch.randint(0, 8, (3, 3), dtype=torch.int32),
            "serialized_order": torch.arange(3).reshape(1, 3),
            "serialized_inverse": torch.arange(3).reshape(1, 3),
            "offset": torch.tensor([3], dtype=torch.long),
        }
    )

    output = attention(point)

    assert output.feat.shape == (3, 32)
    assert attention.patch_size == 3


def test_point_sequential_skips_dense_module_on_empty_sparse_tensor() -> None:
    spconv = pytest.importorskip("spconv.pytorch")
    sparse_tensor = spconv.SparseConvTensor(
        features=torch.empty((0, 4), dtype=torch.float32),
        indices=torch.empty((0, 4), dtype=torch.int32),
        spatial_shape=[4, 4, 4],
        batch_size=1,
    )
    sequence = PointSequential(nn.BatchNorm1d(4))

    output = sequence(sparse_tensor)

    assert output is sparse_tensor
    assert output.features.shape == (0, 4)


def test_compute_metrics_reports_losses_and_point_level_accuracy() -> None:
    """compute_metrics should run losses on voxel logits and metrics at point level."""
    model = PTv3SegmentationModel.__new__(PTv3SegmentationModel)
    torch.nn.Module.__init__(model)
    model.seg3d_head = build_seg_head(num_classes=3, dec_depths=(0,))

    voxel_logits = torch.tensor(
        [
            [3.0, 0.1, 0.2],
            [0.2, 2.5, 0.1],
            [0.1, 0.3, 4.0],
        ],
        dtype=torch.float32,
    )
    segment = torch.tensor([0, 1, -1], dtype=torch.long)
    # Two source points: one maps to voxel 0, the other to voxel 1.
    inverse = torch.tensor([0, 1], dtype=torch.long)
    origin_segment = torch.tensor([0, 1], dtype=torch.long)

    origin_coord = torch.tensor([[10.0, 0.0, 0.0], [60.0, 0.0, 0.0]], dtype=torch.float32)
    batch = {
        "segment": segment,
        "inverse": inverse,
        "origin_segment": origin_segment,
        "origin_coord": origin_coord,
    }

    metrics = PTv3SegmentationModel.compute_metrics(model, batch, voxel_logits)

    # compute_metrics now returns only losses; quality metrics are produced at
    # epoch end from build_eval_output via the attached AutowareSegmentation3DMetrics.
    assert set(metrics) == {"loss", "loss_ce", "loss_lovasz"}
    assert metrics["loss"] > 0

    eval_out = PTv3SegmentationModel.build_eval_output(model, batch, voxel_logits)
    assert torch.equal(eval_out["seg_pred_labels"], torch.tensor([0, 1]))
    assert torch.equal(eval_out["seg_target_labels"], origin_segment)
    assert torch.equal(eval_out["seg_coord"], origin_coord)


def test_predict_outputs_reconstructs_point_level_predictions() -> None:
    """predict_outputs should scatter voxel logits to source points via inverse."""
    model = PTv3SegmentationModel.__new__(PTv3SegmentationModel)
    torch.nn.Module.__init__(model)

    voxel_logits = torch.tensor([[4.0, 0.1], [0.1, 5.0]], dtype=torch.float32)
    inverse = torch.tensor([0, 1, 0], dtype=torch.long)

    predictions = PTv3SegmentationModel.predict_outputs(
        model,
        {"inverse": inverse},
        voxel_logits,
    )

    assert torch.equal(predictions["pred_labels"], torch.tensor([0, 1, 0]))
    assert predictions["pred_probs"].shape == (3, 2)
    expected_probs = torch.softmax(voxel_logits, dim=1)[inverse]
    assert torch.allclose(predictions["pred_probs"], expected_probs)


def test_point_serialization_accepts_explicit_depth_override() -> None:
    point = Point(
        {
            "coord": torch.tensor([[0.0, 0.0, 0.0], [1.2, 1.0, 0.5]], dtype=torch.float32),
            "grid_coord": torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32),
            "feat": torch.randn(2, 4),
            "offset": torch.tensor([2], dtype=torch.long),
        }
    )

    point_cloud_range = torch.tensor([0.0, 0.0, -2.0, 8.0, 8.0, 2.0])
    axis_extents = (point_cloud_range[3:] - point_cloud_range[:3]) / 1.0
    explicit_depth = point_structures.bit_length_tensor(torch.max(axis_extents))
    point.serialization(("z", "z-trans"), shuffle_orders=False, depth=explicit_depth)

    assert point["serialized_depth"].item() == 4
    assert point["serialized_code"].shape == (2, 2)


def test_ptv3_encoder_dynamic_axes_follow_generated_pooling_inputs() -> None:
    input_names = [
        "grid_coord",
        "feat",
        "serialized_code",
        "serialized_pooling_0_indices",
        "serialized_pooling_0_indptr",
        "serialized_pooling_0_cluster",
        "serialized_pooling_0_head_indices",
        "serialized_pooling_0_grid_coord",
        "serialized_pooling_0_serialized_order",
        "serialized_pooling_0_serialized_inverse",
        "serialized_pooling_1_grid_coord",
    ]

    dynamic_axes = build_ptv3_encoder_dynamic_axes(input_names, stage_count=3)

    assert dynamic_axes["grid_coord"] == {0: "num_voxels"}
    assert dynamic_axes["feat"] == {0: "num_voxels"}
    assert dynamic_axes["serialized_code"] == {1: "num_voxels"}
    assert dynamic_axes["serialized_pooling_0_indices"] == {0: "serialized_pooling_0_in_voxels"}
    assert dynamic_axes["serialized_pooling_0_indptr"] == {
        0: "serialized_pooling_0_out_voxels_plus_one"
    }
    assert dynamic_axes["serialized_pooling_0_cluster"] == {0: "serialized_pooling_0_in_voxels"}
    assert dynamic_axes["serialized_pooling_0_head_indices"] == {
        0: "serialized_pooling_0_out_voxels"
    }
    assert dynamic_axes["serialized_pooling_0_grid_coord"] == {0: "serialized_pooling_0_out_voxels"}
    assert dynamic_axes["serialized_pooling_0_serialized_order"] == {
        1: "serialized_pooling_0_out_voxels"
    }
    assert dynamic_axes["serialized_pooling_0_serialized_inverse"] == {
        1: "serialized_pooling_0_out_voxels"
    }
    assert dynamic_axes["serialized_pooling_1_grid_coord"] == {0: "serialized_pooling_1_out_voxels"}
    assert dynamic_axes["point_feat_0"] == {0: "num_voxels"}
    assert dynamic_axes["point_feat_1"] == {0: "serialized_pooling_0_out_voxels"}
    assert dynamic_axes["point_feat_2"] == {0: "serialized_pooling_1_out_voxels"}


def test_validate_serialization_geometry_rejects_shallow_configs() -> None:
    """Configs whose serialization depth cannot cover all pooling stages should fail."""
    pooling_stages = nn.Sequential(
        *(SerializedPooling(8, 8, stride=2, shuffle_orders=False) for _ in range(3))
    )
    validate_serialization_geometry(pooling_stages, 1.0, (0.0, 0.0, 0.0, 8.0, 8.0, 8.0))
    with pytest.raises(ValueError, match="pooling depth 3"):
        validate_serialization_geometry(pooling_stages, 1.0, (0.0, 0.0, 0.0, 2.0, 2.0, 2.0))


@pytest.mark.parametrize("stride", [0, 3, 6])
def test_serialized_pooling_rejects_non_power_of_two_stride(stride: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        SerializedPooling(6, 8, stride=stride, shuffle_orders=False)


def test_serialized_pooling_export_mode_uses_precomputed_metadata(monkeypatch) -> None:
    """Export-mode pooling should match train-time grouping without in-graph Unique."""
    monkeypatch.setattr(Point, "sparsify", lambda self, pad=96: None)
    torch.manual_seed(0)

    grid_coord = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 2, 1],
            [3, 2, 1],
            [2, 3, 1],
            [3, 3, 1],
        ],
        dtype=torch.int32,
    )
    feat = torch.randn(grid_coord.shape[0], 6)

    def make_point() -> Point:
        point = Point(
            {
                "coord": grid_coord.to(torch.float32),
                "grid_coord": grid_coord,
                "feat": feat,
                "batch": torch.zeros(grid_coord.shape[0], dtype=torch.long),
                "offset": torch.tensor([grid_coord.shape[0]], dtype=torch.long),
                "sparse_shape": torch.tensor([16, 16, 16], dtype=torch.long),
            }
        )
        point.serialization(("z", "z-trans"), shuffle_orders=False, depth=torch.tensor(6))
        return point

    train_module = SerializedPooling(
        6,
        8,
        stride=2,
        shuffle_orders=False,
    )
    export_module = SerializedPooling(
        6,
        8,
        stride=2,
        shuffle_orders=False,
        export_stage_index=0,
    )
    train_module.norm = nn.Identity()
    train_module.act = nn.Identity()
    export_module.norm = nn.Identity()
    export_module.act = nn.Identity()
    export_module.export_mode = True
    export_module.load_state_dict(train_module.state_dict())

    train_out = train_module(make_point())
    export_point = make_point()
    meta, _ = build_serialized_pooling_meta(
        export_point.grid_coord,
        export_point.serialized_code,
        export_point.serialized_order,
        stride=2,
    )
    export_point["serialized_pooling"] = [meta]
    export_out = export_module(export_point)

    for key in (
        "feat",
        "grid_coord",
        "serialized_order",
        "serialized_inverse",
        "batch",
        "sparse_shape",
        "pooling_inverse",
    ):
        left = train_out[key]
        right = export_out[key]
        if left.dtype.is_floating_point:
            torch.testing.assert_close(left, right, msg=f"Mismatch for {key}")
        else:
            assert torch.equal(left, right), f"Mismatch for {key}"


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)
def test_ptv3_frozen_encoder_supports_decoder_block_backward() -> None:
    """Stage-4 regression: a frozen (eval) encoder caches spconv indice pairs
    without backward metadata; decoder blocks must not reuse them by key or
    their CPE backward fails with an empty-indices assert."""
    model = PTv3SegmentationModel(
        encoder=build_ptv3_encoder(),
        seg3d_head=build_seg_head(),
        freeze_encoder=True,
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
    ).cuda()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))

    logits = model(**batch)
    logits.sum().backward()

    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.seg3d_head.parameters())


def _build_attention(patch_size: int) -> SerializedAttention:
    """Return a non-flash attention module with a fixed window size."""
    attention = SerializedAttention(
        channels=32,
        num_heads=4,
        patch_size=patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    ).eval()
    # `__init__` zeroes `patch_size` when flash is off, and `forward` - which these
    # tests bypass to reach `_get_padding_and_inverse` - is what normally sets it.
    attention.patch_size = patch_size
    return attention


def _identity_point(num_points: int) -> Point:
    """Return a point whose serialization order is the identity, for readable indices."""
    order = torch.arange(num_points).unsqueeze(0)
    return Point(
        feat=torch.zeros(num_points, 32),
        grid_coord=torch.zeros(num_points, 3, dtype=torch.long),
        offset=torch.tensor([num_points]),
        serialized_order=order,
        serialized_inverse=order.clone(),
    )


@pytest.mark.parametrize("patch_size", [4, 16, 48])
def test_export_fill_reproduces_the_training_fill_above_one_window(patch_size: int) -> None:
    """The cyclic fill must equal the training fill wherever training pads at all.

    Training borrows the tail of the preceding window, making the last window a
    full-size sliding window over the final ``patch_size`` tokens. Export has to
    match that exactly, or every frame whose voxel count is not a multiple of the
    window size drifts from what the model was trained on.
    """
    attention = _build_attention(patch_size)
    for num_points in range(patch_size + 1, 4 * patch_size + 1):
        attention.export_mode = False
        training_pad, _, _ = attention._get_padding_and_inverse(_identity_point(num_points))
        attention.export_mode = True
        export_pad, _, _ = attention._get_padding_and_inverse(_identity_point(num_points))
        assert torch.equal(export_pad, training_pad), f"n={num_points}, K={patch_size}"


@pytest.mark.parametrize("patch_size", [4, 16, 48])
def test_export_fill_only_ever_indexes_real_tokens(patch_size: int) -> None:
    """No slot may fall outside ``[0, n)``: every padded slot holds a real token."""
    attention = _build_attention(patch_size)
    attention.export_mode = True
    for num_points in range(1, 4 * patch_size + 1):
        pad, unpad, _ = attention._get_padding_and_inverse(_identity_point(num_points))
        assert int(pad.min()) >= 0, f"n={num_points}"
        assert int(pad.max()) < num_points, f"n={num_points}"
        assert pad.numel() % patch_size == 0, f"n={num_points}"
        assert unpad.numel() == num_points


def test_export_fill_wraps_instead_of_repeating_one_token() -> None:
    """Below one window the fill cycles, so no single key dominates the softmax."""
    attention = _build_attention(16)
    attention.export_mode = True
    pad, _, _ = attention._get_padding_and_inverse(_identity_point(5))

    _, counts = torch.unique(pad, return_counts=True)
    # Cycling keeps every multiplicity within one of every other; repeating the
    # last token would put 12 copies of it against 1 of each of the others.
    assert int(counts.max()) - int(counts.min()) <= 1
    assert int(counts.max()) <= 4


@pytest.mark.parametrize("num_points", [2, 4, 8, 16])
def test_export_attention_is_exact_when_the_count_divides_the_window(num_points: int) -> None:
    """A uniform key multiplicity cancels in the softmax, so these cases are exact."""
    torch.manual_seed(0)
    patch_size = 16
    feat = torch.randn(num_points, 32)

    reference = _build_attention(num_points)
    reference.export_mode = False
    padded = _build_attention(patch_size)
    padded.export_mode = True
    padded.load_state_dict(reference.state_dict())

    def run(attention: SerializedAttention) -> torch.Tensor:
        point = _identity_point(num_points)
        point.feat = feat.clone()
        with torch.no_grad():
            return attention(point).feat

    assert torch.allclose(run(padded), run(reference), atol=1e-5)
