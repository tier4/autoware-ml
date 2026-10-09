"""Unit tests for PTv3 encoder components."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import autoware_ml.utils.point_cloud.structures as point_structures
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_point_cloud_batch,
)
from autoware_ml.models.detection3d.tests.ptv3_detection_fixtures import (
    build_inputs,
    build_points,
    build_preprocessor,
    build_ptv3_encoder,
    build_seg_head,
    build_seg_model,
)
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Point,
    PointSequential,
    PointTransformerV3Encoder,
    SerializedAttention,
    SerializedPooling,
    build_patch_order,
    build_serialized_pooling_meta,
    padded_patch_count,
)
from autoware_ml.models.segmentation3d.encoders.voxel import (
    PastFutureVoxelFeatureEncoder,
    SweepSplitVoxelFeatureEncoder,
)
from autoware_ml.models.segmentation3d.heads.ptv3 import (
    segmentation_eval_output,
    segmentation_point_loss,
    segmentation_predict_outputs,
    voxel_supervision,
)
from autoware_ml.models.segmentation3d.ptv3 import (
    PTv3SegmentationModel,
    _PTv3SegmentationExportModule,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    build_ptv3_encoder_dynamic_axes,
    build_ptv3_export_context,
    export_patch_sizes,
    prepare_ptv3_export_inputs,
    require_single_sample_export_batch,
    validate_serialization_geometry,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.base import DataPreprocessing


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
            voxel_encoder=SweepSplitVoxelFeatureEncoder(),
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
    serialized_order = torch.arange(3).reshape(1, 3)
    point = Point(
        {
            "feat": torch.randn(3, 32),
            "grid_coord": torch.randint(0, 8, (3, 3), dtype=torch.int32),
            "serialized_order": serialized_order,
            "serialized_inverse": torch.arange(3).reshape(1, 3),
            "patch_order": build_patch_order(serialized_order, patch_size=4),
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


def test_serialized_attention_export_padding_matches_batched_branch() -> None:
    """Export-mode padding wraps the preceding patch exactly like the batched branch."""
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
    attention.patch_size = attention.patch_size_max

    def make_point(num_points: int) -> Point:
        return Point(
            {
                "feat": torch.randn(num_points, 32),
                "offset": torch.tensor([num_points], dtype=torch.long),
            }
        )

    # Both a non-divisible count (padded last window) and a divisible one (no padding).
    for num_points in (10, 8):
        point = make_point(num_points)
        attention.export_mode = False
        batched_pad, batched_unpad, _ = attention._get_padding_and_inverse(point)
        attention.export_mode = True
        export_pad, export_unpad, _ = attention._get_padding_and_inverse(point)
        assert torch.equal(export_pad, batched_pad)
        assert torch.equal(export_unpad, batched_unpad)

    # Sequences shorter than one patch cannot wrap a full patch back, and the padded
    # indices must still stay within the real token range.
    attention.export_mode = True
    short_pad, _, _ = attention._get_padding_and_inverse(make_point(2))
    assert short_pad.min().item() >= 0
    assert short_pad.max().item() < 2


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


def _seg_processed(
    point_frames: list[torch.Tensor],
    point_voxel_indices: torch.Tensor,
    segment_frames: list[torch.Tensor] | None = None,
) -> ModelBatchInputs:
    """Build model inputs from per frame points and handmade voxelizer outputs."""
    num_voxels = int(point_voxel_indices.max()) + 1 if point_voxel_indices.numel() else 0
    point_cloud = build_point_cloud_batch(point_frames, timestamp_difference_dim=4)
    labels = (
        Segmentation3DGTBatch(
            gt_semantic_masks=torch.cat(list(segment_frames), dim=0),
            batch_indices=point_cloud.batch_indices,
        )
        if segment_frames is not None
        else None
    )
    voxels = VoxelsData(
        voxels=torch.zeros((num_voxels, 1, point_frames[0].shape[1])),
        coords=torch.zeros((num_voxels, 3), dtype=torch.int32),
        num_points=torch.ones(num_voxels, dtype=torch.int32),
        batch_indices=torch.zeros(num_voxels, dtype=torch.int32),
        point_voxel_indices=point_voxel_indices,
        num_dropped_voxels=torch.zeros((), dtype=torch.int64),
    )
    return build_batch_inputs(point_cloud=point_cloud, segmentation=labels, voxels=voxels)


def _split_voxels() -> tuple[torch.Tensor, torch.Tensor]:
    """Build five voxels covering every combination of current, past and future points."""
    voxels = torch.zeros(5, 4, 5)
    # Current and past points
    voxels[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.5, 0.0])
    voxels[0, 1] = torch.tensor([2.0, 0.0, 0.0, 0.7, 0.1])
    # Past points only
    voxels[1, 0] = torch.tensor([5.0, 0.0, 0.0, 0.2, 0.1])
    # Current points only
    voxels[2, 0] = torch.tensor([9.0, 1.0, 0.0, 0.4, 0.0])
    voxels[2, 1] = torch.tensor([9.2, 1.0, 0.0, 0.6, 0.0])
    # Current and future points
    voxels[3, 0] = torch.tensor([3.0, 2.0, 0.0, 0.3, 0.0])
    voxels[3, 1] = torch.tensor([4.0, 2.0, 0.0, 0.9, -0.1])
    # Future points only
    voxels[4, 0] = torch.tensor([7.0, 3.0, 0.0, 0.8, -0.2])
    return voxels, torch.tensor([2, 1, 2, 2, 1], dtype=torch.int32)


# Column layout of the encoder output: the voxel itself, then the past block and the future one
PAST_OFFSET = slice(5, 8)
PAST_INTENSITY = 8
PAST_SHARE = 9
PAST_LAG = 10
FUTURE_OFFSET = slice(11, 14)
FUTURE_INTENSITY = 14
FUTURE_SHARE = 15
FUTURE_LAG = 16


def test_voxel_encoder_describes_the_current_frame_and_the_past_points() -> None:
    """The current frame points give the position, the past points their offset from it."""
    voxels, num_points = _split_voxels()
    features = SweepSplitVoxelFeatureEncoder()(voxels[:3], num_points[:3])

    assert features.shape == (3, 11)
    assert torch.allclose(features[0, :5], torch.tensor([1.0, 0.0, 0.0, 0.5, 0.0]))
    assert torch.allclose(features[0, PAST_OFFSET], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(features[0, PAST_INTENSITY : PAST_LAG + 1], torch.tensor([0.7, 0.5, 0.1]))
    # Past points only: the voxel takes their position and the past share is 1.
    assert torch.allclose(features[1, :5], torch.tensor([5.0, 0.0, 0.0, 0.2, 0.1]))
    assert float(features[1, PAST_SHARE]) == 1.0
    assert torch.allclose(features[2, 5:], torch.zeros(6))


def test_past_future_voxel_encoder_extends_the_past_one_without_future_points() -> None:
    """Without future points the 17 channel features start with the 11 channel ones."""
    voxels, num_points = _split_voxels()
    past = SweepSplitVoxelFeatureEncoder()(voxels[:3], num_points[:3])
    past_future = PastFutureVoxelFeatureEncoder()(voxels[:3], num_points[:3])

    assert torch.equal(past_future[:, :11], past)
    assert torch.equal(past_future[:, 11:], torch.zeros(3, 6))


def test_past_future_voxel_encoder_places_the_voxel_on_its_current_frame_points() -> None:
    """The current frame points give the position, the sweep points their offset from it."""
    features = PastFutureVoxelFeatureEncoder()(*_split_voxels())

    assert features.shape == (5, 17)
    # Current and past points: the current point positions the voxel.
    assert torch.allclose(features[0, :5], torch.tensor([1.0, 0.0, 0.0, 0.5, 0.0]))
    assert torch.allclose(features[0, PAST_OFFSET], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(features[0, PAST_INTENSITY : PAST_LAG + 1], torch.tensor([0.7, 0.5, 0.1]))
    # Nothing was measured after the sample, so the future block stays empty.
    assert torch.allclose(features[0, FUTURE_OFFSET.start :], torch.zeros(6))
    # Current points only: nothing to report about either side.
    assert torch.allclose(features[2, :5], torch.tensor([9.1, 1.0, 0.0, 0.5, 0.0]))
    assert torch.allclose(features[2, 5:], torch.zeros(12))


def test_past_future_voxel_encoder_keeps_the_past_and_the_future_points_apart() -> None:
    """A surface crossing the voxel leaves the two sides in opposite directions.

    One mean over both sides would cancel the displacement, so each side reports its own
    offset, intensity, share and lag.
    """
    features = PastFutureVoxelFeatureEncoder()(*_split_voxels())

    # Current and future points: the future point sits ahead of the voxel position.
    assert torch.allclose(features[3, :5], torch.tensor([3.0, 2.0, 0.0, 0.3, 0.0]))
    assert torch.allclose(features[3, PAST_OFFSET.start : PAST_LAG + 1], torch.zeros(6))
    assert torch.allclose(features[3, FUTURE_OFFSET], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(
        features[3, FUTURE_INTENSITY : FUTURE_LAG + 1], torch.tensor([0.9, 0.5, -0.1])
    )


def test_past_future_voxel_encoder_falls_back_to_the_sweep_points_and_reports_their_lag() -> None:
    """A voxel with no current frame points takes the position and lag of its sweep points."""
    features = PastFutureVoxelFeatureEncoder()(*_split_voxels())

    # Past points only: the voxel takes their position and the past share is 1.
    assert torch.allclose(features[1, :5], torch.tensor([5.0, 0.0, 0.0, 0.2, 0.1]))
    assert torch.allclose(features[1, PAST_OFFSET], torch.zeros(3))
    assert float(features[1, PAST_SHARE]) == 1.0
    assert float(features[1, FUTURE_SHARE]) == 0.0
    # Future points only: the same, on the other side.
    assert torch.allclose(features[4, :5], torch.tensor([7.0, 3.0, 0.0, 0.8, -0.2]))
    assert torch.allclose(features[4, FUTURE_OFFSET], torch.zeros(3))
    assert float(features[4, FUTURE_SHARE]) == 1.0
    assert float(features[4, PAST_SHARE]) == 0.0


def test_past_future_voxel_encoder_keeps_the_average_over_every_point_recoverable() -> None:
    """Splitting the points by side keeps all the information of the voxel."""
    voxels, num_points = _split_voxels()
    features = PastFutureVoxelFeatureEncoder()(voxels, num_points)

    filled = torch.arange(voxels.shape[1]).unsqueeze(0) < num_points.long().unsqueeze(1)
    expected = (voxels * filled.unsqueeze(-1)).sum(dim=1) / num_points.unsqueeze(1)
    recovered = (
        features[:, :3]
        + features[:, PAST_SHARE : PAST_SHARE + 1] * features[:, PAST_OFFSET]
        + features[:, FUTURE_SHARE : FUTURE_SHARE + 1] * features[:, FUTURE_OFFSET]
    )
    assert torch.allclose(recovered, expected[:, :3])


def test_past_future_voxel_encoder_ignores_the_padded_slots() -> None:
    """Zero padding is not read as a current frame point."""
    voxels, num_points = _split_voxels()
    padded = PastFutureVoxelFeatureEncoder()(voxels, num_points)
    voxels[1, 1] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    assert torch.allclose(PastFutureVoxelFeatureEncoder()(voxels, num_points), padded)


def test_voxel_encoder_rejects_a_point_layout_without_the_time_lag() -> None:
    with pytest.raises(ValueError, match="time_lag"):
        SweepSplitVoxelFeatureEncoder()(
            torch.zeros(2, 3, 4), torch.tensor([1, 1], dtype=torch.int32)
        )
    with pytest.raises(ValueError, match="time_lag"):
        PastFutureVoxelFeatureEncoder()(
            torch.zeros(2, 3, 4), torch.tensor([1, 1], dtype=torch.int32)
        )


def test_model_rejects_future_points_its_voxel_encoder_does_not_read() -> None:
    voxels, num_points = _split_voxels()
    model = build_seg_model()

    with pytest.raises(ValueError, match="future"):
        model.encode(
            voxels=voxels,
            num_points=num_points,
            voxel_coords=torch.zeros(5, 4, dtype=torch.int32),
            num_dropped_voxels=torch.tensor(0),
            time_lag_column=4,
        )


def test_point_loss_and_eval_output_work_at_the_point_level() -> None:
    """The loss runs on the voxel labels and eval keeps the current-frame points only."""
    head = build_seg_head(num_classes=3, dec_depths=(0,))

    voxel_logits = torch.tensor(
        [
            [3.0, 0.1, 0.2],
            [0.2, 2.5, 0.1],
            [0.1, 0.3, 4.0],
        ],
        dtype=torch.float32,
    )
    # Three points in two voxels, the last point comes from an earlier sweep.
    points = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.5, 0.0],
            [60.0, 0.0, 0.0, 0.5, 0.0],
            [60.5, 0.0, 0.0, 0.5, 0.1],
        ],
        dtype=torch.float32,
    )
    batch = _seg_processed(
        [points],
        torch.tensor([0, 1, 1], dtype=torch.long),
        segment_frames=[torch.tensor([0, 1, -1], dtype=torch.long)],
    )

    metrics = segmentation_point_loss(head, voxel_logits, batch)

    assert set(metrics) == {"loss", "loss_ce", "loss_lovasz"}
    assert metrics["loss"] > 0

    eval_out = segmentation_eval_output(voxel_logits, batch)
    (frame,) = eval_out["seg_frames"]
    assert torch.equal(frame["pred"], torch.tensor([0, 1]))
    assert torch.equal(frame["target"], torch.tensor([0, 1]))
    assert torch.equal(frame["coord"], points[:2, :3])
    assert frame["scores"].shape == (2, 3)


def test_voxel_supervision_takes_the_majority_of_the_voxel_outside_training() -> None:
    """The label that scores the most points of the voxel supervises it."""
    labels, _ = voxel_supervision(
        torch.tensor([0, 0, 0, 1], dtype=torch.long),
        torch.tensor([2, 2, 1, 1], dtype=torch.long),
        num_voxels=2,
        num_classes=3,
        ignore_index=-1,
        sample=False,
        mixed_weight=0.0,
    )

    assert torch.equal(labels, torch.tensor([2, 1]))


@pytest.mark.parametrize("ignore_index", [-1, 255])
def test_voxel_supervision_leaves_ignored_points_out_whatever_the_ignore_index(
    ignore_index: int,
) -> None:
    """Ignored points do not index the class table, and a voxel of ignored points only is
    unsupervised."""
    labels, weights = voxel_supervision(
        torch.tensor([0, 0, 1, 1, -1], dtype=torch.long),
        torch.tensor([2, ignore_index, ignore_index, ignore_index, 1], dtype=torch.long),
        num_voxels=2,
        num_classes=3,
        ignore_index=ignore_index,
        sample=False,
        mixed_weight=0.0,
    )

    assert labels.tolist() == [2, ignore_index]
    assert float(weights[1]) == 0.0


def test_voxel_supervision_weighs_a_voxel_by_how_much_its_points_disagree() -> None:
    """One label describes a voxel worse the more of its points it leaves out."""
    _, weights = voxel_supervision(
        torch.tensor([0, 0, 0, 1, 2, 2], dtype=torch.long),
        torch.tensor([2, 2, 1, 1, 0, 1], dtype=torch.long),
        num_voxels=3,
        num_classes=3,
        ignore_index=-1,
        sample=False,
        mixed_weight=1.0,
    )

    # Two of three points share the label, one of one, one of two.
    assert torch.allclose(weights, torch.tensor([1 + 1 / 3, 1.0, 1.5]))


def test_voxel_supervision_leaves_unsupervised_voxels_weightless() -> None:
    """A voxel holding only ignored points takes no part in the loss."""
    labels, weights = voxel_supervision(
        torch.tensor([0, 1, 1], dtype=torch.long),
        torch.tensor([-1, -1, 2], dtype=torch.long),
        num_voxels=3,
        num_classes=3,
        ignore_index=-1,
        sample=False,
        mixed_weight=1.0,
    )

    assert torch.equal(labels, torch.tensor([-1, 2, -1]))
    assert torch.equal(weights, torch.tensor([0.0, 1.0, 0.0]))


def test_voxel_supervision_draws_from_the_voxel_distribution_in_training() -> None:
    """Sampling supervises a voxel in proportion to how its points are labelled."""
    point_voxel_indices = torch.zeros(4, dtype=torch.long)
    segment = torch.tensor([0, 0, 0, 1], dtype=torch.long)

    torch.manual_seed(0)
    drawn = [
        int(
            voxel_supervision(
                point_voxel_indices,
                segment,
                num_voxels=1,
                num_classes=2,
                ignore_index=-1,
                sample=True,
                mixed_weight=0.0,
            )[0][0]
        )
        for _ in range(400)
    ]

    assert set(drawn) == {0, 1}
    assert 0.6 < drawn.count(0) / len(drawn) < 0.9


def test_points_of_a_voxel_do_not_weight_the_loss() -> None:
    """Repeating a label inside a voxel leaves the loss unchanged."""
    head = build_seg_head(num_classes=3, dec_depths=(0,)).eval()
    voxel_logits = torch.tensor([[3.0, 0.1, 0.2], [0.2, 2.5, 0.1]], dtype=torch.float32)
    dense = _seg_processed(
        [torch.zeros((5, 5), dtype=torch.float32)],
        torch.tensor([0, 0, 0, 0, 1], dtype=torch.long),
        segment_frames=[torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)],
    )
    sparse = _seg_processed(
        [torch.zeros((2, 5), dtype=torch.float32)],
        torch.tensor([0, 1], dtype=torch.long),
        segment_frames=[torch.tensor([0, 1], dtype=torch.long)],
    )

    assert torch.allclose(
        segmentation_point_loss(head, voxel_logits, dense)["loss"],
        segmentation_point_loss(head, voxel_logits, sparse)["loss"],
    )


def test_encode_rejects_dropped_voxels_and_a_misplaced_time_lag() -> None:
    model = build_seg_model()
    voxels = torch.zeros((1, 1, 5))
    num_points = torch.ones(1, dtype=torch.int32)
    voxel_coords = torch.zeros((1, 4), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="max_voxels"):
        model.encode(voxels, num_points, voxel_coords, torch.tensor(3), 4)
    with pytest.raises(ValueError, match="time lag"):
        model.encode(voxels, num_points, voxel_coords, torch.tensor(0), -1)


def test_points_outside_the_voxel_grid_are_excluded_from_loss_and_eval() -> None:
    """A point without a voxel (outside the grid) neither supervises nor gets scored."""
    head = build_seg_head(num_classes=3, dec_depths=(0,))
    voxel_logits = torch.tensor([[3.0, 0.1, 0.2], [0.2, 2.5, 0.1]], dtype=torch.float32)
    points = torch.zeros((3, 5), dtype=torch.float32)
    batch = _seg_processed(
        [points],
        torch.tensor([0, -1, 1], dtype=torch.long),
        segment_frames=[torch.tensor([0, 2, 1], dtype=torch.long)],
    )
    reference = _seg_processed(
        [points[[0, 2]]],
        torch.tensor([0, 1], dtype=torch.long),
        segment_frames=[torch.tensor([0, 1], dtype=torch.long)],
    )

    metrics = segmentation_point_loss(head, voxel_logits, batch)
    expected = segmentation_point_loss(head, voxel_logits, reference)
    assert torch.allclose(metrics["loss"], expected["loss"])

    (frame,) = segmentation_eval_output(voxel_logits, batch)["seg_frames"]
    assert torch.equal(frame["pred"], torch.tensor([0, 1]))
    assert torch.equal(frame["target"], torch.tensor([0, 1]))


def test_predict_outputs_reconstructs_current_frame_point_predictions() -> None:
    """predict_outputs gathers the voxel logits of the current frame points only."""
    voxel_logits = torch.tensor([[4.0, 0.1], [0.1, 5.0]], dtype=torch.float32)
    point_voxel_indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    points = torch.zeros((4, 5), dtype=torch.float32)
    # the last point comes from an earlier sweep and gets no prediction
    points[3, 4] = 0.1

    predictions = segmentation_predict_outputs(
        voxel_logits,
        _seg_processed([points[:2], points[2:]], point_voxel_indices),
    )

    assert torch.equal(predictions.pred_labels, torch.tensor([0, 1, 0]))
    assert predictions.pred_probs.shape == (3, 2)
    expected_probs = torch.softmax(voxel_logits, dim=1)[point_voxel_indices[:3]]
    assert torch.allclose(predictions.pred_probs, expected_probs)


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
        "voxels",
        "num_points_per_voxel",
        "grid_coord",
        "patch_order",
        "serialized_inverse",
        "serialized_pooling_0_patch_order",
        "serialized_pooling_0_indices",
        "serialized_pooling_0_indptr",
        "serialized_pooling_0_cluster",
        "serialized_pooling_0_head_indices",
        "serialized_pooling_0_grid_coord",
        "serialized_pooling_0_serialized_inverse",
        "serialized_pooling_1_grid_coord",
    ]

    dynamic_axes = build_ptv3_encoder_dynamic_axes(input_names, stage_count=3)

    assert dynamic_axes["voxels"] == {0: "num_voxels"}
    assert dynamic_axes["num_points_per_voxel"] == {0: "num_voxels"}
    assert dynamic_axes["grid_coord"] == {0: "num_voxels"}
    assert dynamic_axes["serialized_inverse"] == {1: "num_voxels"}
    # Padded to whole attention windows, so its extent is its own axis, not the voxel count.
    assert dynamic_axes["patch_order"] == {1: "padded_voxels"}
    assert dynamic_axes["serialized_pooling_0_patch_order"] == {
        1: "serialized_pooling_0_padded_voxels"
    }
    assert dynamic_axes["serialized_pooling_0_indices"] == {0: "serialized_pooling_0_in_voxels"}
    assert dynamic_axes["serialized_pooling_0_indptr"] == {
        0: "serialized_pooling_0_out_voxels_plus_one"
    }
    assert dynamic_axes["serialized_pooling_0_cluster"] == {0: "serialized_pooling_0_in_voxels"}
    assert dynamic_axes["serialized_pooling_0_head_indices"] == {
        0: "serialized_pooling_0_out_voxels"
    }
    assert dynamic_axes["serialized_pooling_0_grid_coord"] == {0: "serialized_pooling_0_out_voxels"}
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
    meta, _, _ = build_serialized_pooling_meta(
        export_point.grid_coord,
        export_point.serialized_code,
        export_point.serialized_order,
        stride=2,
        patch_size=4,
    )
    export_point["serialized_pooling"] = [meta]
    export_out = export_module(export_point)

    for key in (
        "feat",
        "grid_coord",
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
    # The pooled level hands its blocks the precomputed gather order instead of the bare order.
    assert torch.equal(
        export_out["patch_order"], build_patch_order(train_out["serialized_order"], 4)
    )
    assert "serialized_order" not in export_out
    assert "serialized_code" not in export_out


def _reference_patch_order(serialized_order: torch.Tensor, patch_size: int) -> torch.Tensor:
    """The formula the exported attention block used to trace, kept as the oracle."""
    n = serialized_order.shape[1]
    padded_n = ((n + patch_size - 1) // patch_size) * patch_size
    divisor = max(n, 1)
    cycle = ((patch_size + divisor - 1) // divisor) * divisor
    index = torch.arange(padded_n)
    pad = torch.where(index < n, index, (index - patch_size + cycle) % divisor)
    return serialized_order[:, pad]


@pytest.mark.parametrize(
    ("count", "patch_size"),
    [
        (3, 4),  # below one window: wraps around
        (4, 4),  # exactly one window: no padding
        (8, 4),  # whole windows: no padding
        (9, 4),  # one token into the last window: borrows three backwards
        (13, 4),
        (5, 512),  # deployment window, tiny sample: wraps many times
        (1, 4),
    ],
)
def test_build_patch_order_matches_the_traced_formula(count: int, patch_size: int) -> None:
    """The precomputed gather order is exactly what the graph used to compute per block.

    It is also the contract the deployed runtime reimplements, so the reference stays here
    as the oracle for both sides. Every slot must hold a real token (no mask in attention).
    """
    torch.manual_seed(count)
    serialized_order = torch.stack([torch.randperm(count), torch.randperm(count)])

    patch_order = build_patch_order(serialized_order, patch_size)

    assert torch.equal(patch_order, _reference_patch_order(serialized_order, patch_size))
    assert patch_order.shape == (2, padded_patch_count(count, patch_size))
    assert patch_order.shape[1] % patch_size == 0
    assert torch.equal(patch_order[:, :count], serialized_order)
    assert (patch_order >= 0).all() and (patch_order < count).all()


#: Literal `patch_order` vectors, hand-derived from the contract, shared with the deployed
#: runtime's test (autoware.universe, autoware_ptv3 `serialized_pooling_metadata_test.cpp`):
#: two implementations of one formula in two languages must agree on the same numbers, not
#: each on its own reimplementation of the formula. (patch_size, serialized_order, expected).
PATCH_ORDER_GOLDEN_VECTORS = [
    # below one window: the tail wraps around (cycle = 6 for count 3)
    (4, [[0, 1, 2]], [[0, 1, 2, 2]]),
    (4, [[2, 0, 1]], [[2, 0, 1, 1]]),
    # exactly one window and whole windows: nothing to pad
    (4, [[3, 0, 2, 1]], [[3, 0, 2, 1]]),
    (4, [[7, 0, 6, 1, 5, 2, 4, 3]], [[7, 0, 6, 1, 5, 2, 4, 3]]),
    # one token into the last window (k * K + 1): borrows three backwards
    (4, [[4, 0, 3, 1, 2]], [[4, 0, 3, 1, 2, 0, 3, 1]]),
    # a partial last window: borrows the two before it
    (4, [[5, 2, 7, 1, 0, 3]], [[5, 2, 7, 1, 0, 3, 7, 1]]),
    # two distinct rows are padded independently, from their own entries
    (
        4,
        [[0, 5, 2, 1, 4, 3], [3, 1, 5, 0, 2, 4]],
        [[0, 5, 2, 1, 4, 3, 2, 1], [3, 1, 5, 0, 2, 4, 5, 0]],
    ),
    # the deployed window on a tiny sample: 507 tail slots cycle through the 5 tokens
    (512, [[4, 0, 3, 1, 2]], [[4, 0, 3, 1, 2] + [1, 2, 4, 0, 3] * 101 + [1, 2]]),
]


@pytest.mark.parametrize(("patch_size", "serialized_order", "expected"), PATCH_ORDER_GOLDEN_VECTORS)
def test_build_patch_order_golden_vectors(
    patch_size: int, serialized_order: list[list[int]], expected: list[list[int]]
) -> None:
    """The cross-language contract: fixed inputs, fixed outputs, no formula in the assertion."""
    patch_order = build_patch_order(torch.tensor(serialized_order), patch_size)
    assert patch_order.tolist() == expected


def test_build_patch_order_of_an_empty_level_is_empty() -> None:
    assert build_patch_order(torch.empty(2, 0, dtype=torch.long), 4).shape == (2, 0)


def test_export_patch_sizes_rejects_an_encoder_decoder_mismatch() -> None:
    model = build_seg_model().eval()
    assert export_patch_sizes(model) == list(model.encoder.enc_patch_size)
    model.seg3d_head.dec_patch_size[0] += 1
    with pytest.raises(ValueError, match="disagree"):
        export_patch_sizes(model)


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
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        seg3d_head=build_seg_head(),
        freeze_encoder=True,
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
    ).cuda()
    batch = build_inputs(device=torch.device("cuda"))

    logits = model(**model.forward_inputs(batch)).segmentation3d().logits
    logits.sum().backward()
    assert logits.shape == (batch.voxels_data.voxels.shape[0], 3)

    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.seg3d_head.parameters())


def _build_attention(patch_size: int, order_index: int = 0) -> SerializedAttention:
    """Return a non-flash attention module with a fixed window size."""
    attention = SerializedAttention(
        channels=32,
        num_heads=4,
        patch_size=patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=order_index,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    ).eval()
    # `__init__` zeroes `patch_size` when flash is off, and `forward` - which the fill tests
    # bypass to reach `_get_padding_and_inverse` - is what normally sets it.
    attention.patch_size = patch_size
    return attention


def _identity_point(num_points: int, patch_size: int | None = None) -> Point:
    """Return a point whose serialization order is the identity, for readable indices.

    With ``patch_size`` the point also carries the precomputed ``patch_order`` an
    export-mode block reads (the serialize stage's job in deployment).
    """
    order = torch.arange(num_points).unsqueeze(0)
    point = Point(
        feat=torch.zeros(num_points, 32),
        grid_coord=torch.zeros(num_points, 3, dtype=torch.long),
        offset=torch.tensor([num_points]),
        serialized_order=order,
        serialized_inverse=order.clone(),
    )
    if patch_size is not None:
        point["patch_order"] = build_patch_order(order, patch_size)
    return point


def _export_fill(num_points: int, patch_size: int) -> torch.Tensor:
    """The exported gather order over an identity serialization: the fill itself."""
    return build_patch_order(torch.arange(num_points).unsqueeze(0), patch_size)[0]


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
        training_pad, _, _ = attention._get_padding_and_inverse(_identity_point(num_points))
        export_pad = _export_fill(num_points, patch_size)
        assert torch.equal(export_pad, training_pad), f"n={num_points}, K={patch_size}"


@pytest.mark.parametrize("patch_size", [4, 16, 48])
def test_export_fill_only_ever_indexes_real_tokens(patch_size: int) -> None:
    """No slot may fall outside ``[0, n)``: every padded slot holds a real token."""
    for num_points in range(1, 4 * patch_size + 1):
        pad = _export_fill(num_points, patch_size)
        assert int(pad.min()) >= 0, f"n={num_points}"
        assert int(pad.max()) < num_points, f"n={num_points}"
        assert pad.numel() % patch_size == 0, f"n={num_points}"


def test_export_fill_wraps_instead_of_repeating_one_token() -> None:
    """Below one window the fill cycles, so no single key dominates the softmax."""
    pad = _export_fill(5, 16)

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
        point = _identity_point(num_points, patch_size if attention.export_mode else None)
        point.feat = feat.clone()
        with torch.no_grad():
            return attention(point).feat

    assert torch.allclose(run(padded), run(reference), atol=1e-5)


@pytest.mark.parametrize("num_points", [6, 9])
def test_export_attention_matches_training_on_a_non_identity_second_order(
    num_points: int,
) -> None:
    """The export wiring end to end: `patch_order[order_index]` with `serialized_inverse[order_index]`.

    Two distinct non-identity serialization rows, `order_index=1`, and a count that does not
    divide the window, so a row mix-up or a drift in the precomputed padding changes the
    output (the helper tests cannot see either). The training path is the oracle; it takes
    its window from the smallest sample of the batch, so the sample under test rides along
    with a window-sized filler sample that keeps the training window at ``patch_size``.
    """
    torch.manual_seed(1)
    patch_size = 4
    feat = torch.randn(num_points, 32)
    orders = torch.stack([torch.randperm(num_points), torch.randperm(num_points)])
    assert not torch.equal(orders[0], orders[1])

    export = _build_attention(patch_size, order_index=1)
    export.export_mode = True
    reference = _build_attention(patch_size, order_index=1)
    reference.load_state_dict(export.state_dict())

    export_point = Point(
        feat=feat.clone(),
        grid_coord=torch.zeros(num_points, 3, dtype=torch.long),
        offset=torch.tensor([num_points]),
        serialized_order=orders,
        serialized_inverse=torch.argsort(orders, dim=1),
        patch_order=build_patch_order(orders, patch_size),
    )
    filler_orders = torch.stack([torch.randperm(patch_size), torch.randperm(patch_size)])
    batch_orders = torch.cat([filler_orders, orders + patch_size], dim=1)
    training_point = Point(
        feat=torch.cat([torch.randn(patch_size, 32), feat]),
        grid_coord=torch.zeros(patch_size + num_points, 3, dtype=torch.long),
        offset=torch.tensor([patch_size, patch_size + num_points]),
        serialized_order=batch_orders,
        serialized_inverse=torch.argsort(batch_orders, dim=1),
    )

    with torch.no_grad():
        export_feat = export(export_point).feat
        training_feat = reference(training_point).feat[patch_size:]
    assert reference.patch_size == patch_size
    torch.testing.assert_close(export_feat, training_feat, atol=1e-5, rtol=1e-5)


def test_export_input_builders_reject_a_multi_sample_batch() -> None:
    """The graphs pad and offset one frame; a second sample fails here, not by attending across."""
    model = build_seg_model().eval()
    single = model.forward_inputs(build_inputs())
    require_single_sample_export_batch(
        model.build_encoder_inputs(single["voxels"], single["num_points"], single["voxel_coords"])
    )

    points = build_points()
    double = DataPreprocessing([build_preprocessor()])(
        build_batch_inputs(
            point_cloud=build_point_cloud_batch([points, points], timestamp_difference_dim=4)
        ).multi_task_gt_batch,
        is_training=True,
    )
    for builder in (prepare_ptv3_export_inputs, build_ptv3_export_context):
        with pytest.raises(ValueError, match="single-sample"):
            builder(model, double)
