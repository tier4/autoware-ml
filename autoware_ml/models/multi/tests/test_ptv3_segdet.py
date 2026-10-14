"""Unit tests for the PTv3 joint segdet model's supervision masking and fusion."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.models.detection3d.ptv3 import PTv3DetFeatureFusion
from autoware_ml.models.detection3d.tests.head_output_fixtures import build_transfusion_outputs
from autoware_ml.models.detection3d.tests.ptv3_detection_fixtures import build_seg_head
from autoware_ml.models.multi.ptv3_segdet import PTv3SegDetModel
from autoware_ml.models.segmentation3d.heads.ptv3 import current_frame_mask
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_detection_gt_batch,
    build_point_cloud_batch,
)
from autoware_ml.utils.point_cloud.structures import Point


def _make_masking_model(recorded_calls: list) -> SimpleNamespace:
    """Build a duck-typed segdet model recording bbox_head.loss inputs."""

    def bbox_loss(
        outputs, gt_bboxes_3d, gt_labels_3d, gt_valid_bboxes, gt_traffic_cone_barrier_bbox_status
    ):
        det_outputs = outputs.transfusion_head_outputs
        recorded_calls.append((det_outputs, gt_bboxes_3d, gt_labels_3d))
        return {"loss": det_outputs.separate_head_outputs.heatmap.sum() * 0.0 + 1.0}

    def seg_loss(seg_logits, segment, weights):
        zero = seg_logits.sum() * 0.0
        return {"loss_ce": zero, "loss_lovasz": zero, "loss": zero}

    return SimpleNamespace(
        seg3d_head=SimpleNamespace(
            loss=seg_loss, ignore_index=-1, training=False, mixed_voxel_weight=0.0
        ),
        bbox_head=SimpleNamespace(loss=bbox_loss),
        segmentation_loss_weight=1.0,
        detection_loss_weight=1.0,
        _detection_gt_batch=PTv3SegDetModel._detection_gt_batch,
        _detection_frame_mask=PTv3SegDetModel._detection_frame_mask,
        _mask_list=PTv3SegDetModel._mask_list,
    )


def _make_inputs(has_boxes: list[bool]) -> ModelBatchInputs:
    """Detection supervision is carried by the ground truth itself: frames
    without boxes are detection-unsupervised. Every frame carries two points
    that map to their own voxels."""
    num_frames = len(has_boxes)
    num_points = 2 * num_frames
    point_cloud = build_point_cloud_batch(
        [torch.zeros((2, 5), dtype=torch.float32) for _ in range(num_frames)],
        timestamp_difference_dim=4,
    )
    detection = build_detection_gt_batch(
        [
            torch.full((1, 9), float(index)) if with_boxes else torch.zeros((0, 9))
            for index, with_boxes in enumerate(has_boxes)
        ],
        [
            torch.tensor([index]) if with_boxes else torch.zeros((0,), dtype=torch.long)
            for index, with_boxes in enumerate(has_boxes)
        ],
    )
    labels = Segmentation3DGTBatch(
        gt_semantic_masks=torch.tensor([0, 1] * num_frames, dtype=torch.int64),
        batch_indices=point_cloud.batch_indices,
    )
    voxels = VoxelsData(
        voxels=torch.zeros((num_points, 1, 5), dtype=torch.float32),
        coords=torch.zeros((num_points, 3), dtype=torch.int32),
        num_points=torch.ones(num_points, dtype=torch.int32),
        batch_indices=point_cloud.batch_indices,
        point_voxel_indices=torch.arange(num_points, dtype=torch.long),
        num_dropped_voxels=torch.tensor(0),
    )
    return build_batch_inputs(
        point_cloud=point_cloud, detection=detection, segmentation=labels, voxels=voxels
    )


def _make_outputs(batch_size: int) -> ModelOutputs:
    det_outputs = build_transfusion_outputs(
        {
            name: torch.randn(batch_size, channels, 8, requires_grad=True)
            for name, channels in (
                ("heatmap", 2),
                ("center", 2),
                ("height", 1),
                ("dim", 3),
                ("rot", 2),
            )
        }
    )
    return ModelOutputs(
        detection3d_head_outputs=Detection3DHeadOutputs(
            center_head_outputs=None, transfusion_head_outputs=det_outputs
        ),
        segmentation3d_head_outputs=Segmentation3DHeadOutputs(
            logits=torch.randn(4, 3, requires_grad=True)
        ),
    )


def test_compute_metrics_masks_detection_loss_to_frames_with_boxes() -> None:
    recorded_calls: list = []
    model = _make_masking_model(recorded_calls)
    batch = _make_inputs([True, False])
    outputs = _make_outputs(batch_size=2)

    metrics = PTv3SegDetModel.compute_metrics(model, batch, outputs)

    assert len(recorded_calls) == 1
    det_outputs, gt_boxes, gt_labels = recorded_calls[0]
    heatmap = det_outputs.separate_head_outputs.heatmap
    assert heatmap.shape[0] == 1
    reference = outputs.detection3d().transfusion_head().separate_head_outputs.heatmap
    assert torch.equal(heatmap[0], reference[0])
    assert len(gt_boxes) == 1 and float(gt_boxes[0][0, 0]) == 0.0
    assert len(gt_labels) == 1
    assert "det_loss" in metrics and "loss" in metrics


def test_compute_metrics_keeps_every_frame_when_all_have_boxes() -> None:
    recorded_calls: list = []
    model = _make_masking_model(recorded_calls)
    batch = _make_inputs([True, True])
    outputs = _make_outputs(batch_size=2)

    PTv3SegDetModel.compute_metrics(model, batch, outputs)

    det_outputs, gt_boxes, _ = recorded_calls[0]
    assert torch.equal(
        det_outputs.separate_head_outputs.heatmap,
        outputs.detection3d().transfusion_head().separate_head_outputs.heatmap,
    )
    assert len(gt_boxes) == 2


def test_compute_metrics_keeps_det_branch_in_graph_without_supervised_frames() -> None:
    recorded_calls: list = []
    model = _make_masking_model(recorded_calls)
    batch = _make_inputs([False, False])
    outputs = _make_outputs(batch_size=2)

    metrics = PTv3SegDetModel.compute_metrics(model, batch, outputs)

    assert not recorded_calls
    assert float(metrics["det_loss"].detach()) == 0.0
    # The zero detection loss must stay connected to the detection outputs so
    # DDP reducers see gradients (of zero) for every detection parameter.
    assert metrics["det_loss"].grad_fn is not None
    assert float(metrics["loss"].detach()) == 0.0


def _make_eval_model() -> SimpleNamespace:
    def decode_outputs(outputs):
        det_outputs = outputs.transfusion_head_outputs
        batch_size = det_outputs.separate_head_outputs.heatmap.shape[0]
        return ModelPredictions(
            detection3d_predictions=[
                Detection3DSamplePredictions(
                    bboxes_3d=torch.full((2, 9), float(index)),
                    scores_3d=torch.full((2,), float(index)),
                    labels_3d=torch.zeros(2, dtype=torch.long),
                )
                for index in range(batch_size)
            ]
        )

    return SimpleNamespace(
        bbox_head=SimpleNamespace(decode_outputs=decode_outputs),
        _detection_gt_batch=PTv3SegDetModel._detection_gt_batch,
        _detection_frame_mask=PTv3SegDetModel._detection_frame_mask,
    )


def test_build_eval_output_neutralizes_unflagged_frames_keeping_one_entry_per_frame() -> None:
    """Per-frame metric-state alignment: torchmetrics list-state sync issues one
    all_gather per entry, so every rank must contribute exactly one entry per
    frame regardless of its seg/det frame mix."""
    model = _make_eval_model()
    outputs = _make_outputs(batch_size=2)
    batch = _make_inputs([False, True])

    eval_out = PTv3SegDetModel.build_eval_output(model, batch, outputs)

    assert len(eval_out["seg_frames"]) == 2
    assert len(eval_out["predictions"]) == 2
    assert len(eval_out["gt_boxes"]) == 2
    # Unflagged frame: empty predictions with preserved trailing dims.
    assert eval_out["predictions"][0]["bboxes_3d"].shape == (0, 9)
    assert eval_out["predictions"][0]["scores_3d"].shape == (0,)
    # Flagged frame: predictions kept as-is.
    assert eval_out["predictions"][1]["bboxes_3d"].shape == (2, 9)
    assert float(eval_out["predictions"][1]["scores_3d"][0]) == 1.0


def test_build_eval_output_without_flagged_frames_keeps_neutral_entries() -> None:
    model = _make_eval_model()
    outputs = _make_outputs(batch_size=1)
    batch = _make_inputs([False])

    eval_out = PTv3SegDetModel.build_eval_output(model, batch, outputs)

    assert len(eval_out["seg_frames"]) == 1
    assert len(eval_out["predictions"]) == 1
    assert eval_out["predictions"][0]["bboxes_3d"].shape == (0, 9)
    assert len(eval_out["gt_boxes"]) == 1


def test_seg_head_loss_returns_connected_zero_when_all_targets_ignored() -> None:
    """A batch can carry zero seg supervision (e.g. seg-masked det-val frames);
    CE over zero valid targets is nan, so the head must short-circuit to a
    graph-connected zero."""
    head = build_seg_head(num_classes=3, dec_depths=(0,))
    logits = torch.randn(5, 3, requires_grad=True)
    all_ignored = torch.full((5,), -1, dtype=torch.long)

    metrics = head.loss(logits, all_ignored, torch.zeros(5))

    assert float(metrics["loss_ce"].detach()) == 0.0
    assert float(metrics["loss"].detach()) == 0.0
    assert metrics["loss"].grad_fn is not None
    # Sanity: valid targets still produce a real loss.
    assert torch.isfinite(head.loss(logits, torch.tensor([0, 1, 2, 0, 1]), torch.ones(5))["loss"])


def test_det_feature_fusion_reads_pooling_chain_non_destructively() -> None:
    torch.manual_seed(0)
    fusion = PTv3DetFeatureFusion(in_channels=16, skip_channels=8, out_channels=16).eval()
    parent_feat = torch.randn(6, 8)
    parent = Point(
        feat=parent_feat.clone(),
        grid_coord=torch.randint(0, 8, (6, 3)),
        offset=torch.tensor([6], dtype=torch.long),
    )
    deepest = Point(
        feat=torch.randn(3, 16),
        pooling_parent=parent,
        pooling_inverse=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
    )

    fused, grid_coord, offset = fusion(deepest)

    assert fused.shape == (6, 16)
    assert torch.equal(grid_coord, parent.grid_coord)
    assert torch.equal(offset, parent.offset)
    # Nothing popped, nothing mutated: the segmentation decoder still sees the
    # intact chain afterwards.
    assert "pooling_parent" in deepest
    assert "pooling_inverse" in deepest
    assert torch.equal(parent.feat, parent_feat)


def test_current_frame_mask_reads_the_time_lag_column() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1]])

    assert current_frame_mask(points).tolist() == [True, False]
