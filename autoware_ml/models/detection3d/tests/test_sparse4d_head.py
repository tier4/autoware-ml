# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Sparse4D query detection head."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.models.detection3d.heads.sparse4d import (
    ANCHOR_DIM,
    InstanceBank,
    InstanceTracker,
    Sparse4DHead,
    SparseBox3DKeyPointsGenerator,
    SparseDeformableAggregation,
    SparseLevelGrid,
    decode_anchor_space_to_boxes,
    encode_boxes_to_anchor_space,
    encode_ground_truth,
)
from autoware_ml.models.detection3d.task_modules.nms_groups import (
    NMSGroupConfig,
    resolve_nms_groups,
)
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
    random_stage,
)

POINT_CLOUD_RANGE = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]
NUM_ANCHOR = 16
EMBED_DIMS = 32
FEATURE_STRIDE = 2


@pytest.fixture()
def head(tmp_path) -> Sparse4DHead:
    """Build a small Sparse4D head with a synthetic anchor file."""
    rng = np.random.default_rng(0)
    anchors = np.zeros((NUM_ANCHOR, ANCHOR_DIM), dtype=np.float32)
    anchors[:, :2] = rng.uniform(-40.0, 40.0, size=(NUM_ANCHOR, 2))
    anchors[:, 2] = rng.uniform(-1.0, 2.0, size=NUM_ANCHOR)
    anchors[:, 3:6] = np.log(rng.uniform(0.5, 5.0, size=(NUM_ANCHOR, 3)))
    anchors[:, 7] = 1.0
    anchor_file = tmp_path / "anchors.npy"
    np.save(anchor_file, anchors)
    return Sparse4DHead(
        num_classes=3,
        embed_dims=EMBED_DIMS,
        anchor_file=str(anchor_file),
        num_anchor=NUM_ANCHOR,
        point_cloud_range=POINT_CLOUD_RANGE,
        grid_size=1.0,
        feature_strides=(FEATURE_STRIDE,),
        num_decoder=2,
        num_heads=4,
        num_groups=4,
        feedforward_channels=64,
        num_learnable_pts=2,
        num_output=8,
    )


def _sparse_stage_inputs(
    batch_size: int, num_voxels: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a random sparse stage covering the fixture's pooled grid."""
    generator = torch.Generator().manual_seed(0)
    dims = (51, 51, 4)  # ceil(round(102.4 / 1.0) / 2), same for y; ceil(8 / 2) for z
    feat = torch.randn((batch_size * num_voxels, EMBED_DIMS), generator=generator)
    grid_coord = torch.stack(
        [
            torch.randint(0, dims[0], (batch_size * num_voxels,), generator=generator),
            torch.randint(0, dims[1], (batch_size * num_voxels,), generator=generator),
            torch.randint(0, dims[2], (batch_size * num_voxels,), generator=generator),
        ],
        dim=1,
    )
    offset = torch.arange(1, batch_size + 1, dtype=torch.long) * num_voxels
    return feat, grid_coord, offset


def test_encode_decode_roundtrip() -> None:
    boxes = torch.tensor(
        [
            [5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 1.0, -0.5],
            [-10.0, 8.0, 1.0, 12.0, 2.9, 3.4, -2.5, 0.0, 0.0],
        ]
    )
    decoded = decode_anchor_space_to_boxes(encode_boxes_to_anchor_space(boxes))
    assert torch.allclose(decoded[:, :6], boxes[:, :6], atol=1e-5)
    assert torch.allclose(decoded[:, 7:], boxes[:, 7:], atol=1e-5)
    yaw_delta = torch.remainder(decoded[:, 6] - boxes[:, 6] + torch.pi, 2 * torch.pi) - torch.pi
    assert torch.allclose(yaw_delta, torch.zeros_like(yaw_delta), atol=1e-5)


def test_keypoints_follow_anchor_yaw() -> None:
    generator = SparseBox3DKeyPointsGenerator(
        embed_dims=EMBED_DIMS,
        fix_scale=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        num_learnable_pts=0,
    )
    anchor = torch.zeros((1, 1, ANCHOR_DIM))
    anchor[..., 0] = 10.0  # x
    anchor[..., 3:6] = torch.log(torch.tensor([4.0, 2.0, 1.5]))
    anchor[..., 6] = 1.0  # sin yaw (90 degrees)
    anchor[..., 7] = 0.0  # cos yaw
    key_points = generator(anchor, None)
    center, front = key_points[0, 0]
    assert torch.allclose(center, torch.tensor([10.0, 0.0, 0.0]), atol=1e-5)
    # +x half-length offset rotated by 90 degrees points along +y.
    assert torch.allclose(front, torch.tensor([10.0, 2.0, 0.0]), atol=1e-5)


def test_assignment_matches_exact_prediction(head: Sparse4DHead) -> None:
    gt_boxes = torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])
    ground_truth = encode_ground_truth(
        gt_boxes, torch.tensor([1]), torch.device("cpu"), torch.float32
    )

    pred_anchor = torch.randn(1, NUM_ANCHOR, ANCHOR_DIM) * 20.0
    pred_anchor[0, 3] = ground_truth.anchor[0]
    cls_logits = torch.full((1, NUM_ANCHOR, 3), -6.0)
    cls_logits[0, 3, 1] = 6.0

    targets = head._assign_targets(
        cls_logits, pred_anchor, [ground_truth], [torch.tensor([7])], None
    )
    assert targets.num_pos == 1
    assert targets.assigned_instance[0, 3] == 7
    assert targets.cls_targets[0, 3, 1] == 1.0
    assert torch.allclose(targets.box_targets[0, 3], ground_truth.anchor[0])
    assert torch.allclose(targets.box_weights[0, 3], head.reg_weights)
    assert targets.box_weights[0, :3].sum() == 0
    assert targets.matched[0].tolist() == [index == 3 for index in range(NUM_ANCHOR)]


def _query_facing_away(class_index: int) -> tuple[torch.Tensor, list, torch.Tensor]:
    """A box of the class and the predictions, query 3 on the box with the opposite heading."""
    gt_boxes = torch.tensor([[5.0, -2.0, 0.5, 2.0, 0.5, 1.0, 0.4, 0.0, 0.0]])
    ground_truth = encode_ground_truth(
        gt_boxes, torch.tensor([class_index]), torch.device("cpu"), torch.float32
    )
    pred_anchor = torch.randn(1, NUM_ANCHOR, ANCHOR_DIM) * 20.0
    pred_anchor[0, 3] = ground_truth.anchor[0]
    pred_anchor[0, 3, 6:8] = -ground_truth.anchor[0, 6:8]
    cls_logits = torch.full((1, NUM_ANCHOR, 2), -6.0)
    cls_logits[0, 3, class_index] = 6.0
    return cls_logits, [ground_truth], pred_anchor


def test_a_yaw_free_class_takes_its_heading_out_of_the_regression(tmp_path) -> None:
    """
    Input: a head with class 1 yaw free, a box of class 1 and a query on it facing away.
    Expected: the query is matched and its heading channels get no regression weight.
    """
    head = build_sparse4d_head(
        make_anchor_file(tmp_path), class_names=["car", "cone"], yaw_free_classes=["cone"]
    )
    cls_logits, ground_truth, pred_anchor = _query_facing_away(1)
    targets = head._assign_targets(cls_logits, pred_anchor, ground_truth, [torch.tensor([7])], None)
    assert targets.assigned_instance[0, 3] == 7
    assert targets.box_weights[0, 3, 6:8].tolist() == [0.0, 0.0]
    assert bool((targets.box_weights[0, 3, :6] > 0).all())


def test_a_reversible_query_facing_away_regresses_the_reversed_heading(tmp_path) -> None:
    """
    Input: a head with class 1 yaw reversible, a query facing away from a box of class 1 and from
      a box of class 0.
    Expected: the class 1 target takes the query's heading, the class 0 target keeps its own.
    """
    head = build_sparse4d_head(
        make_anchor_file(tmp_path),
        class_names=["car", "barrier"],
        yaw_reversible_classes=["barrier"],
    )
    for class_index, reversed_heading in ((1, True), (0, False)):
        cls_logits, ground_truth, pred_anchor = _query_facing_away(class_index)
        targets = head._assign_targets(
            cls_logits, pred_anchor, ground_truth, [torch.tensor([7])], None
        )
        heading = head._reverse_facing_away(targets, pred_anchor).box_targets[0, 3, 6:8]
        expected = pred_anchor[0, 3, 6:8] if reversed_heading else ground_truth[0].anchor[0, 6:8]
        torch.testing.assert_close(heading, expected)


def test_unknown_velocity_is_masked_out_of_the_regression(head: Sparse4DHead) -> None:
    gt_boxes = torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, float("nan"), float("nan")]])
    ground_truth = encode_ground_truth(
        gt_boxes, torch.tensor([0]), torch.device("cpu"), torch.float32
    )
    assert torch.isfinite(ground_truth.anchor).all()
    assert ground_truth.channel_valid[0].tolist() == [True] * 8 + [False] * 2

    outputs = head([_sparse_stage_inputs(batch_size=1)])
    losses = head.loss(outputs, [gt_boxes], [torch.tensor([0])], [torch.tensor([0])], [0])
    assert torch.isfinite(losses["loss"])


def test_unknown_box_center_is_rejected(head: Sparse4DHead) -> None:
    gt_boxes = torch.tensor([[float("nan"), -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])
    with pytest.raises(ValueError, match="finite center"):
        encode_ground_truth(gt_boxes, torch.tensor([0]), torch.device("cpu"), torch.float32)


def test_loss_is_finite_without_ground_truth(head: Sparse4DHead) -> None:
    outputs = head([_sparse_stage_inputs(batch_size=2)])
    losses = head.loss(
        outputs,
        gt_boxes=[torch.zeros((0, 9)), torch.zeros((0, 9))],
        gt_labels=[torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long)],
        gt_instance_ids=[torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long)],
        lanes=[0, 1],
    )
    assert outputs.classification.shape == (2, head.num_decoder, NUM_ANCHOR, 3)
    assert outputs.prediction.shape == (2, head.num_decoder, NUM_ANCHOR, ANCHOR_DIM)
    assert outputs.quality.shape == (2, head.num_decoder, NUM_ANCHOR, 2)
    layer_keys = [key for key in losses if key.startswith("layer_")]
    assert len(layer_keys) == 4 * head.num_decoder
    assert torch.isfinite(losses["loss"])
    assert float(losses["layer_-1_loss_bbox"]) == 0.0
    losses["loss"].backward()


def test_every_parameter_gets_a_gradient_before_any_query_is_confident(
    head: Sparse4DHead,
) -> None:
    """DDP needs a gradient on every parameter before any query passes cls_threshold_to_reg."""
    gt_boxes = [torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]
    gt_labels = [torch.tensor([1])]
    head.cls_threshold_to_reg = 1.0

    losses = head.loss(
        head([_sparse_stage_inputs(batch_size=1)]), gt_boxes, gt_labels, [torch.tensor([0])], [0]
    )
    losses["loss"].backward()

    missing = [
        name
        for name, parameter in head.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    assert not head.instance_bank.instance_feature.requires_grad


def test_quality_losses_follow_the_matched_box(head: Sparse4DHead) -> None:
    gt_boxes = [torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]
    gt_labels = [torch.tensor([1])]
    head.cls_threshold_to_reg = 0.0

    losses = head.loss(
        head([_sparse_stage_inputs(batch_size=1)]), gt_boxes, gt_labels, [torch.tensor([0])], [0]
    )
    for prefix in ("layer_0", "layer_-1"):
        assert losses[f"{prefix}_loss_cns"] > 0.0
        assert losses[f"{prefix}_loss_yns"] >= 0.0


def test_decode_decodes_and_filters(head: Sparse4DHead) -> None:
    head.eval()
    head.score_threshold = None
    with torch.no_grad():
        outputs = head([_sparse_stage_inputs(batch_size=1)])
    predictions = _decode(head, outputs)
    assert predictions[0].bboxes_3d.shape == (head.num_output, 9)
    assert predictions[0].scores_3d.shape == (head.num_output,)
    assert predictions[0].labels_3d.max() < head.num_classes
    assert (predictions[0].bboxes_3d[:, 3:6] > 0).all()

    head.score_threshold = 1.1  # impossible threshold removes everything
    filtered = _decode(head, outputs)
    assert filtered[0].bboxes_3d.shape[0] == 0


def _decode(head: Sparse4DHead, outputs: Sparse4DHeadOutputs) -> list:
    """Decode raw Sparse4D outputs through the head interface every detection head shares."""
    return head.decode_outputs(
        Detection3DHeadOutputs(
            center_head_outputs=None, transfusion_head_outputs=None, sparse4d_head_outputs=outputs
        )
    ).detection3d_predictions


def _three_queries_at_one_place(outputs, labels: list[int], logits: list[float]):
    """Put queries 5, 6 and 7 on the anchor of query 5 with one confident class each."""
    prediction = outputs.prediction.clone()
    prediction[0, -1, 6:8] = prediction[0, -1, 5]
    classification = torch.full_like(outputs.classification, -10.0)
    for query, (label, logit) in enumerate(zip(labels, logits, strict=True), start=5):
        classification[0, -1, query, label] = logit
    return classification, prediction


def test_decode_nms_groups_drop_the_weaker_class_at_one_place(head: Sparse4DHead) -> None:
    head.eval()
    head.nms_groups = resolve_nms_groups(
        [NMSGroupConfig(class_names=["car", "pedestrian"], nms_radius=0.5, max_size=8)],
        ["car", "bicycle", "pedestrian"],
        default_radius=0.0,
        default_max_size=8,
    )
    with torch.no_grad():
        outputs = head([_sparse_stage_inputs(batch_size=1)])
    classification, prediction = _three_queries_at_one_place(outputs, [2, 0, 1], [2.0, 1.0, 0.5])
    outputs = type(outputs)(
        classification=classification,
        prediction=prediction,
        quality=torch.zeros_like(outputs.quality),
        instance_ids=None,
        denoising=None,
    )

    result = _decode(head, outputs)[0]
    at_place = torch.isclose(
        result.bboxes_3d, decode_anchor_space_to_boxes(prediction[0, -1, 5])
    ).all(dim=1)
    # The grouped car falls to the pedestrian, the ungrouped bicycle stays
    assert set(result.labels_3d[at_place].tolist()) == {1, 2}
    assert result.instance_ids is None


def test_decode_nms_groups_keep_the_tracked_box(head: Sparse4DHead) -> None:
    head.eval()
    head.nms_groups = resolve_nms_groups(
        [NMSGroupConfig(class_names=["car", "pedestrian"], nms_radius=0.5, max_size=8)],
        ["car", "bicycle", "pedestrian"],
        default_radius=0.0,
        default_max_size=8,
    )
    with torch.no_grad():
        outputs = head([_sparse_stage_inputs(batch_size=1)])
    classification, prediction = _three_queries_at_one_place(outputs, [2, 0, 1], [2.0, 1.0, 0.5])
    instance_ids = torch.full((1, NUM_ANCHOR), -1, dtype=torch.long)
    instance_ids[0, 6] = 41
    head.tracker = InstanceTracker(
        num_temp_instances=2, min_observed_points=1, track_birth_score=0.3
    )
    outputs = type(outputs)(
        classification=classification,
        prediction=prediction,
        quality=torch.zeros_like(outputs.quality),
        instance_ids=instance_ids,
        denoising=None,
    )

    result = _decode(head, outputs)[0]
    at_place = torch.isclose(
        result.bboxes_3d, decode_anchor_space_to_boxes(prediction[0, -1, 5])
    ).all(dim=1)
    # The confident tracked car outranks the stronger new pedestrian
    assert set(result.labels_3d[at_place].tolist()) == {0, 1}
    assert 41 in result.instance_ids.tolist()


def test_decode_scores_class_times_centerness(head: Sparse4DHead) -> None:
    head.eval()
    with torch.no_grad():
        outputs = head([_sparse_stage_inputs(batch_size=1)])
    classification = torch.full_like(outputs.classification, -10.0)
    classification[0, -1, 5, 2] = 2.0
    classification[0, -1, 5, 0] = 1.0
    quality = torch.zeros_like(outputs.quality)
    quality[0, -1, 5, 0] = 1.0
    outputs = type(outputs)(
        classification=classification,
        prediction=outputs.prediction,
        quality=quality,
        instance_ids=None,
        denoising=None,
    )

    prediction = _decode(head, outputs)[0]
    # One anchor may yield boxes of several classes, ranked by class score times centerness
    assert prediction.labels_3d[:2].tolist() == [2, 0]
    expected = torch.sigmoid(torch.tensor(2.0)) * torch.sigmoid(torch.tensor(1.0))
    assert torch.allclose(prediction.scores_3d[0], expected)
    assert torch.allclose(
        prediction.bboxes_3d[0], decode_anchor_space_to_boxes(outputs.prediction[0, -1, 5])
    )


NUM_TEMP = 4
BANK_ANCHORS = 8


def _make_bank(tmp_path, num_temp_instances: int = NUM_TEMP) -> InstanceBank:
    """Build a small temporal instance bank with a synthetic anchor file."""
    anchors = np.zeros((BANK_ANCHORS, ANCHOR_DIM), dtype=np.float32)
    anchors[:, 7] = 1.0
    anchor_file = tmp_path / "bank_anchors.npy"
    np.save(anchor_file, anchors)
    return InstanceBank(
        num_anchor=BANK_ANCHORS,
        embed_dims=8,
        anchor_file=str(anchor_file),
        num_temp_instances=num_temp_instances,
    )


def _pose(yaw: float, translation: tuple[float, float, float]) -> torch.Tensor:
    """Build a rigid (1, 4, 4) lidar-to-global pose."""
    pose = torch.eye(4, dtype=torch.float64)
    pose[0, 0] = np.cos(yaw)
    pose[0, 1] = -np.sin(yaw)
    pose[1, 0] = np.sin(yaw)
    pose[1, 1] = np.cos(yaw)
    pose[:3, 3] = torch.tensor(translation, dtype=torch.float64)
    return pose[None]


def _cache_frame(
    bank: InstanceBank,
    anchor: torch.Tensor,
    timestamp: torch.Tensor,
    pose: torch.Tensor,
    scene: str,
) -> None:
    """Run one get + cache cycle so the bank holds the given anchors."""
    bank.get(1, timestamp, pose, [scene])
    feature = torch.full((1, BANK_ANCHORS, 8), 7.0)
    logits = torch.full((1, BANK_ANCHORS, 3), 4.0)
    bank.cache(feature, anchor, logits, timestamp, pose, [scene])


def test_instance_bank_warps_static_object_across_frames(tmp_path) -> None:
    """A static object's cached anchor must land on the object in the next frame."""
    bank = _make_bank(tmp_path)
    global_point = torch.tensor([10.0, 4.0, 1.0, 1.0], dtype=torch.float64)
    pose1 = _pose(yaw=0.3, translation=(5.0, 2.0, 0.0))
    pose2 = _pose(yaw=-0.2, translation=(6.0, 2.5, 0.0))
    timestamp1 = torch.tensor([100.0], dtype=torch.float64)
    timestamp2 = torch.tensor([100.1], dtype=torch.float64)

    anchor1 = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
    anchor1[0, :, :3] = (pose1[0].inverse() @ global_point)[:3].float()
    anchor1[0, :, 7] = 1.0
    _cache_frame(bank, anchor1, timestamp1, pose1, "scene")

    _, _, time_interval, _, temp_anchor = bank.get(1, timestamp2, pose2, ["scene"])
    expected = (pose2[0].inverse() @ global_point)[:3].float()
    assert torch.allclose(time_interval, torch.tensor([0.1]), atol=1e-5)
    assert torch.allclose(temp_anchor[0, 0, :3], expected, atol=1e-4)


def test_instance_bank_warp_advances_by_instance_velocity(tmp_path) -> None:
    bank = _make_bank(tmp_path)
    pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
    timestamp1 = torch.tensor([10.0], dtype=torch.float64)
    timestamp2 = torch.tensor([10.5], dtype=torch.float64)

    anchor1 = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
    anchor1[0, :, 0] = 2.0
    anchor1[0, :, 7] = 1.0
    anchor1[0, :, 8] = 3.0  # vx
    _cache_frame(bank, anchor1, timestamp1, pose, "scene")

    _, _, _, _, temp_anchor = bank.get(1, timestamp2, pose, ["scene"])
    assert torch.allclose(temp_anchor[0, 0, 0], torch.tensor(2.0 + 3.0 * 0.5), atol=1e-5)


def test_instance_bank_resets_on_scene_change_and_time_gap(tmp_path) -> None:
    bank = _make_bank(tmp_path)
    pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
    timestamp = torch.tensor([10.0], dtype=torch.float64)
    anchor = torch.ones((1, BANK_ANCHORS, ANCHOR_DIM))
    _cache_frame(bank, anchor, timestamp, pose, "scene")

    # Scene change resets the slot.
    _, _, interval, temp_feature, _ = bank.get(1, timestamp + 0.1, pose, ["other"])
    assert torch.all(temp_feature == 0)
    assert torch.allclose(interval, torch.tensor([bank.default_time_interval]))

    # Excessive time gap resets the slot.
    _cache_frame(bank, anchor, timestamp, pose, "scene")
    _, _, interval, temp_feature, _ = bank.get(
        1, timestamp + bank.max_time_interval + 1.0, pose, ["scene"]
    )
    assert torch.all(temp_feature == 0)
    assert torch.allclose(interval, torch.tensor([bank.default_time_interval]))

    # A gap shorter than a frame period resets the slot.
    _cache_frame(bank, anchor, timestamp, pose, "scene")
    _, _, interval, temp_feature, _ = bank.get(1, timestamp + 1e-4, pose, ["scene"])
    assert torch.all(temp_feature == 0)
    assert torch.allclose(interval, torch.tensor([bank.default_time_interval]))


@pytest.mark.parametrize(("min_interval", "max_interval"), [(0.0, 2.0), (2.0, 2.0)])
def test_instance_bank_rejects_an_empty_time_gate(
    tmp_path, min_interval: float, max_interval: float
) -> None:
    anchor_file = tmp_path / "bank_anchors.npy"
    np.save(anchor_file, np.zeros((BANK_ANCHORS, ANCHOR_DIM), dtype=np.float32))
    with pytest.raises(ValueError, match="time gate"):
        InstanceBank(
            num_anchor=BANK_ANCHORS,
            embed_dims=8,
            anchor_file=str(anchor_file),
            min_time_interval=min_interval,
            max_time_interval=max_interval,
        )


def test_instance_bank_update_mixes_cached_and_top_fresh(tmp_path) -> None:
    bank = _make_bank(tmp_path)
    pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
    timestamp1 = torch.tensor([10.0], dtype=torch.float64)
    anchor = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
    anchor[0, :, 7] = 1.0
    _cache_frame(bank, anchor, timestamp1, pose, "scene")
    bank.get(1, timestamp1 + 0.1, pose, ["scene"])

    fresh_feature = torch.full((1, BANK_ANCHORS, 8), 1.0)
    fresh_anchor = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
    logits = torch.arange(BANK_ANCHORS, dtype=torch.float32).flip(0)[None, :, None]
    logits = logits.expand(-1, -1, 3).contiguous()

    merged_feature, _ = bank.update(fresh_feature, fresh_anchor, logits)
    # Leading block holds the propagated (cached) features, tail the top fresh.
    assert torch.all(merged_feature[0, :NUM_TEMP] == 7.0)
    assert torch.all(merged_feature[0, NUM_TEMP:] == 1.0)


def test_instance_bank_cache_applies_confidence_decay(tmp_path) -> None:
    bank = _make_bank(tmp_path)
    pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
    timestamp1 = torch.tensor([10.0], dtype=torch.float64)
    anchor = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
    anchor[0, :, 7] = 1.0
    _cache_frame(bank, anchor, timestamp1, pose, "scene")  # sigmoid(4) ~ 0.982
    previous_confidence = torch.sigmoid(torch.tensor(4.0))

    bank.get(1, timestamp1 + 0.1, pose, ["scene"])
    low_logits = torch.full((1, BANK_ANCHORS, 3), -10.0)
    bank.cache(
        torch.zeros((1, BANK_ANCHORS, 8)),
        anchor,
        low_logits,
        timestamp1 + 0.1,
        pose,
        ["scene"],
    )
    # Decay is per second: 0.1 s frame gap applies confidence_decay ** 0.1.
    expected = previous_confidence * bank.confidence_decay**0.1
    assert torch.allclose(bank._cached_confidence[0, 0], expected, atol=1e-5)


def _build_temporal_head(tmp_path) -> Sparse4DHead:
    """Build a small temporal Sparse4D head with a synthetic anchor file."""
    rng = np.random.default_rng(0)
    anchors = np.zeros((NUM_ANCHOR, ANCHOR_DIM), dtype=np.float32)
    anchors[:, :2] = rng.uniform(-40.0, 40.0, size=(NUM_ANCHOR, 2))
    anchors[:, 7] = 1.0
    anchor_file = tmp_path / "temporal_anchors.npy"
    np.save(anchor_file, anchors)
    return Sparse4DHead(
        num_classes=3,
        embed_dims=EMBED_DIMS,
        anchor_file=str(anchor_file),
        num_anchor=NUM_ANCHOR,
        point_cloud_range=POINT_CLOUD_RANGE,
        grid_size=1.0,
        feature_strides=(FEATURE_STRIDE,),
        num_decoder=2,
        num_heads=4,
        num_groups=4,
        feedforward_channels=64,
        num_learnable_pts=2,
        num_output=8,
        num_temp_instances=8,
        min_observed_points=1,
        track_birth_score=0.0,
    )


def test_temporal_head_runs_two_frames_and_backprops(tmp_path) -> None:
    head = _build_temporal_head(tmp_path)
    pose = _pose(yaw=0.1, translation=(1.0, 0.0, 0.0))
    gt_boxes = [torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]
    gt_labels = [torch.tensor([1])]

    for frame_index in range(2):
        outputs = head(
            [_sparse_stage_inputs(batch_size=1)],
            timestamps_seconds=torch.tensor([100.0 + 0.1 * frame_index], dtype=torch.float64),
            lidar2global=pose,
            scene_tokens=["scene"],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_instance_ids=[torch.tensor([3])],
            gt_num_points=[torch.tensor([5])],
        )
        losses = head.loss(outputs, gt_boxes, gt_labels, [torch.tensor([3])], [0])
        assert torch.isfinite(losses["loss"])
        losses["loss"].backward()


def test_temporal_export_matches_training_across_two_frames(tmp_path) -> None:
    """The export state loop must reproduce the training-side instance bank."""
    head = _build_temporal_head(tmp_path).eval()
    export_head = head.prepare_for_export()
    stage_inputs = _sparse_stage_inputs(batch_size=1)
    pose1 = _pose(yaw=0.3, translation=(5.0, 2.0, 0.0))
    pose2 = _pose(yaw=-0.2, translation=(6.0, 2.5, 0.0))
    timestamp1 = torch.tensor([100.0], dtype=torch.float64)
    timestamp2 = torch.tensor([100.5], dtype=torch.float64)

    with torch.no_grad():
        train1 = head(
            [stage_inputs],
            timestamps_seconds=timestamp1,
            lidar2global=pose1,
            scene_tokens=["scene"],
        )
        export1 = export_head.forward_temporal_export(
            [stage_inputs], *export_head.temporal_state_export_inputs(torch.device("cpu"))
        )
        train2 = head(
            [stage_inputs],
            timestamps_seconds=timestamp2,
            lidar2global=pose2,
            scene_tokens=["scene"],
        )
        export2 = export_head.forward_temporal_export(
            [stage_inputs],
            export1["temp_instance_feature_out"],
            export1["temp_anchor_out"],
            export1["temp_confidence_out"],
            torch.ones(1),
            torch.tensor([0.5]),
            torch.linalg.solve(pose2, pose1).to(torch.float32),
        )

    bank = head.instance_bank
    for train_outputs, export_outputs in ((train1, export1), (train2, export2)):
        assert torch.allclose(
            export_outputs["classification"], train_outputs.classification[:, -1], atol=1e-5
        )
        assert torch.allclose(
            export_outputs["prediction"], train_outputs.prediction[:, -1], atol=1e-5
        )
        assert torch.allclose(export_outputs["quality"], train_outputs.quality[:, -1], atol=1e-5)
    assert torch.allclose(export2["temp_instance_feature_out"], bank._cached_feature, atol=1e-5)
    assert torch.allclose(export2["temp_anchor_out"], bank._cached_anchor, atol=1e-4)
    assert torch.allclose(export2["temp_confidence_out"], bank._cached_confidence, atol=1e-5)
    assert torch.equal(export2["temp_query_index_out"], bank.cached_query_index)


def test_export_mode_matches_training_forward(head: Sparse4DHead) -> None:
    head.eval()
    export_head = head.prepare_for_export()
    stage_inputs = _sparse_stage_inputs(batch_size=1)
    with torch.no_grad():
        train_outputs = head([stage_inputs])
        export_outputs = export_head([stage_inputs])
    classification, prediction = export_outputs.export_tensors(["classification", "prediction"])
    assert export_outputs.classification.shape == (1, 1, NUM_ANCHOR, 3)
    assert classification.shape == (1, NUM_ANCHOR, 3)
    assert torch.allclose(classification, train_outputs.classification[:, -1], atol=1e-5)
    assert torch.allclose(prediction, train_outputs.prediction[:, -1], atol=1e-5)


def test_every_sampled_stage_reaches_the_queries(tmp_path) -> None:
    head = build_sparse4d_head(make_anchor_file(tmp_path), feature_strides=(1, 2)).eval()
    stages = [
        random_stage(grid.grid_dims, 64 // (index + 1))
        for index, grid in enumerate(head.level_grids)
    ]

    outputs = head(stages)
    outputs.classification.sum().backward()
    with torch.no_grad():
        plain = head(stages)

    assert all(feat.grad.abs().sum() > 0 for feat, _, _ in stages)
    # The checkpointed sampling of the gradient pass matches the plain one
    assert torch.allclose(plain.classification, outputs.classification.detach(), atol=1e-6)


def test_level_sampling_interpolates_and_reads_zeros_off_the_voxels() -> None:
    grid = SparseLevelGrid([0.0, 0.0, 0.0, 4.0, 4.0, 4.0], grid_size=1.0, feature_stride=1)
    feat = torch.tensor([[2.0, -4.0]], requires_grad=True)
    level = grid(feat, torch.tensor([[1, 1, 1]]), torch.tensor([1]))
    # The voxel center, halfway to the empty neighbour, an empty cell and outside the grid
    key_points = torch.tensor([[[[1.5, 1.5, 1.5], [2.0, 1.5, 1.5], [3.5, 3.5, 3.5], [-5.0, 0, 0]]]])

    sampled, occupancy = level.sample(key_points)
    sampled.sum().backward()

    expected = torch.tensor([[2.0, -4.0], [1.0, -2.0], [0.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(sampled[0, 0], expected)
    assert torch.allclose(occupancy[0, 0], torch.tensor([1.0, 0.5, 0.0, 0.0]))
    assert torch.allclose(feat.grad, torch.tensor([[1.5, 1.5]]))


def test_keypoints_in_empty_space_do_not_dilute_the_aggregated_features() -> None:
    aggregation = SparseDeformableAggregation(
        embed_dims=4,
        num_groups=1,
        num_levels=2,
        kps_generator=SparseBox3DKeyPointsGenerator(
            embed_dims=4, fix_scale=((0.0, 0.0, 0.0), (0.45, 0.0, 0.0)), num_learnable_pts=0
        ),
    ).eval()
    nn.init.eye_(aggregation.output_proj.weight)
    nn.init.zeros_(aggregation.output_proj.bias)
    grid = SparseLevelGrid([0.0, 0.0, 0.0, 8.0, 8.0, 8.0], grid_size=1.0, feature_stride=1)
    feat = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    occupied = grid(feat, torch.tensor([[4, 4, 4]]), torch.tensor([1]))
    empty = grid(feat, torch.tensor([[0, 0, 0]]), torch.tensor([1]))
    # A tiny box at the occupied voxel center, every keypoint reads that voxel
    anchor = torch.zeros((1, 1, ANCHOR_DIM))
    anchor[..., :3] = 4.5
    anchor[..., 3:6] = math.log(0.01)
    anchor[..., 7] = 1.0
    zeros = torch.zeros((1, 1, 4))

    with torch.no_grad():
        both = aggregation(zeros, anchor, zeros, [occupied, empty])

    # Half the keypoints sample the empty stage, the feature still reads at full strength
    assert torch.allclose(both[0, 0], feat[0], atol=1e-5)


def test_a_lidar_block_rejects_a_frame_context(tmp_path) -> None:
    head = build_sparse4d_head(make_anchor_file(tmp_path))
    levels = head.levels([random_stage(grid.grid_dims, 64) for grid in head.level_grids])
    anchor = head.instance_bank.anchor[None].detach()

    with pytest.raises(ValueError, match="no frame context"):
        head.deformable[0](
            torch.zeros(1, 8, 32), anchor, head.anchor_encoder(anchor), levels, context=object()
        )
