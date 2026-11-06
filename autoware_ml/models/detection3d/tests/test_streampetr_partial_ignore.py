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

"""Unit tests for the partial annotations of the StreamPETR head."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.dataclasses.models.detection3d.head_outputs import StreamPETRHeadOutputs
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations

CLASS_NAMES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
PARTIAL_ANNOTATIONS = PartialAnnotations(CLASS_NAMES, ["traffic_cone", "barrier"])
POINT_CLOUD_RANGE = [-10.0, -10.0, -5.0, 10.0, 10.0, 5.0]


def _masks(status: list[bool]) -> torch.Tensor:
    return PARTIAL_ANNOTATIONS.build_annotated_class_masks(torch.tensor(status))


def _build_head() -> StreamPETRHead:
    return StreamPETRHead(
        class_names=CLASS_NAMES,
        in_channels=32,
        hidden_dim=32,
        num_queries=16,
        num_decoder_layers=2,
        num_heads=4,
        feedforward_channels=64,
        memory_len=16,
        topk_proposals=4,
        num_propagated=4,
        with_dn=True,
        with_ego_pos=True,
        depth_num=8,
        LID=True,
        position_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
        scalar=2,
        noise_scale=0.5,
        dn_weight=1.0,
        split=0.5,
        bbox_coder=NMSFreeBBoxCoder3D(
            pc_range=POINT_CLOUD_RANGE,
            post_center_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
            score_threshold=0.01,
            max_num=8,
        ),
        assigner=HungarianAssigner3D(
            cls_cost=ClassificationCost(weight=2.0),
            reg_cost=BBox3DL1Cost(weight=0.25),
            iou_cost=IoU3DCost(weight=0.0),
            point_cloud_range=POINT_CLOUD_RANGE,
        ),
        point_cloud_range=POINT_CLOUD_RANGE,
        code_weights=[2.0, 2.0] + [1.0] * 8,
    )


def test_partial_annotations_mask_only_the_cone_and_barrier_columns() -> None:
    """
    Input: the traffic-cone and barrier status of two frames over the seven classes, True for
    the first frame and False for the second.
    Expected: the first frame supervises every class, the second one every class but
    traffic_cone and barrier.
    Check: the mask shape, every column of the first row, and columns 5 and 6 against the first
    five of the second row.
    """
    masks = _masks([True, False])
    assert masks.shape == (2, 7)
    assert masks[0].all()
    assert not masks[1, [5, 6]].any()
    assert masks[1, :5].all()


def test_sigmoid_focal_loss_classwise_weights_zero_masked_columns() -> None:
    """
    Input: random logits of four queries over seven classes, all background, scored without
    weights, with weights of one, and with weights of zero on columns 5 and 6.
    Expected: weights of one change nothing, and zero-weighted columns drop out of the loss as
    if their classes did not exist.
    Check: the unweighted loss against the one with weights of one, and the masked loss against
    the loss of the first five columns alone.
    """
    loss_fn = SigmoidFocalLoss()
    logits = torch.randn(4, 7)
    targets = torch.zeros(4, 7)
    ones = torch.ones(4, 7)
    baseline = loss_fn(logits, targets, avg_factor=1.0)
    weighted = loss_fn(logits, targets, weights=ones, avg_factor=1.0)
    assert torch.allclose(baseline, weighted)

    masked_weights = ones.clone()
    masked_weights[:, [5, 6]] = 0.0
    masked = loss_fn(logits, targets, weights=masked_weights, avg_factor=1.0)
    manual = loss_fn(logits[:, :5], targets[:, :5], avg_factor=1.0)
    assert torch.allclose(masked, manual, atol=1e-6)


def test_get_targets_weights_every_query_by_the_mask_of_its_frame() -> None:
    """
    Input: the classification targets of a frame with one box, built with the mask of a frame
    annotated without traffic cones and barriers, without a mask, and for a batch of two frames,
    the first annotated with them and the second without.
    Expected: every query of a frame annotated without them but the matched one loses only the
    cone and barrier columns of its classification weights, while the matched query keeps every
    column; no mask keeps the weights uniform, and in a batch every frame weights its queries by
    its own mask.
    Check: the label weights of the masked frame by query and column, None without a mask, and
    the weights of both frames of the batch.
    """
    head = _build_head()
    num_queries = 16
    cls_logits = torch.randn(1, num_queries, 7)
    box_params = torch.randn(1, num_queries, 10)
    box_params[..., :3] = 0.0
    gt_boxes = [torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    targets = head._get_targets(cls_logits, box_params, gt_boxes, gt_labels, _masks([False]))[0]
    assert targets.label_weights is not None
    assert targets.label_weights.shape == (num_queries, 7)

    # Every query of the un-annotated frame but the matched one loses only the cone/barrier
    # columns; the matched query is vouched for by its box.
    matched = targets.labels >= 0
    assert int(matched.sum()) == 1
    assert torch.all(targets.label_weights[matched] == 1.0)
    assert torch.all(targets.label_weights[~matched][:, [5, 6]] == 0.0)
    assert torch.all(targets.label_weights[:, :5] == 1.0)

    # Without a mask the weights stay uniform (None).
    plain = head._get_targets(cls_logits, box_params, gt_boxes, gt_labels)[0]
    assert plain.label_weights is None

    # In a mixed batch every sample carries its own row.
    mixed = head._get_targets(
        cls_logits.repeat(2, 1, 1),
        box_params.repeat(2, 1, 1),
        gt_boxes * 2,
        gt_labels * 2,
        _masks([True, False]),
    )
    assert torch.all(mixed[0].label_weights == 1.0)
    assert torch.all(mixed[1].label_weights[mixed[1].labels < 0][:, [5, 6]] == 0.0)


def test_dn_label_weights_mask_every_denoising_query_of_unannotated_samples() -> None:
    """
    Input: six denoising queries of two samples, sample 1 annotated without traffic cones and
    barriers; one of its queries keeps the label of its box (a traffic cone), two are noised
    into the background.
    Expected: the background queries of sample 1 lose the cone and barrier columns, its query
    with a label keeps every column, the queries of sample 0 stay fully weighted, and no mask
    gives no weights.
    Check: the weights of the queries of each sample by column, and None without a mask.
    """
    head = _build_head()
    cls_scores = torch.randn(6, 7)
    known_bids = torch.tensor([0, 0, 1, 1, 1, 0])
    # 7 is the background label of the 7 classes.
    known_labels = torch.tensor([0, 7, 5, 7, 7, 0])
    weights = head._dn_label_weights(cls_scores, known_bids, known_labels, _masks([True, False]))
    assert weights is not None
    # The background rows of sample 1 (indices 3, 4) are masked on columns 5/6.
    assert torch.all(weights[[3, 4]][:, [5, 6]] == 0.0)
    assert torch.all(weights[[3, 4]][:, :5] == 1.0)
    # Its traffic cone row and the rows of the annotated sample 0 stay fully weighted.
    assert torch.all(weights[[0, 1, 2, 5]] == 1.0)

    assert head._dn_label_weights(cls_scores, known_bids, known_labels, None) is None


def test_a_matched_query_of_an_unannotated_class_keeps_its_terms() -> None:
    """
    Input: one traffic cone in a frame annotated without traffic cones and barriers, scored by
    the StreamPETR head loss.
    Expected: the query matched to the cone is vouched for by its box and keeps every
    classification term, as in TransFusion, so the cone is learned even there; every other
    query loses its cone and barrier terms.
    Check: the gradient of the classification loss on the cone and barrier logits of the last
    decoder layer, non-zero on exactly one query.
    """
    torch.manual_seed(0)
    head = _build_head().eval()
    cls_scores = torch.randn(2, 1, 16, 7, requires_grad=True)
    outputs = StreamPETRHeadOutputs(
        all_cls_scores=cls_scores, all_bbox_preds=torch.rand(2, 1, 16, 10), denoising=None
    )
    gt_boxes = [torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.1, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([5], dtype=torch.long)]

    losses = head.loss(
        outputs,
        gt_boxes,
        gt_labels,
        padding_mask=torch.zeros(1, dtype=torch.bool),
        gt_annotated_class_masks=_masks([False]),
    )

    (grad,) = torch.autograd.grad(losses["loss_cls"], cls_scores)
    assert torch.count_nonzero(grad[-1, 0, :, 5:7].abs().sum(dim=-1)) == 1


def test_head_loss_applies_the_annotated_class_masks() -> None:
    """
    Input: the StreamPETR head loss of one frame with one box, without a mask, with the mask of
    a frame annotated with traffic cones and barriers, with the mask of a frame annotated
    without them, and with the masks of two frames.
    Expected: the mask of an annotated frame gives the plain loss, and the other mask removes
    non-negative background terms from the classification loss only; masks of another batch size
    are refused.
    Check: the total losses, the classification and regression losses, and a ValueError naming
    the expected shape.
    """
    torch.manual_seed(0)
    head = _build_head().eval()
    outputs = StreamPETRHeadOutputs(
        all_cls_scores=torch.randn(2, 1, 16, 7),
        all_bbox_preds=torch.rand(2, 1, 16, 10),
        denoising=None,
    )
    gt_boxes = [torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]
    real = torch.zeros(1, dtype=torch.bool)

    plain = head.loss(outputs, gt_boxes, gt_labels, padding_mask=real)
    annotated = head.loss(
        outputs, gt_boxes, gt_labels, padding_mask=real, gt_annotated_class_masks=_masks([True])
    )
    ignored = head.loss(
        outputs, gt_boxes, gt_labels, padding_mask=real, gt_annotated_class_masks=_masks([False])
    )
    torch.testing.assert_close(annotated["loss"], plain["loss"])
    assert ignored["loss_cls"] < annotated["loss_cls"]
    assert torch.allclose(ignored["loss_bbox"], annotated["loss_bbox"])

    with pytest.raises(ValueError, match="batch_size, num_classes"):
        head.loss(
            outputs,
            gt_boxes,
            gt_labels,
            padding_mask=real,
            gt_annotated_class_masks=_masks([True, False]),
        )
