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

"""Unit tests of the annotated class masks in the StreamPETR head loss.

The head takes the ``(batch_size, num_classes)`` mask of the classes annotated in every sample;
``PartialAnnotations``, which builds it from the dataset flag, is tested in ``task_modules``.
Every term of the sigmoid focal loss is independent of the others, so the masks are checked term
by term: with them, the gradient of every logit is the gradient without them times the weight its
term should have.
"""

from __future__ import annotations

from typing import NamedTuple
import unittest

import torch

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    StreamPETRDenoisingOutputs,
    StreamPETRHeadOutputs,
)
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)

CLASS_NAMES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
NUM_CLASSES = len(CLASS_NAMES)
TRAFFIC_CONE, BARRIER = 5, 6
NUM_QUERIES = 16
NUM_LAYERS = 2
SCALAR = 2
POINT_CLOUD_RANGE = [-10.0, -10.0, -5.0, 10.0, 10.0, 5.0]


def _build_head() -> StreamPETRHead:
    torch.manual_seed(0)
    return StreamPETRHead(
        class_names=CLASS_NAMES,
        in_channels=32,
        hidden_dim=32,
        num_queries=NUM_QUERIES,
        num_decoder_layers=NUM_LAYERS,
        num_heads=4,
        feedforward_channels=64,
        memory_len=16,
        topk_proposals=4,
        num_propagated=4,
        with_dn=True,
        depth_num=8,
        position_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
        scalar=SCALAR,
        noise_scale=0.5,
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
    ).train()


class _Batch(NamedTuple):
    """Head outputs of a three-sample batch with its ground truth and masks."""

    cls_scores: torch.Tensor
    bbox_preds: torch.Tensor
    gt_boxes: list[torch.Tensor]
    gt_labels: list[torch.Tensor]
    padding_mask: torch.Tensor
    annotated_class_masks: torch.Tensor
    # The denoising queries, None without boxes: (logits, boxes, labels, samples, slots).
    denoising: tuple[torch.Tensor, ...] | None


def _batch(with_boxes: bool) -> _Batch:
    """Sample 0 is annotated with every class and holds a car and a traffic cone; sample 1 is
    annotated without traffic cones and barriers and still holds a truck and a traffic cone;
    sample 2 pads the batch and is flagged as annotated with every class.

    Without boxes the samples hold none, so every query is unmatched and there is no denoising.
    """
    generator = torch.Generator().manual_seed(0)
    labels = [torch.tensor([0, TRAFFIC_CONE]), torch.tensor([1, TRAFFIC_CONE])]
    if not with_boxes:
        labels = [torch.zeros(0, dtype=torch.long)] * 2
    labels.append(torch.zeros(0, dtype=torch.long))
    boxes = []
    for sample_labels in labels:
        sample_boxes = torch.rand(sample_labels.numel(), 9, generator=generator) * 8.0 - 4.0
        sample_boxes[:, 3:6] = sample_boxes[:, 3:6].abs() + 0.5
        boxes.append(sample_boxes)
    annotated_class_masks = torch.ones(3, NUM_CLASSES, dtype=torch.bool)
    annotated_class_masks[1, [TRAFFIC_CONE, BARRIER]] = False

    denoising = None
    if with_boxes:
        # One query per box and group, laid out as the head lays them out (two slots per
        # sample and group); three of them are noised into the background.
        known_labels = torch.cat(labels).repeat(SCALAR)
        known_labels[[1, 3, 6]] = NUM_CLASSES
        denoising = (
            torch.randn(NUM_LAYERS, 3, 2 * SCALAR, NUM_CLASSES, generator=generator),
            torch.randn(NUM_LAYERS, 3, 2 * SCALAR, 10, generator=generator),
            known_labels,
            torch.tensor([0, 0, 1, 1]).repeat(SCALAR),
            torch.tensor([0, 1, 0, 1, 2, 3, 2, 3]),
        )
    return _Batch(
        cls_scores=torch.randn(NUM_LAYERS, 3, NUM_QUERIES, NUM_CLASSES, generator=generator),
        bbox_preds=torch.randn(NUM_LAYERS, 3, NUM_QUERIES, 10, generator=generator),
        gt_boxes=boxes,
        gt_labels=labels,
        padding_mask=torch.tensor([False, False, True]),
        annotated_class_masks=annotated_class_masks,
        denoising=denoising,
    )


def _losses_and_gradients(
    head: StreamPETRHead, batch: _Batch, masks: torch.Tensor | None
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """The losses of the batch and the gradients of the logits and boxes, denoising ones last."""
    leaves = [batch.cls_scores.clone().requires_grad_(), batch.bbox_preds.clone().requires_grad_()]
    denoising = None
    if batch.denoising is not None:
        dn_cls_scores, dn_bbox_preds, known_labels, batch_indices, slot_indices = batch.denoising
        leaves += [dn_cls_scores.clone().requires_grad_(), dn_bbox_preds.clone().requires_grad_()]
        denoising = StreamPETRDenoisingOutputs(
            cls_scores=leaves[2],
            bbox_preds=leaves[3],
            known_labels=known_labels,
            known_boxes=torch.cat(batch.gt_boxes).repeat(SCALAR, 1),
            batch_indices=batch_indices,
            slot_indices=slot_indices,
        )
    outputs = StreamPETRHeadOutputs(
        all_cls_scores=leaves[0], all_bbox_preds=leaves[1], denoising=denoising
    )
    losses = head.loss(
        outputs,
        batch.gt_boxes,
        batch.gt_labels,
        padding_mask=batch.padding_mask,
        gt_annotated_class_masks=masks,
    )
    losses["loss"].backward()
    return losses, [leaf.grad for leaf in leaves]


def _expected_weights(head: StreamPETRHead, batch: _Batch) -> list[torch.Tensor]:
    """The weight every classification term should have, for the queries and the denoising ones.

    Every query weights its terms by the annotated classes of its sample, none for a padding
    sample. A query the assigner matches to a box, or a denoising query that keeps the label of
    its box, is vouched for by the box and keeps every term.
    """
    supervised = (batch.annotated_class_masks & ~batch.padding_mask.unsqueeze(-1)).float()
    weights = supervised[None, :, None, :].repeat(NUM_LAYERS, 1, NUM_QUERIES, 1)
    for layer in range(NUM_LAYERS):
        targets = head._get_targets(
            batch.cls_scores[layer], batch.bbox_preds[layer], batch.gt_boxes, batch.gt_labels
        )
        for sample, sample_targets in enumerate(targets):
            weights[layer, sample, sample_targets.labels >= 0] = 1.0
    if batch.denoising is None:
        return [weights]
    dn_cls_scores, _, known_labels, batch_indices, slot_indices = batch.denoising
    dn_weights = torch.zeros_like(dn_cls_scores)
    for label, sample, slot in zip(known_labels, batch_indices, slot_indices):
        dn_weights[:, sample, slot] = 1.0 if label < NUM_CLASSES else supervised[sample]
    return [weights, dn_weights]


class TestStreamPETRPartialAnnotations(unittest.TestCase):
    def test_every_classification_term_takes_the_annotated_classes_of_its_sample(self) -> None:
        """
        Input: a batch of a sample annotated with every class, a sample annotated without
            traffic cones and barriers that still holds a traffic cone, and a padding sample
            flagged as annotated; with boxes and denoising queries, and without any box.
        Expected: a matched query and a denoising query that keeps its label keep every term,
            the traffic cone of the unannotated sample included; the other queries and the
            denoising queries noised into the background keep the annotated classes of their
            sample; the padding sample keeps none.
        Check: every classification gradient with the masks against the gradient without them
            times the expected weight, to the bit; that some terms are dropped, and that in the
            unannotated sample exactly its two matched queries keep the traffic-cone and barrier
            terms.
        """
        head = _build_head()
        for with_boxes in (True, False):
            with self.subTest(with_boxes=with_boxes):
                batch = _batch(with_boxes)
                _, plain = _losses_and_gradients(head, batch, None)
                _, masked = _losses_and_gradients(head, batch, batch.annotated_class_masks)
                weights = _expected_weights(head, batch)

                # The logits are the first and, with denoising, the third leaves.
                for observed, unmasked, weight in zip(masked[::2], plain[::2], weights):
                    torch.testing.assert_close(observed, unmasked * weight, rtol=0.0, atol=0.0)
                self.assertTrue(torch.any((weights[0] == 0.0) & (plain[0] != 0.0)))
                kept = masked[0][:, 1, :, [TRAFFIC_CONE, BARRIER]].ne(0.0).any(dim=-1)
                self.assertEqual(kept.sum(dim=-1).tolist(), [2 if with_boxes else 0] * NUM_LAYERS)

    def test_masks_change_neither_the_regression_nor_the_counts(self) -> None:
        """
        Input: the batch with boxes and denoising queries, with and without the masks.
        Expected: the masks only drop classification terms, so the regression losses, divided
            by the same counts, stay the same.
        Check: the regression losses and the gradients of every box, to the bit.
        """
        head = _build_head()
        batch = _batch(with_boxes=True)
        plain_losses, plain = _losses_and_gradients(head, batch, None)
        masked_losses, masked = _losses_and_gradients(head, batch, batch.annotated_class_masks)

        for key in [key for key in plain_losses if "bbox" in key]:
            self.assertTrue(torch.equal(masked_losses[key], plain_losses[key]), key)
        for observed, unmasked in zip(masked[1::2], plain[1::2]):
            self.assertTrue(torch.equal(observed, unmasked))

    def test_masks_with_every_class_annotated_give_the_loss_without_masks(self) -> None:
        """
        Input: the batch with boxes and denoising queries, without masks and with masks that
            annotate every class of every sample.
        Expected: nothing is dropped, so the losses are those without masks.
        Check: every loss term and every gradient, to the bit.
        """
        head = _build_head()
        batch = _batch(with_boxes=True)
        plain_losses, plain = _losses_and_gradients(head, batch, None)
        full_losses, full = _losses_and_gradients(
            head, batch, torch.ones(3, NUM_CLASSES, dtype=torch.bool)
        )

        self.assertEqual(full_losses.keys(), plain_losses.keys())
        for key, value in plain_losses.items():
            self.assertTrue(torch.equal(full_losses[key], value), key)
        for observed, unmasked in zip(full, plain):
            self.assertTrue(torch.equal(observed, unmasked))

    def test_masks_of_another_shape_are_refused(self) -> None:
        """
        Input: annotated class masks of another batch size or class count, and a padding mask
            of another batch size.
        Expected: such a mask would weight the wrong samples or classes, so the head refuses it.
        Check: the ValueError naming the expected shape.
        """
        head = _build_head()
        batch = _batch(with_boxes=False)
        cases = {
            "masks of another batch size": (
                {"gt_annotated_class_masks": torch.ones(2, NUM_CLASSES, dtype=torch.bool)},
                r"\(3, 7\)",
            ),
            "masks of another class count": (
                {"gt_annotated_class_masks": torch.ones(3, 3, dtype=torch.bool)},
                r"\(3, 7\)",
            ),
            "padding mask of another batch size": (
                {"padding_mask": torch.zeros(2, dtype=torch.bool)},
                r"\(3,\)",
            ),
        }
        for case, (overrides, expected_shape) in cases.items():
            arguments = {"padding_mask": batch.padding_mask, **overrides}
            with self.subTest(case):
                with self.assertRaisesRegex(ValueError, expected_shape):
                    head.loss(
                        StreamPETRHeadOutputs(
                            all_cls_scores=batch.cls_scores,
                            all_bbox_preds=batch.bbox_preds,
                            denoising=None,
                        ),
                        batch.gt_boxes,
                        batch.gt_labels,
                        **arguments,
                    )


if __name__ == "__main__":
    unittest.main()
