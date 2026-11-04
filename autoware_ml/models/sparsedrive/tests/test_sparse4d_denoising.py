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

"""Unit tests for the Sparse4D denoising training."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.models.sparsedrive.sparse4d_head import (
    ANCHOR_DIM,
    build_denoising_attention_mask,
    build_denoising_queries,
)
from autoware_ml.models.sparsedrive.lidar_stages import SparseStage
from autoware_ml.models.detection3d.task_modules.bbox_coders import normalize_boxes3d
from autoware_ml.models.sparsedrive.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    forward_frame,
    gt_batch,
    head_loss,
    make_anchor_recipe,
    random_stage,
)

_REG_WEIGHTS = torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])
_NOISE_KWARGS = {
    "noise_center": 0.5,
    "noise_center_neg": 1.0,
    "noise_size": 0.2,
    "noise_yaw": 0.39,
    "noise_velocity": 1.0,
}


def _build_gt() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Two-sample batch: two boxes and one box (padding case)."""
    gt_boxes = [
        torch.tensor(
            [
                [2.5, 3.0, 0.2, 1.8, 0.9, 1.6, 0.1, 0.5, -0.2],
                [5.5, 5.0, -0.4, 3.6, 1.7, 1.5, -0.8, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[4.0, 2.0, 0.0, 1.0, 1.0, 1.8, 1.2, 0.1, 0.1]],
            dtype=torch.float32,
        ),
    ]
    gt_labels = [torch.tensor([0, 1], dtype=torch.long), torch.tensor([1], dtype=torch.long)]
    return gt_boxes, gt_labels


def _build_gt_batch() -> Detection3DGTBatch:
    """The two-sample batch as the detection ground truth of the head."""
    return gt_batch(*_build_gt())


def _build_stages() -> list[SparseStage]:
    """Sparse stage of the two samples for the fixture head (grid 8x8x4, 32 channels)."""
    return [random_stage((8, 8, 4), 6, batch_size=2)]


class TestSparse4DDenoising(unittest.TestCase):
    def test_denoising_queries_targets_and_layout(self) -> None:
        gt_boxes, gt_labels = _build_gt()
        torch.manual_seed(0)
        denoising = build_denoising_queries(
            gt_boxes,
            gt_labels,
            num_query=8,
            num_classes=2,
            query_budget=12,
            reg_weights=_REG_WEIGHTS,
            **_NOISE_KWARGS,
        )

        # budget 12 with max_gt 2 gives 3 groups of width 4 (2 positive and 2 negative).
        self.assertEqual(denoising.anchor.shape, (2, 12, ANCHOR_DIM))
        self.assertEqual(denoising.cls_targets.shape, (2, 12, 2))

        encoded_sample0 = normalize_boxes3d(gt_boxes[0])
        for group in range(3):
            pos = slice(group * 4, group * 4 + 2)
            neg = slice(group * 4 + 2, group * 4 + 4)
            # Positives: clean targets, one-hot labels, full regression weights.
            self.assertTrue(torch.allclose(denoising.box_targets[0, pos], encoded_sample0))
            self.assertTrue(
                torch.allclose(denoising.box_weights[0, pos], _REG_WEIGHTS.expand(2, -1))
            )
            self.assertTrue(torch.equal(denoising.cls_targets[0, pos].argmax(dim=-1), gt_labels[0]))
            self.assertEqual(denoising.cls_targets[0, pos].sum(), 2.0)
            # Negatives: background targets, no box supervision, cls supervised.
            self.assertEqual(denoising.cls_targets[0, neg].sum(), 0.0)
            self.assertEqual(denoising.box_weights[0, neg].sum(), 0.0)
            self.assertTrue(torch.all(denoising.cls_weights[0, group * 4 : group * 4 + 4] == 1.0))
            # Sample 1 has one box: its second slot per half is padding.
            self.assertEqual(denoising.cls_weights[1, group * 4 + 1], 0.0)
            self.assertEqual(denoising.cls_weights[1, group * 4 + 3], 0.0)
            self.assertEqual(denoising.box_weights[1, group * 4 + 1].sum(), 0.0)

        # Noise bounds: z offsets stay within the configured half-extent fractions
        # and heading rotations within the yaw half-range.
        half_height = encoded_sample0[:, 5].exp() / 2.0
        for group in range(3):
            pos_anchor = denoising.anchor[0, group * 4 : group * 4 + 2]
            neg_anchor = denoising.anchor[0, group * 4 + 2 : group * 4 + 4]
            pos_dz = (pos_anchor[:, 2] - encoded_sample0[:, 2]).abs()
            neg_dz = (neg_anchor[:, 2] - encoded_sample0[:, 2]).abs()
            self.assertTrue(torch.all(pos_dz <= 0.5 * half_height + 1e-6))
            self.assertTrue(torch.all(neg_dz >= 0.5 * half_height - 1e-6))
            self.assertTrue(torch.all(neg_dz <= 1.0 * half_height + 1e-6))
            clean_yaw = torch.atan2(encoded_sample0[:, 6], encoded_sample0[:, 7])
            noised_yaw = torch.atan2(pos_anchor[:, 6], pos_anchor[:, 7])
            delta_yaw = torch.atan2(
                torch.sin(noised_yaw - clean_yaw), torch.cos(noised_yaw - clean_yaw)
            )
            self.assertTrue(torch.all(delta_yaw.abs() <= 0.39 + 1e-6))

        empty = build_denoising_queries(
            [torch.zeros((0, 9))],
            [torch.zeros((0,), dtype=torch.long)],
            num_query=8,
            num_classes=2,
            query_budget=12,
            reg_weights=_REG_WEIGHTS,
            **_NOISE_KWARGS,
        )
        self.assertEqual(empty.anchor.shape, (1, 0, ANCHOR_DIM))
        self.assertEqual(empty.attn_mask.shape, (8, 8))
        self.assertFalse(empty.attn_mask.any())

    def test_denoising_attention_mask_structure(self) -> None:
        mask = build_denoising_attention_mask(4, 2, 3, torch.device("cpu"))

        self.assertEqual(mask.shape, (10, 10))
        # Regular queries never read the denoising tail but see each other.
        self.assertTrue(torch.all(mask[:4, 4:]))
        self.assertFalse(torch.any(mask[:4, :4]))
        # Denoising groups read the regular queries and themselves only.
        self.assertFalse(torch.any(mask[4:7, :4]))
        self.assertFalse(torch.any(mask[4:7, 4:7]))
        self.assertTrue(torch.all(mask[4:7, 7:]))
        self.assertTrue(torch.all(mask[7:, 4:7]))
        self.assertFalse(torch.any(mask[7:, 7:]))

    def test_denoising_isolation_keeps_real_outputs_identical(self) -> None:
        head = build_sparse4d_head(
            make_anchor_recipe(), dn_query_budget=12, dropout=0.0, attn_drop=0.0
        )
        stages = _build_stages()
        gt = _build_gt_batch()

        with torch.no_grad():
            plain = forward_frame(head.eval(), stages, gt_detections=gt)
            head.reset()
            torch.manual_seed(0)
            denoised = forward_frame(head.train(), stages, gt_detections=gt)

        # The eval step runs the fused attention kernel, hence the float tolerance
        self.assertIsNone(plain.denoising)
        self.assertTrue(torch.allclose(plain.classification, denoised.classification, atol=1e-5))
        self.assertTrue(torch.allclose(plain.prediction, denoised.prediction, atol=1e-5))
        self.assertTrue(torch.allclose(plain.quality, denoised.quality, atol=1e-5))
        self.assertEqual(denoised.denoising.classification.shape, (2, 2, 12, 2))
        self.assertEqual(denoised.denoising.prediction.shape, (2, 2, 12, ANCHOR_DIM))

    def test_denoising_loss_terms(self) -> None:
        head = build_sparse4d_head(make_anchor_recipe(), dn_query_budget=12)
        gt = _build_gt_batch()

        torch.manual_seed(0)
        outputs = forward_frame(head, _build_stages(), gt_detections=gt)
        losses = head_loss(head, outputs, gt, [0, 1])

        for prefix in ("layer_0", "layer_-1"):
            self.assertTrue(torch.isfinite(losses[f"{prefix}_loss_dn_cls"]))
            self.assertTrue(torch.isfinite(losses[f"{prefix}_loss_dn_bbox"]))
            self.assertGreater(losses[f"{prefix}_loss_dn_bbox"], 0.0)
        # 3 groups x 3 boxes in the batch = 9 positive denoising queries.
        self.assertEqual(losses["dn_num_pos"], 9.0)
        dn_sum = sum(value for key, value in losses.items() if "loss_dn" in key)
        self.assertGreater(losses["loss"], dn_sum)

        losses.pop("loss").backward()
        self.assertIsNotNone(head.instance_bank.anchor.grad)

    def test_a_negative_denoising_query_skips_the_unannotated_class(self) -> None:
        """
        Input: denoising queries of a batch whose second sample does not annotate class 0.
        Expected: the negatives of that sample get no gradient toward class 0, its positives and the
        negatives of the first sample do.
        """
        head = build_sparse4d_head(make_anchor_recipe(), dn_query_budget=12)
        gt = _build_gt_batch()
        torch.manual_seed(0)
        outputs = forward_frame(head, _build_stages(), gt_detections=gt)
        annotated = torch.tensor([[True, True], [False, True]])

        loss = head_loss(head, outputs, gt, [0, 1], annotated)["layer_-1_loss_dn_cls"]
        gradient = torch.autograd.grad(loss, outputs.denoising.classification)[0][:, -1, :, 0]

        denoising = outputs.denoising
        positive = denoising.cls_targets.sum(dim=-1) > 0
        negative = ~positive & (denoising.cls_weights > 0)
        self.assertTrue(negative[1].any())
        self.assertTrue((gradient[1][negative[1]] == 0).all())
        self.assertTrue((gradient[1][positive[1]] != 0).all())
        self.assertTrue((gradient[0][negative[0]] != 0).all())

    def test_a_batch_without_boxes_reports_zero_denoising_terms(self) -> None:
        """
        Input: a training step whose samples carry no box.
        Expected: an empty denoising group with zero terms, the same terms as a step with boxes.
        """
        head = build_sparse4d_head(make_anchor_recipe(), dn_query_budget=12)
        gt = gt_batch([torch.zeros((0, 9))] * 2, [torch.zeros((0,), dtype=torch.long)] * 2)

        outputs = forward_frame(head, _build_stages(), gt_detections=gt)
        losses = head_loss(head, outputs, gt, [0, 1])

        self.assertEqual(outputs.denoising.classification.shape[2], 0)
        self.assertEqual(losses["layer_-1_loss_dn_cls"], 0.0)
        self.assertEqual(losses["layer_-1_loss_dn_bbox"], 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_denoising_runs_with_temporal_memory(self) -> None:
        head = build_sparse4d_head(make_anchor_recipe(), num_temp_instances=2, dn_query_budget=12)
        gt = _build_gt_batch()

        for frame_index in range(2):
            outputs = forward_frame(head, _build_stages(), 10.0 + 0.1 * frame_index, gt)
            losses = head_loss(head, outputs, gt, [0, 1])
            self.assertIsNotNone(outputs.denoising)
            self.assertTrue(torch.isfinite(losses["loss"]))
            losses["loss"].backward()

        # The temporal state holds regular queries only, never the denoising tail
        self.assertEqual(head.instance_bank.lane_continues.tolist(), [True, True])
        self.assertEqual(head.instance_bank._cached_feature.shape, (2, 2, 32))
        self.assertLess(
            int(head.instance_bank.cached_query_index.max()), head.instance_bank.num_anchor
        )

    def test_denoising_masks_unknown_velocity(self) -> None:
        gt_boxes, gt_labels = _build_gt()
        gt_boxes[1][0, 7:9] = float("nan")
        denoising = build_denoising_queries(
            gt_boxes,
            gt_labels,
            num_query=8,
            num_classes=2,
            query_budget=12,
            reg_weights=_REG_WEIGHTS,
            **_NOISE_KWARGS,
        )

        self.assertTrue(torch.isfinite(denoising.anchor).all())
        self.assertTrue(torch.isfinite(denoising.box_targets).all())
        self.assertTrue(torch.all(denoising.box_weights[1, 0, 8:] == 0.0))
        self.assertTrue(torch.all(denoising.box_weights[1, 0, :8] == _REG_WEIGHTS[:8]))
