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

"""Unit tests for the contrastive denoising queries of RT-DETR."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from autoware_ml.models.detection2d.task_modules.denoising import (
    DenoisingQueries,
    build_contrastive_denoising_queries,
)

NUM_CLASSES = 5
NUM_MATCHING_QUERIES = 10


class TestBuildContrastiveDenoisingQueries(unittest.TestCase):
    """The layout, noise and attention mask of the denoising queries."""

    def setUp(self) -> None:
        """Build a seeded label embedding of 5 classes plus the padding entry."""
        torch.manual_seed(0)
        self.class_embed = nn.Embedding(NUM_CLASSES + 1, 8, padding_idx=NUM_CLASSES)

    def _build(
        self, box_counts: tuple[int, ...], **kwargs: float | int
    ) -> tuple[DenoisingQueries, list[torch.Tensor], list[torch.Tensor]]:
        """Build the queries of images holding the given box counts.

        The boxes are 0.2 x 0.1 and centered away from the border, so the noise is never
        clipped; the labels cycle through the classes.
        """
        labels = [torch.arange(n) % NUM_CLASSES for n in box_counts]
        boxes = [
            torch.tensor([[0.3 + 0.05 * i, 0.4 + 0.02 * i, 0.2, 0.1] for i in range(n)]).reshape(
                n, 4
            )
            for n in box_counts
        ]
        queries = build_contrastive_denoising_queries(
            labels, boxes, NUM_CLASSES, NUM_MATCHING_QUERIES, self.class_embed, **kwargs
        )
        return queries, labels, boxes

    def test_groups_hold_a_positive_and_a_negative_per_box_of_the_fullest_image(self) -> None:
        """
        Input: images with 3 and 1 boxes and a target of 20 denoising queries.
        Expected: 20 // 3 = 6 groups of 3 positives then 3 negatives, 36 queries; the
            positives of each image are its boxes in every group.
        Check: num_groups, the query count, the content shape and the positive indices.
        """
        queries, _, _ = self._build((3, 1), num_denoising=20)

        self.assertEqual(queries.num_groups, 6)
        self.assertEqual(queries.num_queries, 36)
        self.assertEqual(tuple(queries.content.shape), (2, 36, 8))
        self.assertEqual(
            queries.positive_indices[0].tolist(),
            [6 * group + box for group in range(6) for box in range(3)],
        )
        self.assertEqual(queries.positive_indices[1].tolist(), [6 * group for group in range(6)])

    def test_noise_moves_positives_by_up_to_half_and_negatives_by_half_to_once_the_size(
        self,
    ) -> None:
        """
        Input: one image with 4 boxes of 0.2 x 0.1, box noise only, 10 groups.
        Expected: every corner of a positive moves by at most half the box size along its
            axis, every corner of a negative by between half and once. (A negative whose
            corners cross gets a negative size, which the inverse sigmoid clamps, so only the
            negatives that keep their orientation are compared.)
        Check: the corner shifts of every query against its box.
        """
        queries, _, boxes = self._build((4,), label_noise_ratio=0.0, num_denoising=40)

        noised = torch.sigmoid(queries.boxes_unact[0]).reshape(queries.num_groups, 2, 4, 4)
        corners = torch.concat([boxes[0][:, :2] - boxes[0][:, 2:] / 2] * 2, dim=-1)
        corners[:, 2:] += boxes[0][:, 2:]
        noised_corners = torch.concat(
            [noised[..., :2] - noised[..., 2:] / 2, noised[..., :2] + noised[..., 2:] / 2], dim=-1
        )
        shift = (noised_corners - corners).abs() / boxes[0][:, 2:].tile(1, 2)
        upright = (noised[:, 1, :, 2:] > 1e-3).all(-1)
        negative_shift = shift[:, 1][upright]
        self.assertTrue((shift[:, 0] <= 0.5 + 1e-4).all())
        self.assertGreater(len(negative_shift), 10)
        self.assertTrue((negative_shift >= 0.5 - 1e-4).all())
        self.assertTrue((negative_shift <= 1.0 + 1e-4).all())

    def test_without_label_noise_the_content_embeds_the_labels_and_pads_with_zeros(
        self,
    ) -> None:
        """
        Input: images with 2 and 1 boxes, label noise off.
        Expected: each query embeds the label of its box; the padding slot of the second
            image embeds the padding class, a zero vector.
        Check: the content of the first group against the label embedding.
        """
        queries, labels, _ = self._build((2, 1), label_noise_ratio=0.0)

        padded = torch.tensor([[0, 1, 0, 1], [0, NUM_CLASSES, 0, NUM_CLASSES]])
        self.assertEqual(labels[0].tolist(), [0, 1])
        torch.testing.assert_close(queries.content[:, :4], self.class_embed(padded))
        self.assertEqual(queries.content[1, 1].abs().sum().item(), 0.0)

    def test_attention_mask_isolates_the_groups_and_hides_them_from_matching(self) -> None:
        """
        Input: one image with 2 boxes and a target of 4 denoising queries: 2 groups of a
            positive and a negative per box, then 10 matching queries.
        Expected: a denoising query sees its own group and the matching queries; a matching
            query sees the matching queries only.
        Check: the mask against the block layout, True where attention is blocked.
        """
        queries, _, _ = self._build((2,), num_denoising=4)

        visible = torch.zeros(18, 18, dtype=torch.bool)
        visible[:4, :4] = True
        visible[4:8, 4:8] = True
        visible[8:, 8:] = True
        visible[:8, 8:] = True
        torch.testing.assert_close(queries.attn_mask, ~visible)

    def test_batch_without_boxes_gets_one_finite_dummy_pair_without_positives(self) -> None:
        """
        Input: two images without boxes.
        Expected: one group of one dummy positive and one negative, no positive index, and
            finite boxes, so the label embedding stays in the graph (lyuwenyu/RT-DETR#684).
        Check: the counts, the positive indices, the finiteness and the gradient of the
            label embedding.
        """
        queries, _, _ = self._build((0, 0))
        queries.content.sum().backward()

        self.assertEqual((queries.num_groups, queries.num_queries), (1, 2))
        self.assertEqual([len(index) for index in queries.positive_indices], [0, 0])
        self.assertTrue(torch.isfinite(queries.boxes_unact).all())
        self.assertIsNotNone(self.class_embed.weight.grad)

    def test_rejects_a_non_positive_target_and_mismatched_images(self) -> None:
        """
        Input: a target of 0 denoising queries, and labels for two images with boxes for one.
        Expected: each is refused.
        Check: the ValueError.
        """
        labels = [torch.tensor([0]), torch.tensor([1])]
        boxes = [torch.tensor([[0.5, 0.5, 0.1, 0.1]])] * 2
        for kwargs in (
            {"gt_labels": labels, "gt_boxes": boxes, "num_denoising": 0},
            {"gt_labels": labels, "gt_boxes": boxes[:1]},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                build_contrastive_denoising_queries(
                    num_classes=NUM_CLASSES,
                    num_matching_queries=NUM_MATCHING_QUERIES,
                    class_embed=self.class_embed,
                    **kwargs,
                )
