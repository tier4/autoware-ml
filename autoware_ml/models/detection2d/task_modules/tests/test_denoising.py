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
    ContrastiveDenoisingQueries,
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
        self,
        box_counts: tuple[int, ...],
        pad_width: int | None = None,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
    ) -> tuple[ContrastiveDenoisingQueries, list[torch.Tensor], list[torch.Tensor]]:
        """Build the queries of images holding the given box counts.

        The ground truth is padded as it is collated: labels -1 and zero boxes past the valid
        count, up to ``pad_width``, by default two rows wider than the largest count. The boxes
        are 0.2 x 0.1 and centered away from the border, so the noise is never clipped; the
        labels cycle through the classes. Also returns the valid labels and boxes per image.
        """
        if pad_width is None:
            pad_width = max(box_counts) + 2
        labels = [torch.arange(n) % NUM_CLASSES for n in box_counts]
        boxes = [
            torch.tensor([[0.3 + 0.05 * i, 0.4 + 0.02 * i, 0.2, 0.1] for i in range(n)]).reshape(
                n, 4
            )
            for n in box_counts
        ]
        padded_labels = torch.full((len(box_counts), pad_width), -1, dtype=torch.int32)
        padded_boxes = torch.zeros(len(box_counts), pad_width, 4)
        for image, count in enumerate(box_counts):
            padded_labels[image, :count] = labels[image]
            padded_boxes[image, :count] = boxes[image]
        queries = build_contrastive_denoising_queries(
            padded_labels,
            padded_boxes,
            torch.tensor(box_counts, dtype=torch.int32),
            NUM_CLASSES,
            NUM_MATCHING_QUERIES,
            self.class_embed,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
        )
        return queries, labels, boxes

    def test_groups_are_sized_by_the_image_with_the_most_boxes(self) -> None:
        """Test that the groups are sized by the image with the most boxes.

        Input: two images with 3 and 1 boxes, ``num_denoising=20``.
        Expected: 20 // 3 = 6 groups, each of 3 positives then 3 negatives, so 36 queries per
        image. The positives of image 0 are the first 3 queries of each group, the one of
        image 1 is the first query of each group.
        Check: ``num_groups``, the query count, the content shape and the positive indices.
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

    def test_noise_moves_positives_by_up_to_half_and_negatives_by_half_to_the_full_size(
        self,
    ) -> None:
        """Test that positives move by up to half and negatives by half to the full box size.

        Input: one image with 4 boxes of 0.2 x 0.1, label noise off, 10 groups.
        Expected: each corner of a positive moves by at most half the box size along its axis,
        each corner of a negative by half to the full box size. A negative whose corners
        cross gets a negative width or height, which the inverse sigmoid clamps to about
        zero, so such negatives are skipped.
        Check: the shift of every corner, in units of the box size.
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

    def test_content_is_the_label_embedding_and_zero_for_padding(self) -> None:
        """Test that the content is the label embedding, and zero for padding.

        Input: two images with 2 and 1 boxes, label noise off.
        Expected: each query holds the embedding of the label of its box; the padding query
        of image 1 holds zeros.
        Check: the content of the first group against the label embedding.
        """
        queries, labels, _ = self._build((2, 1), label_noise_ratio=0.0)

        padded = torch.tensor([[0, 1, 0, 1], [0, NUM_CLASSES, 0, NUM_CLASSES]])
        self.assertEqual(labels[0].tolist(), [0, 1])
        torch.testing.assert_close(queries.content[:, :4], self.class_embed(padded))
        self.assertEqual(queries.content[1, 1].abs().sum().item(), 0.0)

    def test_attention_mask_separates_the_groups_from_each_other_and_from_matching(
        self,
    ) -> None:
        """Test that the attention mask separates the groups from each other and from matching.

        Input: one image with 2 boxes, ``num_denoising=4``: 2 groups of 4 queries, then 10
        matching queries.
        Expected: a denoising query sees its own group and the matching queries; a matching
        query sees only the matching queries.
        Check: the mask against the expected blocks; True blocks attention.
        """
        queries, _, _ = self._build((2,), num_denoising=4)

        visible = torch.zeros(18, 18, dtype=torch.bool)
        visible[:4, :4] = True
        visible[4:8, 4:8] = True
        visible[8:, 8:] = True
        visible[:8, 8:] = True
        torch.testing.assert_close(queries.attn_mask, ~visible)

    def test_attention_mask_hides_both_neighbors_from_a_middle_group(self) -> None:
        """Test that a middle group sees neither of its neighboring groups.

        Input: one image with 2 boxes, ``num_denoising=6``: 3 groups of 4 queries, then 10
        matching queries.
        Expected: the middle group sees neither the group before it nor the one after it;
        every group sees itself and the matching queries.
        Check: the mask against the expected blocks; True blocks attention.
        """
        queries, _, _ = self._build((2,), num_denoising=6)

        self.assertEqual(queries.num_groups, 3)
        visible = torch.zeros(22, 22, dtype=torch.bool)
        for group in range(3):
            rows = slice(4 * group, 4 * (group + 1))
            visible[rows, rows] = True
        visible[:12, 12:] = True
        visible[12:, 12:] = True
        torch.testing.assert_close(queries.attn_mask, ~visible)

    def test_batch_without_boxes_gets_one_dummy_pair(self) -> None:
        """Test that a batch without boxes gets one dummy pair.

        Input: two images without boxes, box noise off.
        Expected: two dummy queries with the box (0.5, 0.5, 0.1, 0.1), as in the official code,
        and no positive index; the label embedding still gets a gradient.
        Check: the query counts, the positive indices, the boxes after the sigmoid and the
        gradient of the label embedding.
        """
        queries, _, _ = self._build((0, 0), box_noise_scale=0.0)
        queries.content.sum().backward()

        self.assertEqual((queries.num_groups, queries.num_queries), (1, 2))
        self.assertEqual([len(index) for index in queries.positive_indices], [0, 0])
        torch.testing.assert_close(
            queries.boxes_unact.sigmoid(), torch.tensor([0.5, 0.5, 0.1, 0.1]).expand(2, 2, 4)
        )
        self.assertIsNotNone(self.class_embed.weight.grad)

    def test_queries_do_not_depend_on_the_padded_width(self) -> None:
        """Test that the queries do not depend on the padded width of the ground truth.

        Input: two images with 3 and 1 boxes, padded once to 3 and once to 50 rows, each with
        the same seed.
        Expected: the queries are laid out by the largest valid count, so both paddings give
        the same groups, query count, noise and positive indices.
        Check: every field of the two results.
        """
        torch.manual_seed(1)
        narrow, _, _ = self._build((3, 1), pad_width=3)
        torch.manual_seed(1)
        wide, _, _ = self._build((3, 1), pad_width=50)

        self.assertEqual((narrow.num_groups, narrow.num_queries), (33, 198))
        self.assertEqual((wide.num_groups, wide.num_queries), (33, 198))
        torch.testing.assert_close(wide.content, narrow.content, rtol=0, atol=0)
        torch.testing.assert_close(wide.boxes_unact, narrow.boxes_unact, rtol=0, atol=0)
        torch.testing.assert_close(wide.attn_mask, narrow.attn_mask)
        for wide_index, narrow_index in zip(
            wide.positive_indices, narrow.positive_indices, strict=True
        ):
            torch.testing.assert_close(wide_index, narrow_index)

    def test_seeded_queries_match_the_official_code(self) -> None:
        """Test that a seeded run gives the queries of the official code.

        Input: two images with 2 and 1 boxes, ``num_denoising=4`` (2 groups),
        ``label_noise_ratio=1.0`` so that labels get replaced, seed 1.
        Expected: the labels and noised boxes that lyuwenyu/RT-DETR@df2a96c
        (``get_contrastive_denoising_training_group``) gives for the same inputs and seed;
        padding queries keep the padding label and a zero box. Reordering the label or box
        draws changes them.
        Check: the content against the embedding of the expected labels, and the boxes of
        the first group after the sigmoid.
        """
        torch.manual_seed(1)
        queries, _, _ = self._build((2, 1), num_denoising=4, label_noise_ratio=1.0)

        pad = NUM_CLASSES
        labels = torch.tensor([[0, 1, 2, 1, 0, 1, 1, 1], [0, pad, 0, pad, 0, pad, 3, pad]])
        torch.testing.assert_close(queries.content, self.class_embed(labels))
        boxes = queries.boxes_unact.sigmoid()
        torch.testing.assert_close(
            boxes[0, :4],
            torch.tensor(
                [
                    [0.3479, 0.4235, 0.1617, 0.0880],
                    [0.3588, 0.3919, 0.0260, 0.0845],
                    [0.4578, 0.4650, 0.2803, 0.1226],
                    [0.3591, 0.5069, 0.0000, 0.1135],
                ]
            ),
            rtol=0,
            atol=1e-4,
        )
        torch.testing.assert_close(
            boxes[1, :4],
            torch.tensor(
                [
                    [0.3489, 0.4393, 0.1401, 0.0822],
                    [0.0000, 0.0000, 0.0000, 0.0000],
                    [0.1513, 0.3323, 0.2658, 0.1275],
                    [0.0000, 0.0000, 0.0000, 0.0000],
                ]
            ),
            rtol=0,
            atol=1e-4,
        )

    def test_rejects_zero_num_denoising(self) -> None:
        """Test that a non-positive ``num_denoising`` is refused.

        Input: ``num_denoising=0``.
        Expected: refused.
        Check: a ValueError is raised.
        """
        with self.assertRaises(ValueError):
            self._build((1, 1), num_denoising=0)
