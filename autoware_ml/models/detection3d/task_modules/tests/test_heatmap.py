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

"""Unit tests for heatmap utilities."""

import math
import unittest
from typing import Sequence

from jaxtyping import Bool, Float32, Int64
import torch

from autoware_ml.models.detection3d.task_modules.heatmap import (
    _vectorize_gaussian2d,
    vectorize_gaussian_radii,
    create_gaussian_heatmaps,
    create_oriented_gaussian_heatmaps,
    draw_heatmap_gaussian_oriented,
    batch_circle_nms,
)


class TestVectorizeGaussianRadii(unittest.TestCase):
    """Unit tests for the vectorize_gaussian_radii function."""

    def setUp(self) -> None:
        """Set up the same input tensors for all tests."""
        # (batch_size, 3)
        self.widths = torch.tensor([[1.0, 2.0, 3.2], [3.0, 4.4, 2.8]])
        self.heights = torch.tensor([[1.0, 2.0, 3.2], [3.0, 4.8, 5.0]])
        self.min_overlap = 0.1

    def test_vectorize_gaussian_radii(self) -> None:
        """Test the vectorize_gaussian_radii function."""
        gaussian_radii = vectorize_gaussian_radii(
            widths=self.widths,
            heights=self.heights,
            min_overlap=self.min_overlap,
        )

        self.assertEqual(gaussian_radii.shape, self.widths.shape)
        expected_radii = torch.tensor([[0, 0, 1], [1, 1, 1]], dtype=torch.int32)
        self.assertTrue(torch.allclose(gaussian_radii, expected_radii))


class TestVectorizeGaussian2D(unittest.TestCase):
    """Unit tests for the _vectorize_gaussian2d function."""

    def setUp(self) -> None:
        """Set up the same input tensors for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # (batch_size, 3)
        self.widths = torch.tensor([[1, 2, 4], [3, 4, 5]], device=self.device, dtype=torch.int32)
        self.heights = torch.tensor([[1, 2, 4], [3, 4, 5]], device=self.device, dtype=torch.int32)
        self.min_overlap = 0.1
        # (batch_size, max_num_boxes, max_height, max_width)
        self.expected = torch.zeros(2, 3, 5, 5, device=self.device, dtype=torch.float32)

        # batch 0, box 0: 1x1
        self.expected[0, 0, :1, :1] = torch.tensor([[1.0]], device=self.device, dtype=torch.float32)

        # batch 0, box 1: 2x2
        self.expected[0, 1, :2, :2] = torch.tensor(
            [
                [0.0019, 0.0019],
                [0.0019, 0.0019],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        # batch 0, box 2: 4x4
        self.expected[0, 2, :4, :4] = torch.tensor(
            [
                [0.0000e00, 9.2925e-07, 9.2925e-07, 0.0000e00],
                [9.2925e-07, 6.2177e-02, 6.2177e-02, 9.2925e-07],
                [9.2925e-07, 6.2177e-02, 6.2177e-02, 9.2925e-07],
                [0.0000e00, 9.2925e-07, 9.2925e-07, 0.0000e00],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        # batch 1, box 0: 3x3
        self.expected[1, 0, :3, :3] = torch.tensor(
            [
                [1.4945e-05, 3.8659e-03, 1.4945e-05],
                [3.8659e-03, 1.0000e00, 3.8659e-03],
                [1.4945e-05, 3.8659e-03, 1.4945e-05],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        # batch 1, box 1: 4x4
        self.expected[1, 1, :4, :4] = torch.tensor(
            [
                [7.8115e-07, 4.0465e-04, 4.0465e-04, 7.8115e-07],
                [4.0465e-04, 2.0961e-01, 2.0961e-01, 4.0465e-04],
                [4.0465e-04, 2.0961e-01, 2.0961e-01, 4.0465e-04],
                [7.8115e-07, 4.0465e-04, 4.0465e-04, 7.8115e-07],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        # batch 1, box 2: 5x5
        self.expected[1, 2] = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.7267e-06, 0.0, 0.0],
                [0.0, 3.7267e-06, 1.0000e00, 3.7267e-06, 0.0],
                [0.0, 0.0, 3.7267e-06, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def test_vectorize_gaussian2d(self) -> None:
        """Test the _vectorize_gaussian2d function with 5x5 output but different sigma for each inputs."""
        gaussian_2d = _vectorize_gaussian2d(
            heights=self.heights,
            widths=self.widths,
            sigmas=torch.tensor(
                [[0.1, 0.2, 0.3], [0.3, 0.4, 0.2]], device=self.device, dtype=torch.float32
            ),  # Example sigmas
            valid_masks=torch.tensor([[1, 1, 1], [1, 1, 1]], device=self.device, dtype=torch.bool),
            device=self.device,
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(gaussian_2d, self.expected, atol=1e-4))
        self.assertEqual(gaussian_2d.shape[:2], self.widths.shape)

    def test_vectorize_gaussian2d_invalid_mask(self) -> None:
        """Test the _vectorize_gaussian2d function with 5x5 output but the last box is invalid."""
        gaussian_2d = _vectorize_gaussian2d(
            heights=self.heights,
            widths=self.widths,
            sigmas=torch.tensor(
                [[0.1, 0.2, 0.3], [0.3, 0.4, 0.2]], device=self.device, dtype=torch.float32
            ),  # Example sigmas
            valid_masks=torch.tensor([[1, 1, 1], [1, 1, 0]], device=self.device, dtype=torch.bool),
            device=self.device,
            dtype=torch.float32,
        )

        # batch 1, box 2: 5x5
        # Set to zeros since valid_masks indicates this box is invalid
        self.expected[1, 2] = torch.zeros((5, 5), device=self.device, dtype=torch.float32)
        self.assertTrue(torch.allclose(gaussian_2d, self.expected, atol=1e-4))
        self.assertEqual(gaussian_2d.shape[:2], self.widths.shape)


class TestCreateGaussianHeatmap(unittest.TestCase):
    """Unit tests for the create_gaussian_heatmap function."""

    def setUp(self) -> None:
        """Set up the same input tensors for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.heatmap_width = 8
        self.heatmap_height = 8
        self.num_classes = 5
        self.batch_size = 2
        self.gt_bboxes_labels = torch.tensor(
            [[0, 0, 2], [1, 2, 2]], device=self.device, dtype=torch.int64
        )
        # (batch_size, 3, 2)
        self.centers = torch.tensor(
            [[[1, 2], [3, 4], [5, 6]], [[2, 3], [4, 5], [6, 7]]],
            device=self.device,
            dtype=torch.int64,
        )
        self.gaussian_radii = torch.tensor(
            [[1, 2, 1], [4, 1, 3]], device=self.device, dtype=torch.int32
        )
        self.expected_heatmap = torch.zeros(
            self.batch_size,
            self.num_classes,
            self.heatmap_height,
            self.heatmap_width,
            device=self.device,
            dtype=torch.float32,
        )

        # Manually draw the expected heatmaps for each valid box in the batch
        # Batch 0, Class 0, center (1, 2) and radius 1, center (3, 4) and radius 2
        self.expected_heatmap[0, 0] = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0183, 0.1353, 0.0183, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.1353, 1.0000, 0.1353, 0.0561, 0.0273, 0.0032, 0.0000, 0.0000],
                [0.0183, 0.1353, 0.2369, 0.4868, 0.2369, 0.0273, 0.0000, 0.0000],
                [0.0000, 0.0561, 0.4868, 1.0000, 0.4868, 0.0561, 0.0000, 0.0000],
                [0.0000, 0.0273, 0.2369, 0.4868, 0.2369, 0.0273, 0.0000, 0.0000],
                [0.0000, 0.0032, 0.0273, 0.0561, 0.0273, 0.0032, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Batch 0, Class 2, center (5, 6) and radius 1
        self.expected_heatmap[0, 2, 5:8, :] = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0183, 0.1353, 0.0183, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.1353, 1.0000, 0.1353, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0183, 0.1353, 0.0183, 0.0000],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Batch 1, Class 1, center (2, 3) and radius 4
        self.expected_heatmap[1, 1] = torch.tensor(
            [
                [
                    5.5638e-02,
                    1.0837e-01,
                    1.3534e-01,
                    1.0837e-01,
                    5.5638e-02,
                    1.8316e-02,
                    3.8659e-03,
                    0.0000,
                ],
                [
                    1.6901e-01,
                    3.2919e-01,
                    4.1111e-01,
                    3.2919e-01,
                    1.6901e-01,
                    5.5638e-02,
                    1.1744e-02,
                    0.0000,
                ],
                [
                    3.2919e-01,
                    6.4118e-01,
                    8.0074e-01,
                    6.4118e-01,
                    3.2919e-01,
                    1.0837e-01,
                    2.2873e-02,
                    0.0000,
                ],
                [
                    4.1111e-01,
                    8.0074e-01,
                    1.0000e00,
                    8.0074e-01,
                    4.1111e-01,
                    1.3534e-01,
                    2.8566e-02,
                    0.0000,
                ],
                [
                    3.2919e-01,
                    6.4118e-01,
                    8.0074e-01,
                    6.4118e-01,
                    3.2919e-01,
                    1.0837e-01,
                    2.2873e-02,
                    0.0000,
                ],
                [
                    1.6901e-01,
                    3.2919e-01,
                    4.1111e-01,
                    3.2919e-01,
                    1.6901e-01,
                    5.5638e-02,
                    1.1744e-02,
                    0.0000,
                ],
                [
                    5.5638e-02,
                    1.0837e-01,
                    1.3534e-01,
                    1.0837e-01,
                    5.5638e-02,
                    1.8316e-02,
                    3.8659e-03,
                    0.0000,
                ],
                [
                    1.1744e-02,
                    2.2873e-02,
                    2.8566e-02,
                    2.2873e-02,
                    1.1744e-02,
                    3.8659e-03,
                    8.1599e-04,
                    0.0000,
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Batch 1, Class 2, center (4, 5) and radius 1, center (6, 7) and radius 3
        self.expected_heatmap[1, 2, 4:8, 3:8] = torch.tensor(
            [
                [0.0183, 0.1353, 0.0254, 0.0367, 0.0254],
                [0.1353, 1.0000, 0.1593, 0.2301, 0.1593],
                [0.0254, 0.1593, 0.4797, 0.6926, 0.4797],
                [0.0367, 0.2301, 0.6926, 1.0000, 0.6926],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def test_create_gaussian_heatmaps(self) -> None:
        """Test create_gaussian_heatmap function with 8x8 output but different sigma for each inputs."""
        gaussian_heatmaps = create_gaussian_heatmaps(
            heatmap_width=self.heatmap_width,
            heatmap_height=self.heatmap_height,
            num_classes=self.num_classes,
            centers=self.centers,
            gaussian_radii=self.gaussian_radii,
            gt_bboxes_labels=self.gt_bboxes_labels,
            valid_masks=torch.tensor([[1, 1, 1], [1, 1, 1]], device=self.device, dtype=torch.bool),
            device=self.device,
        )

        self.assertEqual(
            gaussian_heatmaps.shape,
            (self.batch_size, self.num_classes, self.heatmap_height, self.heatmap_width),
        )
        self.assertTrue(torch.allclose(gaussian_heatmaps, self.expected_heatmap, atol=1e-4))

    def test_create_gaussian_heatmaps_with_invalid_mask(self) -> None:
        """Test create_gaussian_heatmap function with 8x8 output with invalid bboxes."""
        gaussian_heatmaps = create_gaussian_heatmaps(
            heatmap_width=self.heatmap_width,
            heatmap_height=self.heatmap_height,
            num_classes=self.num_classes,
            centers=self.centers,
            gaussian_radii=self.gaussian_radii,
            gt_bboxes_labels=self.gt_bboxes_labels,
            valid_masks=torch.tensor([[1, 0, 1], [0, 1, 1]], device=self.device, dtype=torch.bool),
            device=self.device,
        )

        self.assertEqual(
            gaussian_heatmaps.shape,
            (self.batch_size, self.num_classes, self.heatmap_height, self.heatmap_width),
        )
        # For batch 1, the first box is invalid, so the heatmap for class 1 should be all zeros
        self.expected_heatmap[1, 1] = torch.zeros(
            (self.heatmap_height, self.heatmap_width), device=self.device, dtype=torch.float32
        )
        # For batch 0, the second box is invalid, so the heatmap for class 0 should only have
        # the first box's contribution, where 1.0 from the second box (center at 3, 4) is removed
        self.expected_heatmap[
            0,
            0,
        ] = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0183, 0.1353, 0.0183, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.1353, 1.0000, 0.1353, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0183, 0.1353, 0.0183, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.assertTrue(torch.allclose(gaussian_heatmaps, self.expected_heatmap, atol=1e-4))

    def test_create_gaussian_heatmaps_ignores_padded_radii(self) -> None:
        """Test that radii of invalid bboxes do not change the heatmap nor the kernel size."""
        valid_masks = torch.tensor([[1, 0, 1], [0, 1, 1]], device=self.device, dtype=torch.bool)
        # The padded boxes carry radii far larger and smaller than any valid box, so a leaking
        # padded radius would blow up (or collapse) the shared kernel size.
        padded_gaussian_radii = self.gaussian_radii.clone()
        padded_gaussian_radii[0, 1] = 100
        padded_gaussian_radii[1, 0] = -7

        gaussian_heatmaps = create_gaussian_heatmaps(
            heatmap_width=self.heatmap_width,
            heatmap_height=self.heatmap_height,
            num_classes=self.num_classes,
            centers=self.centers,
            gaussian_radii=padded_gaussian_radii,
            gt_bboxes_labels=self.gt_bboxes_labels,
            valid_masks=valid_masks,
            device=self.device,
        )
        expected_heatmaps = create_gaussian_heatmaps(
            heatmap_width=self.heatmap_width,
            heatmap_height=self.heatmap_height,
            num_classes=self.num_classes,
            centers=self.centers,
            gaussian_radii=self.gaussian_radii,
            gt_bboxes_labels=self.gt_bboxes_labels,
            valid_masks=valid_masks,
            device=self.device,
        )
        self.assertTrue(torch.allclose(gaussian_heatmaps, expected_heatmaps, atol=1e-4))

    def test_create_gaussian_heatmaps_with_all_invalid_masks(self) -> None:
        """Test that an all-invalid batch with negative padded radii yields an empty heatmap."""
        gaussian_heatmaps = create_gaussian_heatmaps(
            heatmap_width=self.heatmap_width,
            heatmap_height=self.heatmap_height,
            num_classes=self.num_classes,
            centers=self.centers,
            gaussian_radii=torch.full_like(self.gaussian_radii, -1),
            gt_bboxes_labels=self.gt_bboxes_labels,
            valid_masks=torch.zeros_like(self.gt_bboxes_labels, dtype=torch.bool),
            device=self.device,
        )

        self.assertEqual(
            gaussian_heatmaps.shape,
            (self.batch_size, self.num_classes, self.heatmap_height, self.heatmap_width),
        )
        self.assertTrue(torch.all(gaussian_heatmaps == 0.0))


class TestBatchCircleNMS(unittest.TestCase):
    """Unit tests for the batch_circle_nms function."""

    def setUp(self) -> None:
        """Set up the common device, batch shape and NMS parameters for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 2
        self.num_classes = 3
        # batch_circle_nms takes one value per class channel, never a scalar, so the shared
        # defaults are spelled out per class here.
        self.min_radii = [1.0] * self.num_classes
        self.post_max_sizes = [10] * self.num_classes

    def _build_inputs(
        self,
        centers: Sequence[Sequence[float]],
        scores: Sequence[float],
        valid_masks: Sequence[bool] | None = None,
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_classes max_num_bboxes 2"],
        Float32[torch.Tensor, "batch_size num_classes max_num_bboxes"],
        Float32[torch.Tensor, "batch_size num_classes max_num_bboxes"],
    ]:
        """
        Repeat one row of centers and scores across every sample and class, so each test spells
        out a single scenario while still exercising the full
        (batch_size, num_classes, max_num_bboxes) layout. Placing identical geometry in every row
        also means suppression leaking across rows would remove boxes that must survive.
        """
        max_num_bboxes = len(scores)
        shape = (self.batch_size, self.num_classes, max_num_bboxes)
        bboxes_centers = (
            torch.tensor(centers, dtype=torch.float32, device=self.device)
            .view(1, 1, max_num_bboxes, 2)
            .expand(*shape, 2)
            .contiguous()
        )
        bboxes_scores = (
            torch.tensor(scores, dtype=torch.float32, device=self.device)
            .view(1, 1, max_num_bboxes)
            .expand(shape)
            .contiguous()
        )
        valid_bboxes_masks = (
            torch.tensor(
                valid_masks if valid_masks is not None else [True] * max_num_bboxes,
                dtype=torch.bool,
                device=self.device,
            )
            .view(1, 1, max_num_bboxes)
            .expand(shape)
            .contiguous()
        )
        return (bboxes_centers, bboxes_scores, valid_bboxes_masks)

    def _expected_keep_masks(
        self, keep_masks_row: Sequence[bool]
    ) -> Bool[torch.Tensor, "batch_size num_classes max_num_bboxes"]:
        """Repeat one row of expected results across every sample and class."""
        return (
            torch.tensor(keep_masks_row, dtype=torch.bool, device=self.device)
            .view(1, 1, len(keep_masks_row))
            .expand(self.batch_size, self.num_classes, len(keep_masks_row))
            .contiguous()
        )

    def _expected_keep_masks_per_class(
        self, keep_masks_rows: Sequence[Sequence[bool]]
    ) -> Bool[torch.Tensor, "batch_size num_classes max_num_bboxes"]:
        """Repeat one row of expected results per class across every sample."""
        return (
            torch.tensor(keep_masks_rows, dtype=torch.bool, device=self.device)
            .view(1, self.num_classes, len(keep_masks_rows[0]))
            .expand(self.batch_size, self.num_classes, len(keep_masks_rows[0]))
            .contiguous()
        )

    def test_batch_circle_nms_suppresses_neighbours_within_radius(self) -> None:
        """Test that a lower scoring box within min_radius of a kept box is suppressed."""
        # Two tight pairs 5 m apart: the second box of each pair falls inside min_radius
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [0.5, 0.0], [5.0, 0.0], [5.4, 0.0]],
            scores=[0.9, 0.8, 0.7, 0.6],
        )

        keep_masks = batch_circle_nms(
            bboxes_centers=centers,
            scores=scores,
            min_radii=self.min_radii,
            valid_bboxes_masks=valid_masks,
            post_max_sizes=self.post_max_sizes,
        )

        # Second and fourth boxes are suppressed by the first and third boxes in each
        # batch and classes, respectively
        expected_keep_masks = self._expected_keep_masks([True, False, True, False])
        self.assertTrue(torch.equal(keep_masks, expected_keep_masks))

    def test_batch_circle_nms_keeps_box_suppressed_only_by_a_removed_box(self) -> None:
        """
        Test the greedy chain A -> B -> C, where A suppresses B and B would have suppressed C.
        Because B is already gone it cannot suppress anything, so C survives.
        """
        # Collinear boxes 1.5 m apart with min_radius 2.0: A-B and B-C overlap, A-C does not
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]],
            scores=[0.9, 0.8, 0.7],
        )

        keep_masks = batch_circle_nms(
            bboxes_centers=centers,
            scores=scores,
            min_radii=[2.0] * self.num_classes,
            valid_bboxes_masks=valid_masks,
            post_max_sizes=self.post_max_sizes,
        )

        # The middle box is suppressed by the first box, but the last box survives because it is
        # only suppressed by the middle box which is already gone
        expected_keep_masks = self._expected_keep_masks([True, False, True])
        self.assertTrue(torch.equal(keep_masks, expected_keep_masks))

    def test_batch_circle_nms_invalid_boxes_neither_kept_nor_suppressing(self) -> None:
        """
        Test that an invalid box is dropped and does not suppress its neighbour, so the
        neighbour survives even though it sits inside the invalid box's radius.
        """
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [0.5, 0.0], [5.0, 0.0]],
            scores=[0.9, 0.8, 0.7],
            # The highest scoring box is invalid
            valid_masks=[False, True, True],
        )

        keep_masks = batch_circle_nms(
            bboxes_centers=centers,
            scores=scores,
            min_radii=self.min_radii,
            valid_bboxes_masks=valid_masks,
            post_max_sizes=self.post_max_sizes,
        )

        # The first box is invalid and dropped, so the second box survives even though they overlap
        expected_keep_masks = self._expected_keep_masks([False, True, True])
        self.assertTrue(torch.equal(keep_masks, expected_keep_masks))

    def test_batch_circle_nms_caps_survivors_at_post_max_size(self) -> None:
        """
        Test that post_max_size truncates a class row to its highest scoring survivors even when
        no box overlaps another.
        """
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
            scores=[0.6, 0.9, 0.7, 0.8],
        )

        keep_masks = batch_circle_nms(
            bboxes_centers=centers,
            scores=scores,
            min_radii=self.min_radii,
            valid_bboxes_masks=valid_masks,
            post_max_sizes=[2] * self.num_classes,
        )

        # Only the 0.9 and 0.8 boxes fit under the post_max_size cap, so the 0.6 and 0.7
        # boxes are dropped even though they do not overlap
        expected_keep_masks = self._expected_keep_masks([False, True, False, True])
        self.assertTrue(torch.equal(keep_masks, expected_keep_masks))

    def test_batch_circle_nms_does_not_suppress_across_classes(self) -> None:
        """
        Test that overlapping boxes in different class rows both survive, since NMS is applied
        per class rather than across the whole frame.
        """
        # Every class row of every sample holds the same two overlapping centers, so a box that
        # is suppressed anywhere other than inside its own row would show up as a missing keep
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [0.2, 0.0]],
            scores=[0.9, 0.8],
        )

        keep_masks = batch_circle_nms(
            bboxes_centers=centers,
            scores=scores,
            min_radii=self.min_radii,
            valid_bboxes_masks=valid_masks,
            post_max_sizes=self.post_max_sizes,
        )

        # Each row independently keeps its own top box, for all num_classes*batch_size rows
        expected_keep_masks = self._expected_keep_masks([True, False])
        self.assertTrue(torch.equal(keep_masks, expected_keep_masks))
        self.assertEqual(int(keep_masks.sum().item()), self.batch_size * self.num_classes)

    def test_batch_circle_nms_applies_min_radius_per_class(self) -> None:
        """
        Test that a per-class min_radius sequence gives each class row its own radius, so identical
        geometry is suppressed differently from one row to the next.
        """
        # Two pairs, 0.5 m and 0.4 m apart, sitting 5 m from each other
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [0.5, 0.0], [5.0, 0.0], [5.4, 0.0]],
            scores=[0.9, 0.8, 0.7, 0.6],
        )

        keep_masks = batch_circle_nms(
            bboxes_centers=centers,
            scores=scores,
            min_radii=[0.45, 1.0, 6.0],
            valid_bboxes_masks=valid_masks,
            post_max_sizes=self.post_max_sizes,
        )

        expected_keep_masks = self._expected_keep_masks_per_class(
            [
                # 0.45 reaches the 0.4 m pair but not the 0.5 m one
                [True, True, True, False],
                # 1.0 reaches both pairs
                [True, False, True, False],
                # 6.0 covers the whole row, so only the top box survives
                [True, False, False, False],
            ]
        )
        self.assertTrue(torch.equal(keep_masks, expected_keep_masks))

    def test_batch_circle_nms_applies_post_max_size_per_class(self) -> None:
        """
        Test that a per-class post_max_size sequence caps each class row independently.
        """
        # Far enough apart that no box suppresses another, so only the cap decides
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
            scores=[0.6, 0.9, 0.7, 0.8],
        )

        keep_masks = batch_circle_nms(
            bboxes_centers=centers,
            scores=scores,
            min_radii=self.min_radii,
            valid_bboxes_masks=valid_masks,
            post_max_sizes=[1, 2, 10],
        )

        expected_keep_masks = self._expected_keep_masks_per_class(
            [
                # Only the 0.9 box fits
                [False, True, False, False],
                # The 0.9 and 0.8 boxes fit
                [False, True, False, True],
                # The cap is above the row length, so every box survives
                [True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(keep_masks, expected_keep_masks))

    def test_batch_circle_nms_applies_both_parameters_per_class(self) -> None:
        """
        Test a per-class radius and a per-class cap together, so each row is suppressed by its own
        radius and then truncated by its own cap.
        """
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [0.5, 0.0], [5.0, 0.0], [5.4, 0.0]],
            scores=[0.9, 0.8, 0.7, 0.6],
        )

        keep_masks = batch_circle_nms(
            bboxes_centers=centers,
            scores=scores,
            min_radii=[6.0, 1.0, 0.45],
            valid_bboxes_masks=valid_masks,
            post_max_sizes=[10, 1, 2],
        )

        expected_keep_masks = self._expected_keep_masks_per_class(
            [
                # 6.0 already leaves one box, so the cap of 10 changes nothing
                [True, False, False, False],
                # 1.0 leaves the 0.9 and 0.7 boxes, then the cap of 1 keeps only the 0.9 box
                [True, False, False, False],
                # 0.45 leaves the 0.9, 0.8 and 0.7 boxes, then the cap of 2 drops the 0.7 box
                [True, True, False, False],
            ]
        )
        self.assertTrue(torch.equal(keep_masks, expected_keep_masks))

    def test_batch_circle_nms_rejects_scalar_parameters(self) -> None:
        """
        Test that a scalar is rejected rather than broadcast, so a caller always states the
        per-class length explicitly and cannot silently apply one class's radius to all of them.
        """
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [0.5, 0.0]],
            scores=[0.9, 0.8],
        )

        with self.assertRaises(TypeError):
            batch_circle_nms(
                bboxes_centers=centers,
                scores=scores,
                min_radii=1.0,  # type: ignore
                valid_bboxes_masks=valid_masks,
                post_max_sizes=self.post_max_sizes,
            )

        with self.assertRaises(TypeError):
            batch_circle_nms(
                bboxes_centers=centers,
                scores=scores,
                min_radii=self.min_radii,
                valid_bboxes_masks=valid_masks,
                post_max_sizes=2,  # type: ignore
            )

    def test_batch_circle_nms_per_class_matches_per_class_scalar_calls(self) -> None:
        """
        Test that one per-class call equals one scalar call per class row, which is the property
        that lets a caller collapse a loop over groups into a single call.
        """
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [0.5, 0.0], [5.0, 0.0], [5.4, 0.0]],
            scores=[0.9, 0.8, 0.7, 0.6],
        )
        min_radii = [0.45, 1.0, 6.0]
        post_max_sizes = [10, 1, 2]

        per_class_keep_masks = batch_circle_nms(
            bboxes_centers=centers,
            scores=scores,
            min_radii=min_radii,
            valid_bboxes_masks=valid_masks,
            post_max_sizes=post_max_sizes,
        )

        for class_id in range(self.num_classes):
            class_slice = slice(class_id, class_id + 1)
            scalar_keep_masks = batch_circle_nms(
                bboxes_centers=centers[:, class_slice],
                scores=scores[:, class_slice],
                min_radii=[min_radii[class_id]],
                valid_bboxes_masks=valid_masks[:, class_slice],
                post_max_sizes=[post_max_sizes[class_id]],
            )
            self.assertTrue(
                torch.equal(per_class_keep_masks[:, class_slice], scalar_keep_masks),
                msg=f"class {class_id} differs from its own scalar call",
            )

    def test_batch_circle_nms_rejects_wrong_length_parameters(self) -> None:
        """Test that a per-class sequence not covering every class is rejected."""
        centers, scores, valid_masks = self._build_inputs(
            centers=[[0.0, 0.0], [5.0, 0.0]],
            scores=[0.9, 0.8],
        )

        with self.assertRaises(ValueError):
            batch_circle_nms(
                bboxes_centers=centers,
                scores=scores,
                min_radii=[1.0] * (self.num_classes - 1),
                valid_bboxes_masks=valid_masks,
                post_max_sizes=self.post_max_sizes,
            )

        with self.assertRaises(ValueError):
            batch_circle_nms(
                bboxes_centers=centers,
                scores=scores,
                min_radii=self.min_radii,
                valid_bboxes_masks=valid_masks,
                post_max_sizes=[2] * (self.num_classes + 1),
            )


class TestCreateOrientedGaussianHeatmaps(unittest.TestCase):
    """Unit tests for the create_oriented_gaussian_heatmaps function."""

    def setUp(self) -> None:
        """Set up a common heatmap geometry and a batch of oriented boxes for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.heatmap_width = 24
        self.heatmap_height = 20
        self.num_classes = 3
        self.min_sigma = 1.0

        # (batch_size, max_num_boxes, 2) as (x, y)
        self.centers = torch.tensor(
            [[[6, 5], [16, 12], [3, 15]], [[12, 10], [20, 3], [8, 8]]],
            dtype=torch.int64,
            device=self.device,
        )
        # A long rig, a compact box, and a mid-sized one per sample, in heatmap cells.
        # (batch_size, max_num_boxes)
        self.lengths_cells = torch.tensor(
            [[18.0, 3.0, 9.0], [24.0, 4.5, 6.0]], dtype=torch.float32, device=self.device
        )
        self.widths_cells = torch.tensor(
            [[4.0, 3.0, 5.0], [5.0, 4.5, 2.0]], dtype=torch.float32, device=self.device
        )
        self.yaws = torch.tensor(
            [[0.0, 0.6, -1.2], [math.pi / 2, 2.5, 0.3]], dtype=torch.float32, device=self.device
        )
        self.gt_bboxes_labels = torch.tensor(
            [[0, 1, 2], [2, 0, 1]], dtype=torch.int64, device=self.device
        )

    def _create_oriented_gaussian_heatmaps(
        self,
        valid_masks: Bool[torch.Tensor, "batch_size max_num_boxes"],
        lengths_cells: Float32[torch.Tensor, "batch_size max_num_boxes"] | None = None,
        widths_cells: Float32[torch.Tensor, "batch_size max_num_boxes"] | None = None,
        gt_bboxes_labels: Int64[torch.Tensor, "batch_size max_num_boxes"] | None = None,
    ) -> Float32[torch.Tensor, "batch_size num_classes heatmap_height heatmap_width"]:
        """Run the vectorized oriented heatmap creation with the shared test geometry."""
        return create_oriented_gaussian_heatmaps(
            heatmap_width=self.heatmap_width,
            heatmap_height=self.heatmap_height,
            num_classes=self.num_classes,
            centers=self.centers,
            lengths_cells=self.lengths_cells if lengths_cells is None else lengths_cells,
            widths_cells=self.widths_cells if widths_cells is None else widths_cells,
            yaws=self.yaws,
            gt_bboxes_labels=(
                self.gt_bboxes_labels if gt_bboxes_labels is None else gt_bboxes_labels
            ),
            valid_masks=valid_masks,
            device=self.device,
            min_sigma=self.min_sigma,
        )

    def _expected_oriented_gaussian_heatmaps(
        self,
        valid_masks: Bool[torch.Tensor, "batch_size max_num_boxes"],
        lengths_cells: Float32[torch.Tensor, "batch_size max_num_boxes"] | None = None,
        widths_cells: Float32[torch.Tensor, "batch_size max_num_boxes"] | None = None,
        gt_bboxes_labels: Int64[torch.Tensor, "batch_size max_num_boxes"] | None = None,
    ) -> Float32[torch.Tensor, "batch_size num_classes heatmap_height heatmap_width"]:
        """Build the reference heatmaps by looping the scalar draw_heatmap_gaussian_oriented."""
        lengths_cells = self.lengths_cells if lengths_cells is None else lengths_cells
        widths_cells = self.widths_cells if widths_cells is None else widths_cells
        gt_bboxes_labels = self.gt_bboxes_labels if gt_bboxes_labels is None else gt_bboxes_labels

        batch_size, max_num_boxes = gt_bboxes_labels.shape
        heatmaps = torch.zeros(
            (batch_size, self.num_classes, self.heatmap_height, self.heatmap_width),
            dtype=torch.float32,
            device=self.device,
        )
        for batch_index in range(batch_size):
            for box_index in range(max_num_boxes):
                if not bool(valid_masks[batch_index, box_index]):
                    continue
                draw_heatmap_gaussian_oriented(
                    heatmaps[batch_index, int(gt_bboxes_labels[batch_index, box_index])],
                    (
                        int(self.centers[batch_index, box_index, 0]),
                        int(self.centers[batch_index, box_index, 1]),
                    ),
                    float(lengths_cells[batch_index, box_index]),
                    float(widths_cells[batch_index, box_index]),
                    float(self.yaws[batch_index, box_index]),
                    min_sigma=self.min_sigma,
                )
        return heatmaps

    def test_create_oriented_gaussian_heatmaps(self) -> None:
        """Test that the vectorized oriented heatmaps match the scalar reference."""
        valid_masks = torch.ones_like(self.gt_bboxes_labels, dtype=torch.bool)
        oriented_heatmaps = self._create_oriented_gaussian_heatmaps(valid_masks)

        self.assertEqual(
            oriented_heatmaps.shape,
            (
                self.centers.shape[0],
                self.num_classes,
                self.heatmap_height,
                self.heatmap_width,
            ),
        )
        self.assertTrue(
            torch.allclose(
                oriented_heatmaps,
                self._expected_oriented_gaussian_heatmaps(valid_masks),
                atol=1e-5,
            )
        )

    def test_create_oriented_gaussian_heatmaps_with_invalid_mask(self) -> None:
        """Test that invalid bboxes contribute nothing to the oriented heatmaps."""
        valid_masks = torch.tensor([[1, 0, 1], [0, 1, 1]], device=self.device, dtype=torch.bool)
        oriented_heatmaps = self._create_oriented_gaussian_heatmaps(valid_masks)

        self.assertTrue(
            torch.allclose(
                oriented_heatmaps,
                self._expected_oriented_gaussian_heatmaps(valid_masks),
                atol=1e-5,
            )
        )
        # Batch 0 box 1 is the only source of class 1, and batch 1 box 0 the only source of
        # class 2, so both channels must stay empty once those boxes are masked out.
        self.assertTrue(torch.all(oriented_heatmaps[0, 1] == 0.0))
        self.assertTrue(torch.all(oriented_heatmaps[1, 2] == 0.0))

    def test_create_oriented_gaussian_heatmaps_ignores_padded_geometry(self) -> None:
        """Test that padded box geometry changes neither the heatmap nor the kernel size."""
        valid_masks = torch.tensor([[1, 0, 1], [0, 1, 1]], device=self.device, dtype=torch.bool)
        # The padded boxes carry an absurd length, a negative width and an out-of-range label, so
        # a leaking padded value would blow up the shared kernel size or corrupt the scatter.
        padded_lengths_cells = self.lengths_cells.clone()
        padded_lengths_cells[0, 1] = 5000.0
        padded_widths_cells = self.widths_cells.clone()
        padded_widths_cells[1, 0] = -13.0
        padded_gt_bboxes_labels = self.gt_bboxes_labels.clone()
        padded_gt_bboxes_labels[0, 1] = -1

        oriented_heatmaps = self._create_oriented_gaussian_heatmaps(
            valid_masks,
            lengths_cells=padded_lengths_cells,
            widths_cells=padded_widths_cells,
            gt_bboxes_labels=padded_gt_bboxes_labels,
        )
        expected_heatmaps = self._create_oriented_gaussian_heatmaps(valid_masks)
        self.assertTrue(torch.allclose(oriented_heatmaps, expected_heatmaps, atol=1e-5))

    def test_create_oriented_gaussian_heatmaps_with_all_invalid_masks(self) -> None:
        """Test that an all-invalid batch with padded geometry yields an empty heatmap."""
        oriented_heatmaps = self._create_oriented_gaussian_heatmaps(
            torch.zeros_like(self.gt_bboxes_labels, dtype=torch.bool),
            lengths_cells=torch.full_like(self.lengths_cells, 9999.0),
            widths_cells=torch.full_like(self.widths_cells, -1.0),
        )

        self.assertEqual(
            oriented_heatmaps.shape,
            (
                self.centers.shape[0],
                self.num_classes,
                self.heatmap_height,
                self.heatmap_width,
            ),
        )
        self.assertTrue(torch.all(oriented_heatmaps == 0.0))
        self.assertTrue(torch.all(oriented_heatmaps.isfinite()))

    def test_create_oriented_gaussian_heatmaps_clips_border_tails(self) -> None:
        """Test that blob tails of boxes near the border are clipped instead of wrapping."""
        self.centers = torch.tensor(
            [[[0, 0], [self.heatmap_width - 1, self.heatmap_height - 1], [1, 2]]],
            dtype=torch.int64,
            device=self.device,
        )
        self.lengths_cells = torch.tensor([[20.0, 20.0, 3.0]], device=self.device)
        self.widths_cells = torch.tensor([[4.0, 4.0, 3.0]], device=self.device)
        self.yaws = torch.tensor([[0.7, -1.9, 0.0]], device=self.device)
        self.gt_bboxes_labels = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=self.device)
        valid_masks = torch.ones_like(self.gt_bboxes_labels, dtype=torch.bool)

        oriented_heatmaps = self._create_oriented_gaussian_heatmaps(valid_masks)
        self.assertTrue(
            torch.allclose(
                oriented_heatmaps,
                self._expected_oriented_gaussian_heatmaps(valid_masks),
                atol=1e-5,
            )
        )

    def test_create_oriented_gaussian_heatmaps_elongates_along_yaw(self) -> None:
        """Test that a long box spreads along the x axis at yaw 0 and the y axis at yaw pi/2."""
        self.centers = torch.tensor(
            [[[self.heatmap_width // 2, self.heatmap_height // 2]]],
            dtype=torch.int64,
            device=self.device,
        )
        self.lengths_cells = torch.tensor([[18.0]], device=self.device)
        self.widths_cells = torch.tensor([[3.0]], device=self.device)
        self.gt_bboxes_labels = torch.zeros((1, 1), dtype=torch.int64, device=self.device)
        valid_masks = torch.ones((1, 1), dtype=torch.bool, device=self.device)

        extents = []
        for yaw in (0.0, math.pi / 2):
            self.yaws = torch.tensor([[yaw]], device=self.device)
            oriented_heatmap = self._create_oriented_gaussian_heatmaps(valid_masks)[0, 0]
            # Column and row supports of the blob, in cells.
            extents.append(
                (
                    int((oriented_heatmap.sum(dim=0) > 0.05).sum()),
                    int((oriented_heatmap.sum(dim=1) > 0.05).sum()),
                )
            )

        (x_extent_yaw_0, y_extent_yaw_0), (x_extent_yaw_90, y_extent_yaw_90) = extents
        self.assertGreater(x_extent_yaw_0, y_extent_yaw_0)
        self.assertGreater(y_extent_yaw_90, x_extent_yaw_90)
        # Rotating the same box by 90 degrees swaps its support.
        self.assertEqual(x_extent_yaw_0, y_extent_yaw_90)
        self.assertEqual(y_extent_yaw_0, x_extent_yaw_90)


if __name__ == "__main__":
    unittest.main()
