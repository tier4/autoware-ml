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

import unittest
from typing import Sequence

from jaxtyping import Float32, Bool
import torch

from autoware_ml.models.detection3d.task_modules.heatmap import (
    _vectorize_gaussian2d,
    vectorize_gaussian_radii,
    create_gaussian_heatmaps,
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

    def test_create_gaussian_heatmaps_rejects_labels_beyond_num_classes(self) -> None:
        """Test that a label outside [0, num_classes - 1] is rejected instead of silently clamped."""
        # ``num_classes`` is the first out-of-range index; clamping it would splat the box onto
        # the last class channel and corrupt that class's supervision without any warning.
        for out_of_range_label in (self.num_classes, self.num_classes + 7):
            with self.subTest(label=out_of_range_label):
                gt_bboxes_labels = self.gt_bboxes_labels.clone()
                gt_bboxes_labels[1, 0] = out_of_range_label

                with self.assertRaisesRegex(ValueError, "label"):
                    create_gaussian_heatmaps(
                        heatmap_width=self.heatmap_width,
                        heatmap_height=self.heatmap_height,
                        num_classes=self.num_classes,
                        centers=self.centers,
                        gaussian_radii=self.gaussian_radii,
                        gt_bboxes_labels=gt_bboxes_labels,
                        valid_masks=torch.ones_like(gt_bboxes_labels, dtype=torch.bool),
                        device=self.device,
                    )

    def test_create_gaussian_heatmaps_rejects_out_of_range_label_on_invalid_box(self) -> None:
        """Test that the label range check covers padded boxes, not just the valid ones."""
        # The check runs before ``valid_masks`` is applied, so padding must use a sentinel the
        # clamp handles (such as -1) rather than an above-range one.
        gt_bboxes_labels = self.gt_bboxes_labels.clone()
        gt_bboxes_labels[0, 1] = self.num_classes
        valid_masks = torch.tensor([[1, 0, 1], [1, 1, 1]], device=self.device, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "label"):
            create_gaussian_heatmaps(
                heatmap_width=self.heatmap_width,
                heatmap_height=self.heatmap_height,
                num_classes=self.num_classes,
                centers=self.centers,
                gaussian_radii=self.gaussian_radii,
                gt_bboxes_labels=gt_bboxes_labels,
                valid_masks=valid_masks,
                device=self.device,
            )

    def test_create_gaussian_heatmaps_accepts_last_class_label(self) -> None:
        """Test that ``num_classes - 1`` is still accepted, pinning the check to ``>=``."""
        last_class = self.num_classes - 1
        gt_bboxes_labels = self.gt_bboxes_labels.clone()
        gt_bboxes_labels[1, 0] = last_class

        gaussian_heatmaps = create_gaussian_heatmaps(
            heatmap_width=self.heatmap_width,
            heatmap_height=self.heatmap_height,
            num_classes=self.num_classes,
            centers=self.centers,
            gaussian_radii=self.gaussian_radii,
            gt_bboxes_labels=gt_bboxes_labels,
            valid_masks=torch.ones_like(gt_bboxes_labels, dtype=torch.bool),
            device=self.device,
        )

        self.assertEqual(
            gaussian_heatmaps.shape,
            (self.batch_size, self.num_classes, self.heatmap_height, self.heatmap_width),
        )
        # The relocated box is the only contributor to the last channel, so it must be drawn there.
        self.assertTrue(torch.any(gaussian_heatmaps[1, last_class] > 0.0))


class TestBatchCircleNMS(unittest.TestCase):
    """Unit tests for the batch_circle_nms function."""

    def setUp(self) -> None:
        """Set up the common device, batch shape and NMS parameters for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 2
        self.num_classes = 3
        self.min_radius = 1.0
        self.post_max_size = 10

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
            min_radius=self.min_radius,
            valid_bboxes_masks=valid_masks,
            post_max_size=self.post_max_size,
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
            min_radius=2.0,
            valid_bboxes_masks=valid_masks,
            post_max_size=self.post_max_size,
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
            min_radius=self.min_radius,
            valid_bboxes_masks=valid_masks,
            post_max_size=self.post_max_size,
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
            min_radius=self.min_radius,
            valid_bboxes_masks=valid_masks,
            post_max_size=2,
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
            min_radius=self.min_radius,
            valid_bboxes_masks=valid_masks,
            post_max_size=self.post_max_size,
        )

        # Each row independently keeps its own top box, for all num_classes*batch_size rows
        expected_keep_masks = self._expected_keep_masks([True, False])
        self.assertTrue(torch.equal(keep_masks, expected_keep_masks))
        self.assertEqual(int(keep_masks.sum().item()), self.batch_size * self.num_classes)


if __name__ == "__main__":
    unittest.main()
