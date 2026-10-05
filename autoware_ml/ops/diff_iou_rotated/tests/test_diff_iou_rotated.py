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

"""Unit tests for the differentiable rotated IoU extension."""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.ops.diff_iou_rotated import (
    box2corners,
    diff_iou_rotated_2d,
    diff_iou_rotated_3d,
    enclosing_area,
    oriented_box_intersection_2d,
)
from autoware_ml.ops.diff_iou_rotated.diff_iou_rotated import (
    box_in_box,
    box_intersection,
    build_vertices,
    calculate_area,
    drop_duplicate_vertices,
    sort_indices,
)


@unittest.skipUnless(torch.cuda.is_available(), "diff_iou_rotated requires CUDA")
class TestDiffIoURotated(unittest.TestCase):
    """Check the ported mmcv differentiable rotated IoU against its reference values."""

    def setUp(self) -> None:
        """Pin the device used by the CUDA-only vertex sorting kernel."""
        self.device = torch.device("cuda:0")

    def test_diff_iou_rotated_2d_matches_reference_values(self) -> None:
        """The 2D IoU matches the mmcv reference cases: identical, rotated, shifted, disjoint."""
        boxes1 = torch.tensor([[[0.5, 0.5, 1.0, 1.0, 0.0]] * 5], device=self.device)
        boxes2 = torch.tensor(
            [
                [
                    [0.5, 0.5, 1.0, 1.0, 0.0],
                    [0.5, 0.5, 1.0, 1.0, math.pi / 2],
                    [0.5, 0.5, 1.0, 1.0, math.pi / 4],
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                    [1.5, 1.5, 1.0, 1.0, 0.0],
                ]
            ],
            device=self.device,
        )

        ious = diff_iou_rotated_2d(boxes1, boxes2)

        expected = torch.tensor([[1.0, 1.0, 0.7071, 1.0 / 7.0, 0.0]], device=self.device)
        torch.testing.assert_close(ious, expected, rtol=0.0, atol=1e-4)

    def test_diff_iou_rotated_3d_matches_reference_values(self) -> None:
        """The 3D IoU matches the mmcv reference cases, including a partial z overlap."""
        boxes1 = torch.tensor([[[0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0]] * 5], device=self.device)
        boxes2 = torch.tensor(
            [
                [
                    [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0],
                    [0.5, 0.5, 0.5, 1.0, 1.0, 2.0, math.pi / 2],
                    [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, math.pi / 4],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [-1.5, -1.5, -1.5, 2.5, 2.5, 2.5, 0.0],
                ]
            ],
            device=self.device,
        )

        ious = diff_iou_rotated_3d(boxes1, boxes2)

        expected = torch.tensor([[1.0, 0.5, 0.7071, 1.0 / 15.0, 0.0]], device=self.device)
        torch.testing.assert_close(ious, expected, rtol=0.0, atol=1e-4)

    def test_diff_iou_rotated_2d_rectangle_cases(self) -> None:
        """A 2x1 rectangle has no 90-degree symmetry, so rotation changes the IoU.

        Every expectation is derived by hand. Box1 is the rectangle centered at (0.5, 0.5),
        spanning x in [-0.5, 1.5] and y in [0, 1], area 2.
        """
        boxes1 = torch.tensor([[[0.5, 0.5, 2.0, 1.0, 0.0]] * 6], device=self.device)
        boxes2 = torch.tensor(
            [
                [
                    # Identical rectangle: IoU 1.
                    [0.5, 0.5, 2.0, 1.0, 0.0],
                    # Rotated 180 degrees: same footprint, IoU 1.
                    [0.5, 0.5, 2.0, 1.0, math.pi],
                    # Rotated 90 degrees about the same center: overlap is the central 1x1
                    # square, union 2 + 2 - 1 = 3, IoU 1/3.
                    [0.5, 0.5, 2.0, 1.0, math.pi / 2],
                    # Rotated 90 degrees and shifted to x = 1.25: box2 spans x in [0.75, 1.75],
                    # so the overlap is 0.75 wide and 1 tall, union 4 - 0.75, IoU 3/13.
                    [1.25, 0.5, 2.0, 1.0, math.pi / 2],
                    # Axis-aligned and shifted by (0.5, 0.5): overlap 1.5 x 0.5 = 0.75,
                    # union 4 - 0.75, IoU 3/13.
                    [1.0, 1.0, 2.0, 1.0, 0.0],
                    # Axis-aligned and shifted by (2, 0): the boxes only touch, IoU 0.
                    [2.5, 0.5, 2.0, 1.0, 0.0],
                ]
            ],
            device=self.device,
        )

        ious = diff_iou_rotated_2d(boxes1, boxes2)

        expected = torch.tensor(
            [[1.0, 1.0, 1.0 / 3.0, 3.0 / 13.0, 3.0 / 13.0, 0.0]], device=self.device
        )
        torch.testing.assert_close(ious, expected, rtol=0.0, atol=1e-4)

    def test_diff_iou_rotated_3d_rectangle_cases(self) -> None:
        """A 2x1x1 cuboid combines the BEV rectangle overlap with the z overlap.

        Box1 is centered at (0.5, 0.5, 0.5) with BEV size 2x1 and height 1, volume 2.
        """
        boxes1 = torch.tensor([[[0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 0.0]] * 5], device=self.device)
        boxes2 = torch.tensor(
            [
                [
                    # Identical cuboid: IoU 1.
                    [0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 0.0],
                    # Rotated 90 degrees in BEV: BEV overlap 1, z overlap 1, union 3, IoU 1/3.
                    [0.5, 0.5, 0.5, 2.0, 1.0, 1.0, math.pi / 2],
                    # Rotated 90 degrees and lifted by 0.5: BEV overlap 1, z overlap 0.5,
                    # intersection 0.5, union 2 + 2 - 0.5, IoU 1/7.
                    [0.5, 0.5, 1.0, 2.0, 1.0, 1.0, math.pi / 2],
                    # Same footprint, twice the height: intersection 2, union 2 + 4 - 2, IoU 1/2.
                    [0.5, 0.5, 0.5, 2.0, 1.0, 2.0, 0.0],
                    # Same footprint, stacked on top: no z overlap, IoU 0.
                    [0.5, 0.5, 1.5, 2.0, 1.0, 1.0, 0.0],
                ]
            ],
            device=self.device,
        )

        ious = diff_iou_rotated_3d(boxes1, boxes2)

        expected = torch.tensor([[1.0, 1.0 / 3.0, 1.0 / 7.0, 0.5, 0.0]], device=self.device)
        torch.testing.assert_close(ious, expected, rtol=0.0, atol=1e-4)

    def test_nearly_identical_boxes_have_unit_iou(self) -> None:
        """Boxes a few float ulps apart behave like identical boxes instead of losing vertices.

        Corners that coincide up to float noise used to reach the sorting kernel as distinct
        candidates and broke its ordering. The axis-aligned square and the two rotated
        rectangles below cover the cases where that produced half the intersection area.
        """
        boxes1 = torch.tensor(
            [
                [
                    [8.0, 8.0, 2.0, 2.0, 0.0],
                    [2.0, 2.0, 4.0, 1.6, 0.25],
                    [12.0, 4.0, 3.0, 1.8, 1.0],
                ]
            ],
            device=self.device,
        )
        boxes2 = boxes1.clone()
        boxes2[..., :2] += 1e-6
        boxes2[..., 4] += 1e-6

        ious = diff_iou_rotated_2d(boxes1, boxes2)

        torch.testing.assert_close(ious, torch.ones_like(ious), rtol=0.0, atol=1e-4)

    def test_intersection_area_of_axis_aligned_overlap(self) -> None:
        """Two unit squares offset by half a unit on each axis overlap in a quarter square."""
        corners1 = box2corners(torch.tensor([[[0.0, 0.0, 1.0, 1.0, 0.0]]], device=self.device))
        corners2 = box2corners(torch.tensor([[[0.5, 0.5, 1.0, 1.0, 0.0]]], device=self.device))

        area, polygon = oriented_box_intersection_2d(corners1, corners2)

        torch.testing.assert_close(
            area, torch.tensor([[0.25]], device=self.device), atol=1e-5, rtol=0.0
        )
        self.assertEqual(polygon.shape, (1, 1, 25, 2))

    def test_sort_vertices_keeps_more_than_eight_candidates(self) -> None:
        """The sorting kernel orders and integrates a polygon with more than 8 valid vertices.

        Two rectangles share at most 8 polygon vertices, but float tolerances can leave more
        candidates valid. A regular 12-gon fed through the kernel directly checks that every
        valid candidate ends up in the row, in counter-clockwise order from the positive x axis,
        closed by the first vertex and padded with the first invalid intersection slot.
        """
        angles_deg = [7.5, 37.5, 67.5, 97.5, 127.5, 157.5, 187.5, 217.5, 247.5, 277.5, 307.5, 337.5]
        # Candidate slot holding each vertex, scrambled so the input order carries no information.
        slots = [5, 11, 0, 8, 3, 9, 1, 10, 6, 2, 7, 4]
        vertices = torch.zeros((1, 1, 24, 2), device=self.device)
        mask = torch.zeros((1, 1, 24), dtype=torch.bool, device=self.device)
        for angle_deg, slot in zip(angles_deg, slots):
            angle = math.radians(angle_deg)
            vertices[0, 0, slot, 0] = math.cos(angle)
            vertices[0, 0, slot, 1] = math.sin(angle)
            mask[0, 0, slot] = True

        idx = sort_indices(vertices, mask)
        area, _ = calculate_area(idx, vertices)

        # Slots 8 to 11 hold valid intersections, so slot 12 is the first invalid one.
        expected = slots + [slots[0]] + [12 for _ in range(12)]
        self.assertEqual(idx[0, 0].tolist(), expected)
        # A regular 12-gon of circumradius 1 has area 6 sin(30 degrees) = 3.
        torch.testing.assert_close(
            area, torch.tensor([[3.0]], device=self.device), atol=1e-5, rtol=0.0
        )

    def test_more_than_eight_candidates_keep_the_full_polygon(self) -> None:
        """Box pairs that leave more than 8 valid candidates still get the right IoU.

        Far from the origin, float32 noise exceeds the classification tolerances, so a corner
        sitting on the other box's edge can count as inside while the two edges meeting there
        also register crossings. Nearly identical boxes and a square inscribed in a square
        rotated by 45 degrees then reach the sorting kernel with 9 or more candidates. Clamping
        the count to 8 used to drop a real vertex and return an IoU of 0.69 for identical boxes.
        """
        boxes1 = torch.tensor(
            [
                [
                    # Nearly identical boxes 900 m from the origin: 10 candidates.
                    [-900.0, 4.0, 3.2, 2.9, 4.1],
                    # Square inscribed in a square of side 7.071, 5 times sqrt 2: 9 candidates.
                    [1200.0, 4.0, 5.0, 5.0, 0.5],
                    # The same pair in the other order: 9 candidates.
                    [1200.0, 4.0, 7.071, 7.071, 1.2854],
                ]
            ],
            device=self.device,
        )
        boxes2 = torch.tensor(
            [
                [
                    [-900.0, 4.0, 3.2, 2.9, 4.09998],
                    [1200.0, 4.0, 7.071, 7.071, 1.2854],
                    [1200.0, 4.0, 5.0, 5.0, 0.5],
                ]
            ],
            device=self.device,
        )

        corners1 = box2corners(boxes1)
        corners2 = box2corners(boxes2)
        intersections, valid_mask = box_intersection(corners1, corners2)
        c12, c21 = box_in_box(corners1, corners2)
        vertices, mask = build_vertices(corners1, corners2, c12, c21, intersections, valid_mask)
        num_valid = drop_duplicate_vertices(vertices, mask).sum(dim=-1)
        ious = diff_iou_rotated_2d(boxes1, boxes2)

        # The cases only test the kernel if the dedup indeed leaves more than 8 candidates.
        self.assertTrue(bool((num_valid > 8).all()), num_valid.tolist())
        # Identical boxes: IoU 1. The inscribed square covers half of the larger one: IoU 1/2.
        expected = torch.tensor([[1.0, 0.5, 0.5]], device=self.device)
        torch.testing.assert_close(ious, expected, rtol=0.0, atol=1e-3)

    def test_deduplicated_crossing_does_not_leak_into_the_padding(self) -> None:
        """A crossing dropped as a duplicate must not feed the padding of the polygon row.

        The kernel pads each row with the first invalid intersection slot and the shoelace sum
        relies on that slot being zero. A unit square whose top-left corner pokes 2e-6 outside a
        large square rotated by 45 degrees has that corner inside by tolerance, while the two
        crossings next to it are valid and then dropped as its duplicates. The first of them is
        slot 8, exactly the padding slot, and it kept its coordinates: the unbalanced edge from
        the first polygon vertex to it used to add a quarter of the unit area, IoU 0.079
        instead of 1/16.
        """
        offset = 2e-6
        boxes1 = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 0.0]]], device=self.device)
        boxes2 = torch.tensor(
            [[[math.sqrt(2) - 0.5 + offset, 0.5 - math.sqrt(2) - offset, 4.0, 4.0, math.pi / 4]]],
            device=self.device,
        )

        corners1 = box2corners(boxes1)
        corners2 = box2corners(boxes2)
        intersections, valid_mask = box_intersection(corners1, corners2)
        c12, c21 = box_in_box(corners1, corners2)
        vertices, raw_mask = build_vertices(corners1, corners2, c12, c21, intersections, valid_mask)
        mask = drop_duplicate_vertices(vertices, raw_mask)
        pad = sort_indices(vertices, mask)[0, 0, -1]
        ious = diff_iou_rotated_2d(boxes1, boxes2)

        # The case only tests the padding if the padded slot is a dropped, nonzero crossing.
        self.assertEqual(int(pad), 8)
        self.assertTrue(bool(raw_mask[0, 0, pad]))
        self.assertFalse(bool(mask[0, 0, pad]))
        self.assertGreater(float(vertices[0, 0, pad].abs().sum()), 0.0)
        # The unit square lies inside the 4 x 4 square: IoU 1/16.
        torch.testing.assert_close(
            ious, torch.tensor([[1.0 / 16.0]], device=self.device), rtol=0.0, atol=1e-4
        )

    def test_iou_is_differentiable_with_respect_to_box_parameters(self) -> None:
        """Gradients flow to the box parameters and push a shifted box towards the target."""
        boxes1 = torch.tensor([[[0.3, 0.0, 1.0, 1.0, 0.1]]], device=self.device, requires_grad=True)
        boxes2 = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 0.0]]], device=self.device)

        iou = diff_iou_rotated_2d(boxes1, boxes2)
        iou.sum().backward()

        self.assertIsNotNone(boxes1.grad)
        assert boxes1.grad is not None
        self.assertTrue(torch.isfinite(boxes1.grad).all())
        # Moving box1 back towards the origin raises the IoU, so d(iou)/dx is negative.
        self.assertLess(float(boxes1.grad[0, 0, 0]), 0.0)

    def test_enclosing_area_variants_are_ordered_by_tightness(self) -> None:
        """Convex hull <= smallest rotated box <= axis-aligned box for two rotated squares."""
        corners1 = box2corners(torch.tensor([[[0.0, 0.0, 2.0, 1.0, 0.3]]], device=self.device))
        corners2 = box2corners(torch.tensor([[[1.0, 0.5, 1.0, 1.0, -0.6]]], device=self.device))

        aligned = enclosing_area(corners1, corners2, "aligned")
        smallest = enclosing_area(corners1, corners2, "smallest")
        hull = enclosing_area(corners1, corners2, "convex_hull")

        self.assertLessEqual(float(hull), float(smallest) + 1e-5)
        self.assertLessEqual(float(smallest), float(aligned) + 1e-5)
        with self.assertRaises(ValueError):
            enclosing_area(corners1, corners2, "unknown")

    def test_empty_batch_returns_empty_iou(self) -> None:
        """A batch with no boxes yields an empty IoU tensor instead of launching a bad kernel."""
        boxes = torch.zeros((1, 0, 5), device=self.device)

        ious = diff_iou_rotated_2d(boxes, boxes)

        self.assertEqual(ious.shape, (1, 0))


if __name__ == "__main__":
    unittest.main()
