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

"""Unit tests for the projection of the 3D boxes onto the cameras as 2D targets."""

from __future__ import annotations

from collections.abc import Sequence
import math
import unittest

import torch

from autoware_ml.dataclasses.batch.detection2d import Detection2DGTSample
from autoware_ml.transforms.camera.annotations2d import LoadAnnotations2DFromBoxes3D
from autoware_ml.transforms.camera.tests.pinhole_fixtures import (
    LIDAR2CAM_FORWARD,
    build_boxes,
    build_pinhole_images,
    build_sample,
)


def project_annotations(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int] | None = None,
    lidar2cams: Sequence[torch.Tensor] = (LIDAR2CAM_FORWARD,),
    width: int = 160,
    height: int = 96,
) -> Detection2DGTSample:
    """Run LoadAnnotations2DFromBoxes3D on boxes seen by the given cameras."""
    sample = build_sample(build_pinhole_images(lidar2cams, width, height))._replace(
        detection3d_gt_bboxes_3d=build_boxes(rows, [0] * len(rows) if labels is None else labels)
    )
    return LoadAnnotations2DFromBoxes3D()(sample).detection2d_gt_sample


class TestLoadAnnotations2DFromBoxes3D(unittest.TestCase):
    """Projection of the 3D boxes onto every camera as auxiliary 2D targets."""

    def test_projects_a_box_in_front_of_the_camera(self) -> None:
        # A 2 m cube on the optical axis, 10 m ahead, spans 9-11 m in depth: its gravity
        # center projects onto the principal point and its near face sets the extent.
        annotations = project_annotations(
            [[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], labels=[2], width=100, height=100
        )

        self.assertEqual(annotations.gt_labels_2d[0].tolist(), [2])
        self.assertEqual(annotations.gt_labels_2d[0].dtype, torch.int32)
        torch.testing.assert_close(annotations.gt_centers_2d[0][0], torch.tensor([50.0, 50.0]))
        extent = 100.0 / 9.0
        torch.testing.assert_close(
            annotations.gt_bboxes_2d[0][0],
            torch.tensor([50.0 - extent, 50.0 - extent, 50.0 + extent, 50.0 + extent]),
        )

    def test_drops_a_box_behind_the_camera(self) -> None:
        annotations = project_annotations([[-10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])

        self.assertEqual(tuple(annotations.gt_bboxes_2d[0].shape), (0, 4))
        self.assertEqual(tuple(annotations.gt_centers_2d[0].shape), (0, 2))
        self.assertEqual(tuple(annotations.gt_labels_2d[0].shape), (0,))

    def test_keeps_a_box_whose_center_leaves_the_image(self) -> None:
        # The center at y = 9 m projects to x = -10 while the near-right corners are inside:
        # the box stays a target with its extent clipped and its center clamped to the edge,
        # as in the reference recipe.
        annotations = project_annotations([[10.0, 9.0, 0.0, 2.0, 6.0, 2.0, 0.0]])

        x1, _, x2, _ = annotations.gt_bboxes_2d[0][0].tolist()
        self.assertEqual(x1, 0.0)
        self.assertTrue(0 < x2 <= 160)
        self.assertEqual(float(annotations.gt_centers_2d[0][0][0]), 0.0)

    def test_clips_a_partially_visible_box_to_the_image(self) -> None:
        # The center projects at x = 15, the near-left corners far outside the left edge.
        annotations = project_annotations([[10.0, 6.5, 0.0, 2.0, 6.0, 2.0, 0.0]])

        x1, _, x2, _ = annotations.gt_bboxes_2d[0][0].tolist()
        self.assertEqual(x1, 0.0)
        self.assertTrue(0 < x2 <= 160)
        self.assertTrue(0 <= float(annotations.gt_centers_2d[0][0][0]) < 160)

    def test_follows_the_box_yaw(self) -> None:
        # A 4 m x 1 m box turned by 90 degrees shows its long side to the camera.
        straight = project_annotations([[10.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0]])
        turned = project_annotations([[10.0, 0.0, 0.0, 4.0, 1.0, 1.0, math.pi / 2]])

        def width(annotations: Detection2DGTSample) -> float:
            x1, _, x2, _ = annotations.gt_bboxes_2d[0][0].tolist()
            return x2 - x1

        self.assertGreater(width(turned), 2.0 * width(straight))

    def test_assigns_every_box_to_the_cameras_that_see_it(self) -> None:
        # The second camera looks backward.
        backward = LIDAR2CAM_FORWARD @ torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
        annotations = project_annotations(
            [[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], [-10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]],
            labels=[0, 1],
            lidar2cams=(LIDAR2CAM_FORWARD, backward),
        )

        self.assertEqual(annotations.gt_labels_2d[0].tolist(), [0])
        self.assertEqual(annotations.gt_labels_2d[1].tolist(), [1])

    def test_handles_a_frame_without_boxes(self) -> None:
        annotations = project_annotations([])

        self.assertEqual(tuple(annotations.gt_bboxes_2d[0].shape), (0, 4))
        self.assertEqual(tuple(annotations.gt_centers_2d[0].shape), (0, 2))

    def test_needs_the_3d_boxes(self) -> None:
        sample = build_sample(build_pinhole_images([LIDAR2CAM_FORWARD], 160, 96))

        with self.assertRaises(KeyError):
            LoadAnnotations2DFromBoxes3D()(sample)


if __name__ == "__main__":
    unittest.main()
