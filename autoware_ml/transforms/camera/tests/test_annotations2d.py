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
from autoware_ml.transforms.camera.tests.pinhole_camera_test_case import PinholeCameraTestCase
from autoware_ml.types.geometry import Box2DFieldIndex


class TestLoadAnnotations2DFromBoxes3D(PinholeCameraTestCase):
    """Projection of the 3D boxes onto every camera as auxiliary 2D targets."""

    def project_annotations(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int] | None = None,
        lidar2cams: Sequence[torch.Tensor] | None = None,
        width: int = 160,
        height: int = 96,
    ) -> Detection2DGTSample:
        """Run LoadAnnotations2DFromBoxes3D on boxes seen by the given cameras.

        Args:
            rows: ``[x, y, z, l, w, h, yaw]`` of every gravity-center box.
            labels: Class index of every box, 0 for all when None.
            lidar2cams: Pose of every camera, the forward camera alone when None.
            width: Width of the images, in pixels.
            height: Height of the images, in pixels.

        Returns:
            Detection2DGTSample: The 2D ground truth the transform derives.
        """
        if lidar2cams is None:
            lidar2cams = (self.LIDAR2CAM_FORWARD,)
        images = self.build_pinhole_images(lidar2cams, width, height)
        boxes = self.build_boxes(rows, [0] * len(rows) if labels is None else labels)
        sample = self.build_sample(images)._replace(detection3d_gt_bboxes_3d=boxes)
        return LoadAnnotations2DFromBoxes3D()(sample).detection2d_gt_sample

    def test_projects_a_box_in_front_of_the_camera(self) -> None:
        """
        Input: a 2 m cube of class 2 on the optical axis of a 100 x 100 pixel forward camera,
        10 m ahead.
        Expected: the cube spans 9-11 m in depth, so its near face sets the extent, 100 / 9
        pixels around the principal point, and its gravity center projects onto the principal
        point; the label keeps its int32 class index.
        Check: compare every Box2DFieldIndex column of the box with its expected value, and the
        label and its dtype.
        """
        annotations = self.project_annotations(
            [[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], labels=[2], width=100, height=100
        )

        self.assertEqual(annotations.gt_labels_2d[0].tolist(), [2])
        self.assertEqual(annotations.gt_labels_2d[0].dtype, torch.int32)
        extent = 100.0 / 9.0
        expected = {
            Box2DFieldIndex.X_MIN: 50.0 - extent,
            Box2DFieldIndex.Y_MIN: 50.0 - extent,
            Box2DFieldIndex.X_MAX: 50.0 + extent,
            Box2DFieldIndex.Y_MAX: 50.0 + extent,
            Box2DFieldIndex.CENTER_X: 50.0,
            Box2DFieldIndex.CENTER_Y: 50.0,
        }
        torch.testing.assert_close(
            annotations.gt_bboxes_2d[0][0],
            torch.tensor([expected[field] for field in Box2DFieldIndex]),
        )

    def test_drops_a_box_behind_the_camera(self) -> None:
        """
        Input: a box 10 m behind the forward camera.
        Expected: no corner of the box is in front of the camera, so it is no target.
        Check: the camera holds no box and no label.
        """
        annotations = self.project_annotations([[-10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])

        self.assertEqual(tuple(annotations.gt_bboxes_2d[0].shape), (0, len(Box2DFieldIndex)))
        self.assertEqual(tuple(annotations.gt_labels_2d[0].shape), (0,))

    def test_keeps_a_box_whose_center_leaves_the_image(self) -> None:
        """
        Input: a 6 m wide box 10 m ahead and 9 m to the left, whose center projects to x = -10
        while its near right corners are inside the image.
        Expected: the box stays a target, as in the reference recipe, with its extent clipped
        to the image and its center clamped to the left edge.
        Check: x_min and center_x are 0 and x_max lies inside the image.
        """
        annotations = self.project_annotations([[10.0, 9.0, 0.0, 2.0, 6.0, 2.0, 0.0]])

        box = annotations.gt_bboxes_2d[0][0]
        self.assertEqual(float(box[Box2DFieldIndex.X_MIN]), 0.0)
        self.assertTrue(0 < float(box[Box2DFieldIndex.X_MAX]) <= 160)
        self.assertEqual(float(box[Box2DFieldIndex.CENTER_X]), 0.0)

    def test_clips_a_partially_visible_box_to_the_image(self) -> None:
        """
        Input: a 6 m wide box 10 m ahead and 6.5 m to the left, whose center projects to x = 15
        while its near left corners fall far outside the left edge.
        Expected: the extent is clipped to the image, and the center, inside the image, stays
        where it projects.
        Check: x_min is 0, and x_max and center_x lie inside the image.
        """
        annotations = self.project_annotations([[10.0, 6.5, 0.0, 2.0, 6.0, 2.0, 0.0]])

        box = annotations.gt_bboxes_2d[0][0]
        self.assertEqual(float(box[Box2DFieldIndex.X_MIN]), 0.0)
        self.assertTrue(0 < float(box[Box2DFieldIndex.X_MAX]) <= 160)
        self.assertTrue(0 <= float(box[Box2DFieldIndex.CENTER_X]) < 160)

    def test_follows_the_box_yaw(self) -> None:
        """
        Input: a 4 m x 1 m box 10 m ahead, along the optical axis and turned by 90 degrees.
        Expected: turned, the box shows its long side to the camera, so its 2D box is more than
        twice as wide.
        Check: compare the widths x_max - x_min of the two boxes.
        """
        straight = self.project_annotations([[10.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0]])
        turned = self.project_annotations([[10.0, 0.0, 0.0, 4.0, 1.0, 1.0, math.pi / 2]])

        def width(annotations: Detection2DGTSample) -> float:
            box = annotations.gt_bboxes_2d[0][0]
            return float(box[Box2DFieldIndex.X_MAX] - box[Box2DFieldIndex.X_MIN])

        self.assertGreater(width(turned), 2.0 * width(straight))

    def test_assigns_every_box_to_the_cameras_that_see_it(self) -> None:
        """
        Input: a box of class 0 ahead and a box of class 1 behind, seen by a forward and a
        backward camera.
        Expected: every camera gets the boxes in front of it only.
        Check: the forward camera holds label 0 and the backward camera label 1.
        """
        # The second camera looks backward.
        backward = self.LIDAR2CAM_FORWARD @ torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
        annotations = self.project_annotations(
            [[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], [-10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]],
            labels=[0, 1],
            lidar2cams=(self.LIDAR2CAM_FORWARD, backward),
        )

        self.assertEqual(annotations.gt_labels_2d[0].tolist(), [0])
        self.assertEqual(annotations.gt_labels_2d[1].tolist(), [1])

    def test_handles_a_frame_without_boxes(self) -> None:
        """
        Input: a frame without any 3D box.
        Expected: the camera gets an empty box set laid out by Box2DFieldIndex, not an error.
        Check: the boxes have shape (0, 6) and the labels (0,).
        """
        annotations = self.project_annotations([])

        self.assertEqual(tuple(annotations.gt_bboxes_2d[0].shape), (0, len(Box2DFieldIndex)))
        self.assertEqual(tuple(annotations.gt_labels_2d[0].shape), (0,))

    def test_needs_the_3d_boxes(self) -> None:
        """
        Input: a sample holding images but no 3D boxes.
        Expected: the 2D targets are projected from the 3D boxes, so the transform requires
        them.
        Check: a KeyError is raised.
        """
        sample = self.build_sample(self.build_pinhole_images([self.LIDAR2CAM_FORWARD], 160, 96))

        with self.assertRaises(KeyError):
            LoadAnnotations2DFromBoxes3D()(sample)


if __name__ == "__main__":
    unittest.main()
