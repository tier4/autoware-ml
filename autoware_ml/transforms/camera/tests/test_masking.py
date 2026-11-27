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

"""Unit tests for the masking of the ego vehicle regions and of the boxes it hides."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.transforms.camera.masking import (
    BBoxesEgoVehicleFilter,
    MaskEgoVehicle,
    database_group,
    normalize_polygon,
    rasterize_polygon,
)
from autoware_ml.transforms.camera.tests.pinhole_camera_test_case import PinholeCameraTestCase

# The left half of the image: columns 0 to 2 of a 5 pixel wide image, every row.
LEFT_HALF = [0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0]
# The bottom row of the image.
BOTTOM_ROW = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]


class TestDatabaseGroup(unittest.TestCase):
    """Database group of a scene token."""

    def test_drops_the_version_of_the_database(self) -> None:
        """
        Input: scene tokens of several versions of one database.
        Expected: every version maps onto the database name without ``_v<N>``.
        Check: the group of each token.
        """
        for version in ("v1", "v3", "v12"):
            with self.subTest(version=version):
                self.assertEqual(database_group(f"db_j6gen2_{version}/uuid/0"), "db_j6gen2")

    def test_keeps_databases_that_share_a_prefix_apart(self) -> None:
        """
        Input: scene tokens of db_jpntaxi_v1 and db_jpntaxigen2_v1, one name a prefix of the
        other.
        Expected: the two vehicle generations land in two groups.
        Check: the group of each token.
        """
        self.assertEqual(database_group("db_jpntaxi_v1/uuid/0"), "db_jpntaxi")
        self.assertEqual(database_group("db_jpntaxigen2_v1/uuid/0"), "db_jpntaxigen2")

    def test_keeps_a_database_without_version(self) -> None:
        """
        Input: a scene token whose database has no version suffix.
        Expected: the database name is the group.
        Check: the group of the token.
        """
        self.assertEqual(database_group("db_default/uuid/0"), "db_default")


class TestPolygons(unittest.TestCase):
    """Validation of the normalized polygons and their pixel masks."""

    def test_rejects_malformed_polygons(self) -> None:
        """
        Input: an odd number of values, two vertices, and a coordinate outside [0, 1].
        Expected: each is rejected at configuration time instead of painting a wrong region.
        Check: normalize_polygon raises ValueError.
        """
        for polygon in ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.5, 0.0, 0.0, 1.0]):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                normalize_polygon(polygon)

    def test_rasterizes_onto_the_pixel_grid(self) -> None:
        """
        Input: the left half of a 4x5 image.
        Expected: normalized 0 and 1 are the first and last pixel, so x = 0.5 is column 2 and
        columns 0 to 2 of every row are inside.
        Check: the mask against the expected pixel set.
        """
        mask = rasterize_polygon(normalize_polygon(LEFT_HALF), height=4, width=5)

        expected = torch.zeros((4, 5), dtype=torch.bool)
        expected[:, :3] = True
        self.assertTrue(torch.equal(mask, expected))


class MaskEgoVehicleTestCase(PinholeCameraTestCase):
    """Shared fixtures for the MaskEgoVehicle tests."""

    HEIGHT = 4
    WIDTH = 5

    def build_frame(self, scene_token: str | None, num_cameras: int = 2) -> ModelGTSample:
        """A sample of mid-gray cameras CAM_0, CAM_1... recorded in the given scene."""
        camera_image_data = self.build_pinhole_images(
            [self.LIDAR2CAM_FORWARD] * num_cameras, width=self.WIDTH, height=self.HEIGHT
        )
        camera_image_data = camera_image_data.model_copy(
            update={"images": torch.full_like(camera_image_data.images, 0.5)}
        )
        frame_meta = (
            None
            if scene_token is None
            else FrameMetaSample(ego2global=torch.eye(4), scene_token=scene_token, timestamp=0.0)
        )
        return self.build_sample(camera_image_data)._replace(frame_meta=frame_meta)


class TestMaskEgoVehicle(MaskEgoVehicleTestCase):
    """Painting of the ego vehicle regions of a frame."""

    def test_paints_the_polygon_of_the_configured_camera(self) -> None:
        """
        Input: a frame of db_j6gen2_v3 and a left-half polygon for CAM_1 of db_j6gen2.
        Expected: the pixels of CAM_1 inside the polygon are set to the fill value, its other
        pixels and every pixel of CAM_0 keep their value, and the input sample is untouched.
        Check: the painted images against the expected images.
        """
        sample = self.build_frame("db_j6gen2_v3/uuid/0")
        transform = MaskEgoVehicle(polygons={"db_j6gen2": {"CAM_1": LEFT_HALF}})

        masked = transform(sample)

        expected = torch.full((2, 3, self.HEIGHT, self.WIDTH), 0.5)
        expected[1, :, :, :3] = 0.0
        self.assertTrue(torch.equal(masked.camera_image_data.images, expected))
        self.assertTrue(torch.all(sample.camera_image_data.images == 0.5))

    def test_paints_one_value_per_channel(self) -> None:
        """
        Input: a per-channel fill value and a bottom-row polygon for CAM_0.
        Expected: each channel of the bottom row takes its own value.
        Check: the bottom row of CAM_0, channel by channel.
        """
        sample = self.build_frame("db_largebus_v1/uuid/0")
        transform = MaskEgoVehicle(
            polygons={"db_largebus": {"CAM_0": BOTTOM_ROW}}, fill_value=[0.1, 0.2, 0.3]
        )

        images = transform(sample).camera_image_data.images

        for channel, value in enumerate([0.1, 0.2, 0.3]):
            with self.subTest(channel=channel):
                np.testing.assert_allclose(images[0, channel, -1].numpy(), value)
                np.testing.assert_allclose(images[0, channel, :-1].numpy(), 0.5)

    def test_picks_the_polygons_of_the_group_of_the_frame(self) -> None:
        """
        Input: polygons of db_jpntaxi on CAM_0 and of db_jpntaxigen2 on CAM_1, and a frame of
        db_jpntaxigen2_v2.
        Expected: only the polygon of the frame's own group is painted, the database of the
        other generation sharing its name prefix does not match.
        Check: CAM_0 keeps its value and CAM_1 is painted.
        """
        sample = self.build_frame("db_jpntaxigen2_v2/uuid/0")
        transform = MaskEgoVehicle(
            polygons={
                "db_jpntaxi": {"CAM_0": LEFT_HALF},
                "db_jpntaxigen2": {"CAM_1": LEFT_HALF},
            }
        )

        images = transform(sample).camera_image_data.images

        self.assertTrue(torch.all(images[0] == 0.5))
        self.assertTrue(torch.all(images[1, :, :, :3] == 0.0))

    def test_prefers_the_polygons_of_the_versioned_database(self) -> None:
        """
        Input: polygons of db_jpntaxigen2 on CAM_0 and of db_jpntaxigen2_v2 on CAM_1, and a frame
        of db_jpntaxigen2_v2 and one of db_jpntaxigen2_v1.
        Expected: a platform can change its sensor mounting between versions, so the polygons of
        the versioned name win for v2 and the other versions fall back to the group.
        Check: the painted camera of each frame.
        """
        transform = MaskEgoVehicle(
            polygons={
                "db_jpntaxigen2": {"CAM_0": LEFT_HALF},
                "db_jpntaxigen2_v2": {"CAM_1": LEFT_HALF},
            }
        )
        for version, painted in (("v2", 1), ("v1", 0)):
            with self.subTest(version=version):
                images = transform(
                    self.build_frame(f"db_jpntaxigen2_{version}/uuid/0")
                ).camera_image_data.images
                self.assertTrue(torch.all(images[painted, :, :, :3] == 0.0))
                self.assertTrue(torch.all(images[1 - painted] == 0.5))

    def test_leaves_unconfigured_frames_as_they_are(self) -> None:
        """
        Input: a frame of a group without polygons and a frame without frame metadata.
        Expected: neither is painted, so a dataset that mixes platforms only masks the
        configured ones.
        Check: the transform hands back the input sample.
        """
        transform = MaskEgoVehicle(polygons={"db_j6gen2": {"CAM_0": LEFT_HALF}})
        for scene_token in ("db_j6_v1/uuid/0", None):
            with self.subTest(scene_token=scene_token):
                sample = self.build_frame(scene_token)
                self.assertIs(transform(sample), sample)

    def test_follows_the_camera_names_rather_than_their_order(self) -> None:
        """
        Input: a frame whose cameras are served in the order CAM_1, CAM_0, as after
        ShuffleCameraOrder, and a polygon for CAM_0.
        Expected: the polygon is painted on the camera named CAM_0, at position 1.
        Check: position 0 keeps its value and position 1 is painted.
        """
        sample = self.build_frame("db_j6gen2_v1/uuid/0")
        sample = sample._replace(
            camera_image_data=sample.camera_image_data.model_copy(
                update={"camera_names": ["CAM_1", "CAM_0"]}
            )
        )
        transform = MaskEgoVehicle(polygons={"db_j6gen2": {"CAM_0": LEFT_HALF}})

        images = transform(sample).camera_image_data.images

        self.assertTrue(torch.all(images[0] == 0.5))
        self.assertTrue(torch.all(images[1, :, :, :3] == 0.0))

    def test_rejects_a_camera_that_was_already_augmented(self) -> None:
        """
        Input: a frame whose CAM_1 already carries an image augmentation (a horizontal flip).
        Expected: a polygon drawn on the raw image would land on the wrong pixels, so masking
        CAM_1 raises, while masking only CAM_0, which carries none, still works.
        Check: ValueError for CAM_1, a painted CAM_0 otherwise.
        """
        sample = self.build_frame("db_j6gen2_v1/uuid/0")
        flip = torch.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = self.WIDTH - 1
        augmentation_matrices = sample.camera_image_data.image_augmentation_matrices.clone()
        augmentation_matrices[1] = flip
        sample = sample._replace(
            camera_image_data=sample.camera_image_data.model_copy(
                update={"image_augmentation_matrices": augmentation_matrices}
            )
        )

        with self.assertRaises(ValueError):
            MaskEgoVehicle(polygons={"db_j6gen2": {"CAM_1": LEFT_HALF}})(sample)
        images = MaskEgoVehicle(polygons={"db_j6gen2": {"CAM_0": LEFT_HALF}})(
            sample
        ).camera_image_data.images
        self.assertTrue(torch.all(images[0, :, :, :3] == 0.0))


# The middle of the image: pixels 80 to 119 of a 200 pixel wide image, 30 to 69 of 100 rows.
CENTER = [0.4, 0.3, 0.6, 0.3, 0.6, 0.7, 0.4, 0.7]
# The bottom fifth of the image, down to its last row.
BOTTOM_STRIP = [0.0, 0.8, 1.0, 0.8, 1.0, 1.0, 0.0, 1.0]
# A U open at the top: two arms (pixels 0 to 60 and 139 to 199, rows 30 to 99) joined by a band
# along the bottom (rows 69 to 99), around a notch at pixels 60 to 139, rows 30 to 69.
U_SHAPE = [0.0, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.3, 1.0, 0.3, 1.0, 1.0, 0.0, 1.0]


class TestBBoxesEgoVehicleFilter(PinholeCameraTestCase):
    """Dropping of the boxes the ego vehicle hides from every camera that sees them.

    Every camera looks along the lidar x axis with fx = fy = 100 on a 200x100 image, so a point
    (x, y, z) in front of it lands on pixel (100 - 100 y / x, 50 - 100 z / x).
    """

    WIDTH = 200
    HEIGHT = 100

    def build_frame(
        self,
        rows: list[list[float]],
        num_cameras: int = 1,
        scene_token: str = "db_j6gen2_v1/uuid/0",
    ) -> ModelGTSample:
        """A frame of forward cameras CAM_0, CAM_1... and gravity-center boxes."""
        camera_image_data = self.build_pinhole_images(
            [self.LIDAR2CAM_FORWARD] * num_cameras, width=self.WIDTH, height=self.HEIGHT
        )
        return self.build_sample(camera_image_data)._replace(
            detection3d_gt_bboxes_3d=self.build_boxes(rows, list(range(len(rows)))),
            frame_meta=FrameMetaSample(
                ego2global=torch.eye(4), scene_token=scene_token, timestamp=0.0
            ),
        )

    @staticmethod
    def kept_labels(sample: ModelGTSample) -> list[int]:
        """Labels of the boxes left in the sample, one label per input box index."""
        return sample.detection3d_gt_bboxes_3d.bbox_labels.tolist()

    def test_drops_a_box_hidden_where_it_is_seen(self) -> None:
        """
        Input: one camera with a region in the middle of the image (pixels 80 to 119), a 1 m
        box 10 m ahead that lands inside it (about pixels 95 to 105), one 3 m to the left that
        lands beside it (about 63 to 76), one 2 m to the left that straddles its edge (about 74
        to 86) and one behind the camera.
        Expected: only the box inside the region is dropped. The box behind the camera is seen
        by no camera, so it is not the ego vehicle that hides it.
        Check: the labels of the kept boxes.
        """
        rows = [
            [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [10.0, 3.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [10.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [-10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        ]
        transform = BBoxesEgoVehicleFilter(polygons={"db_j6gen2": {"CAM_0": CENTER}})

        self.assertEqual(self.kept_labels(transform(self.build_frame(rows))), [1, 2, 3])

    def test_keeps_a_box_another_camera_sees(self) -> None:
        """
        Input: the box inside the region of CAM_0, and a second camera CAM_1 without a region
        that sees the same box.
        Expected: CAM_1 still sees the box, so it is kept.
        Check: the labels of the kept boxes.
        """
        rows = [[10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
        transform = BBoxesEgoVehicleFilter(polygons={"db_j6gen2": {"CAM_0": CENTER}})

        self.assertEqual(self.kept_labels(transform(self.build_frame(rows, num_cameras=2))), [0])

    def test_drops_a_box_that_leaves_the_image_inside_the_region(self) -> None:
        """
        Input: a region over the bottom fifth of the image, down to its last row, and a box
        2 m ahead, 1 m below the camera, whose projection runs past the bottom of the image.
        Expected: the extent is clipped to the last pixel row, which the region covers, so the
        box is dropped.
        Check: no box is kept.
        """
        rows = [[2.0, 0.0, -1.0, 0.4, 0.4, 0.4, 0.0]]
        transform = BBoxesEgoVehicleFilter(polygons={"db_j6gen2": {"CAM_0": BOTTOM_STRIP}})

        self.assertEqual(self.kept_labels(transform(self.build_frame(rows))), [])

    def test_keeps_a_box_that_spans_the_notch_of_a_concave_region(self) -> None:
        """
        Input: a U-shaped region, a box 2 m ahead whose extent (about pixels 20 to 180, rows 40
        to 60) has its four corners in the two arms and its middle in the notch, and a box whose
        extent (about pixels 150 to 190, rows 75 to 90) lies in the right arm.
        Expected: the corners alone would call both hidden, but the edges of the notch cross the
        first extent, which is therefore partly visible and kept, while the second is dropped.
        Check: the labels of the kept boxes.
        """
        rows = [
            [2.0, 0.0, 0.0, 0.02, 3.2, 0.4, 0.0],
            [2.0, -1.4, -0.65, 0.02, 0.8, 0.3, 0.0],
        ]
        transform = BBoxesEgoVehicleFilter(polygons={"db_j6gen2": {"CAM_0": U_SHAPE}})

        self.assertEqual(self.kept_labels(transform(self.build_frame(rows))), [0])

    def test_leaves_the_boxes_of_other_groups(self) -> None:
        """
        Input: the box inside the region of CAM_0 of db_j6gen2, in a frame of db_largebus_v1.
        Expected: the frame has no region, so its boxes are left as they are.
        Check: the labels of the kept boxes.
        """
        rows = [[10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
        transform = BBoxesEgoVehicleFilter(polygons={"db_j6gen2": {"CAM_0": CENTER}})

        sample = self.build_frame(rows, scene_token="db_largebus_v1/uuid/0")
        self.assertEqual(self.kept_labels(transform(sample)), [0])

    def test_rejects_a_camera_that_was_already_augmented(self) -> None:
        """
        Input: a frame whose camera with a region already carries an image augmentation.
        Expected: the region is drawn on the raw image, so the filter raises instead of testing
        the boxes against the wrong pixels.
        Check: ValueError.
        """
        sample = self.build_frame([[10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        augmentation_matrices = sample.camera_image_data.image_augmentation_matrices.clone()
        augmentation_matrices[0, 0, 0] = 0.5
        sample = sample._replace(
            camera_image_data=sample.camera_image_data.model_copy(
                update={"image_augmentation_matrices": augmentation_matrices}
            )
        )
        transform = BBoxesEgoVehicleFilter(polygons={"db_j6gen2": {"CAM_0": CENTER}})

        with self.assertRaises(ValueError):
            transform(sample)


if __name__ == "__main__":
    unittest.main()
