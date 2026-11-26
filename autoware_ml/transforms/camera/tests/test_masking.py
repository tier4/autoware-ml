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

"""Unit tests for the masking of the ego vehicle regions."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.transforms.camera.masking import (
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


if __name__ == "__main__":
    unittest.main()
