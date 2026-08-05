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

"""Tests for the crop box filter."""

import numpy as np
import pytest

from autoware_ml.transforms.point_cloud.crop_box import CropBoxFilter, load_crop_boxes
from autoware_ml.transforms.point_cloud.ego_motion import PRE_CORRECTION_POINTS_KEY

_TEST_CROP_BOXES = [
    [-1.5, -1.2, 0.0, 5.7, 1.2, 3.0],  # body-like
    [4.4, -1.2, 0.0, 5.1, 1.2, 0.8],  # wheels-like
]


class TestCropBoxFilter:
    def test_crop_box_filter_removes_points_inside_any_box_and_keeps_arrays_aligned(self):
        sample = {
            "points": np.array(
                [
                    [0.0, 0.0, 0.5, 1.0, 0.0],  # inside the vehicle body
                    [4.7, 0.0, 0.4, 2.0, 0.0],  # inside the wheels box
                    [10.0, 0.0, 0.5, 3.0, 0.0],  # ahead of the vehicle
                    [0.0, 0.0, -0.1, 4.0, 0.0],  # below the box floor (ground return)
                ],
                dtype=np.float32,
            ),
            "labels": np.array([10, 11, 12, 13], dtype=np.int64),
        }

        output = CropBoxFilter(crop_boxes=_TEST_CROP_BOXES)(sample)

        np.testing.assert_allclose(
            output["points"],
            np.array(
                [[10.0, 0.0, 0.5, 3.0, 0.0], [0.0, 0.0, -0.1, 4.0, 0.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(output["labels"], np.array([12, 13], dtype=np.int64))

    def test_crop_box_filter_decides_on_pre_correction_points(self):
        # The point sits outside the boxes where T4Dataset put it, but inside them where it
        # actually was when captured. The vehicle crops pre-correction, so it must be dropped --
        # and the surviving point must keep its corrected coordinates, not the pre-correction ones.
        sample = {
            "points": np.array(
                [[0.0, 0.0, 0.5, 1.0, 0.0], [30.0, 0.0, 0.5, 2.0, 0.0]], dtype=np.float32
            ),
            PRE_CORRECTION_POINTS_KEY: np.array(
                [[30.0, 0.0, 0.5], [1.0, 0.0, 0.5]], dtype=np.float32
            ),
        }

        output = CropBoxFilter(crop_boxes=_TEST_CROP_BOXES)(sample)

        np.testing.assert_allclose(
            output["points"], np.array([[0.0, 0.0, 0.5, 1.0, 0.0]], dtype=np.float32)
        )
        # The parallel array is filtered by the same mask, so the two never drift apart.
        assert output[PRE_CORRECTION_POINTS_KEY].shape[0] == output["points"].shape[0]
        np.testing.assert_allclose(
            output[PRE_CORRECTION_POINTS_KEY], np.array([[30.0, 0.0, 0.5]], dtype=np.float32)
        )

    def test_filters_reject_misaligned_pre_correction_points(self):
        sample = {
            "points": np.zeros((3, 5), dtype=np.float32),
            PRE_CORRECTION_POINTS_KEY: np.zeros((2, 3), dtype=np.float32),
        }

        with pytest.raises(ValueError, match="aligned"):
            CropBoxFilter(crop_boxes=_TEST_CROP_BOXES)(sample)

    def test_crop_box_filter_rejects_malformed_boxes(self):
        with pytest.raises(ValueError, match="at least one box"):
            CropBoxFilter(crop_boxes=[])
        with pytest.raises(ValueError, match="min <= max"):
            CropBoxFilter(crop_boxes=[[1.0, 0.0, 0.0, -1.0, 1.0, 1.0]])

    def test_bundled_crop_boxes_asset_loads(self):
        # Pins the values in configs/assets/aip_x2_gen2/crop_boxes.param.yaml, whose header records
        # how they were derived from that platform's vehicle dimensions.
        boxes = load_crop_boxes("aip_x2_gen2/crop_boxes.param.yaml")

        assert len(boxes) == 2
        np.testing.assert_allclose(
            boxes[0], [-1.525790, -1.226830, 0.0, 5.711110, 1.200580, 3.080], atol=1e-6
        )
        np.testing.assert_allclose(
            boxes[1], [4.372418, -1.225794, 0.0, 5.147822, 1.225794, 0.8195], atol=1e-6
        )
        # Usable directly as the transform's only configuration.
        assert CropBoxFilter(crop_boxes=boxes).crop_boxes.shape == (2, 6)
