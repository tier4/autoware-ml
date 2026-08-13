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

"""Tests for the Nebula downsample mask filter."""

import numpy as np
import pytest

from autoware_ml.transforms.point_cloud.nebula_mask import (
    LidarMask,
    NebulaDownsampleMaskFilter,
)


def lidar_mask(keep, *, elevation_deg=None) -> LidarMask:
    """Build a LidarMask from a keep grid, defaulting the calibration to flat zeros.

    Args:
        keep: ``(channels, azimuth_bins)`` array-like of booleans or 0/1.
        elevation_deg: Per-channel elevation. Defaults to all-zero, which makes the elevation-based
            ring estimate resolve every point to channel 0.
    """
    grid = np.asarray(keep, dtype=bool)
    channels = grid.shape[0]
    return LidarMask(
        keep=grid,
        elevation_rad=np.deg2rad(
            np.asarray(
                np.zeros(channels) if elevation_deg is None else elevation_deg, dtype=np.float32
            )
        ),
    )


def single_source_filter(keep, *, name="front_upper", **kwargs) -> NebulaDownsampleMaskFilter:
    """A filter configured with one LiDAR's mask."""
    return NebulaDownsampleMaskFilter(lidar_masks={name: lidar_mask(keep)}, **kwargs)


class TestNebulaDownsampleMaskFilter:
    def test_keeps_aligned_point_arrays(self):
        # One channel, two azimuth bins; only the first bin survives.
        sample = {
            "points": np.array([[0.0, 10.0, 0.0, 1.0], [10.0, 0.0, 0.0, 2.0]], dtype=np.float32),
            "labels": np.array([7, 8], dtype=np.int64),
            "source_name": "LIDAR_FRONT_UPPER",
        }

        output = single_source_filter([[1, 0]], channel_dim=None)(sample)

        # Azimuth is measured from +y, so the first point lands in bin 0 and the second in bin 1.
        np.testing.assert_allclose(output["points"], np.array([[0.0, 10.0, 0.0, 1.0]]))
        np.testing.assert_array_equal(output["labels"], np.array([7], dtype=np.int64))

    def test_uses_nebula_azimuth_convention(self):
        # Four bins of 90 deg. Nebula measures azimuth from +y toward +x, so a point on +y is at
        # 0 deg (bin 0) and a point on +x is at 90 deg (bin 1). Only bin 0 is kept.
        sample = {
            "points": np.array([[0.0, 10.0, 0.0, 1.0], [10.0, 0.0, 0.0, 2.0]], dtype=np.float32),
            "source_name": "LIDAR_FRONT_UPPER",
        }

        output = single_source_filter([[1, 0, 0, 0]], channel_dim=None)(sample)

        np.testing.assert_allclose(output["points"], np.array([[0.0, 10.0, 0.0, 1.0]]))

    def test_uses_concat_source_slices(self):
        # Two sources with opposite masks: the front keeps bin 0, the rear keeps bin 1.
        sample = {
            "points": np.array(
                [
                    [0.0, 10.0, 0.0, 1.0],  # front, bin 0 -> kept
                    [10.0, 0.0, 0.0, 2.0],  # front, bin 1 -> dropped
                    [0.0, 10.0, 0.0, 3.0],  # rear,  bin 0 -> dropped
                    [10.0, 0.0, 0.0, 4.0],  # rear,  bin 1 -> kept
                ],
                dtype=np.float32,
            ),
            "labels": np.array([1, 2, 3, 4], dtype=np.int64),
            "lidar_sources": {
                "LIDAR_FRONT_UPPER": {
                    "sensor_token": "front",
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": np.eye(3).tolist(),
                },
                "LIDAR_REAR_UPPER": {
                    "sensor_token": "rear",
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": np.eye(3).tolist(),
                },
            },
            "lidar_sources_info": {
                "sources": [
                    {"sensor_token": "front", "idx_begin": 0, "length": 2},
                    {"sensor_token": "rear", "idx_begin": 2, "length": 2},
                ]
            },
        }

        output = NebulaDownsampleMaskFilter(
            lidar_masks={
                "front_upper": lidar_mask([[1, 0, 0, 0]]),
                "rear_upper": lidar_mask([[0, 1, 0, 0]]),
            },
            channel_dim=None,
        )(sample)

        np.testing.assert_array_equal(output["labels"], np.array([1, 4], dtype=np.int64))

    def test_wraps_the_final_half_bin_of_a_full_circle_mask(self):
        # A ray just short of 360 deg rounds up to one past the last column, which on a full circle
        # is column 0 again. Azimuth is reconstructed from Cartesian coordinates, so a nominal
        # 0 deg ray can land on either side of the wrap and must not be dropped for it.
        azimuth = np.deg2rad(-0.04)
        sample = {
            "points": np.array(
                [[10.0 * np.sin(azimuth), 10.0 * np.cos(azimuth), 0.0, 1.0]], dtype=np.float32
            ),
            "source_name": "LIDAR_FRONT_UPPER",
        }

        kept = single_source_filter([[1, 0, 0, 0]], channel_dim=None)(dict(sample))

        assert kept["points"].shape[0] == 1

    def test_does_not_wrap_a_partial_extent_mask(self):
        # A mask covering only part of the circle has no column for a ray outside it, so beyond the
        # extent the point really is out of range.
        sample = {
            "points": np.array([[0.0, -10.0, 0.0, 1.0]], dtype=np.float32),
            "source_name": "LIDAR_FRONT_UPPER",
        }

        dropped = single_source_filter([[1, 1]], channel_dim=None, azimuth_extent_deg=90.0)(
            dict(sample)
        )

        assert dropped["points"].shape[0] == 0

    def test_prefers_the_stored_ring_index(self):
        # Only ring 1 is kept. Both points sit at elevation 0, so estimating the ring from
        # elevation would put both on ring 0 and drop them; the stored index must win.
        # Columns are [x, y, z, intensity, channel].
        sample = {
            "points": np.array(
                [[0.0, 10.0, 0.0, 1.0, 1.0], [0.0, 10.0, 0.0, 2.0, 0.0]], dtype=np.float32
            ),
            "source_name": "LIDAR_FRONT_UPPER",
        }
        keep = [[0, 0], [1, 0]]

        output = single_source_filter(keep, channel_dim=4)(dict(sample))
        np.testing.assert_allclose(output["points"], sample["points"][:1])

        # Opting out falls back to the elevation estimate, which puts both on ring 0 and drops them.
        fallback = single_source_filter(keep, channel_dim=None)(dict(sample))
        assert fallback["points"].shape[0] == 0

    def test_decides_on_pre_correction_points_when_present(self):
        # The point sits in the kept bin where T4Dataset put it, but in a dropped bin where it
        # actually was when captured. The vehicle masks pre-correction, so it must be dropped.
        sample = {
            "points": np.array([[0.0, 10.0, 0.0, 1.0]], dtype=np.float32),
            "pre_correction_points": np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
            "source_name": "LIDAR_FRONT_UPPER",
        }

        output = single_source_filter([[1, 0, 0, 0]], channel_dim=None)(sample)

        assert output["points"].shape[0] == 0

    def test_rejects_unknown_source(self):
        sample = {
            "points": np.zeros((1, 4), dtype=np.float32),
            "source_name": "LIDAR_REAR_LOWER",
        }

        with pytest.raises(KeyError, match="No Nebula mask configured"):
            single_source_filter([[1]], name="front_upper")(sample)

    def test_rejects_empty_masks(self):
        with pytest.raises(ValueError, match="must not be empty"):
            NebulaDownsampleMaskFilter(lidar_masks={})

    def test_lidar_mask_rejects_calibration_length_mismatch(self):
        with pytest.raises(ValueError, match="channels but calibration has"):
            LidarMask(keep=np.ones((4, 8), dtype=bool), elevation_rad=np.zeros(3, dtype=np.float32))
