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

"""Tests for splitting a concatenated cloud back into its LiDARs."""

import numpy as np
import pytest

from autoware_ml.transforms.point_cloud.lidar_sources import (
    infer_lidar_name_from_text,
    iter_pointcloud_sources,
    normalize_lidar_name,
)


def concat_sample(*ranges: tuple[str, int, int], extrinsics: bool = True) -> dict:
    """A sample whose concat metadata places each named source at a given range."""
    sources = {}
    for name, _, _ in ranges:
        meta = {"sensor_token": name}
        if extrinsics:
            meta |= {"translation": [0.0, 0.0, 0.0], "rotation": np.eye(3).tolist()}
        sources[f"LIDAR_{name.upper()}"] = meta
    return {
        "lidar_sources": sources,
        "lidar_sources_info": {
            "sources": [
                {"sensor_token": name, "idx_begin": begin, "length": length}
                for name, begin, length in ranges
            ]
        },
    }


class TestIterPointcloudSources:
    def test_splits_a_tiling_concat_cloud(self):
        sample = concat_sample(("front_upper", 0, 4), ("rear_upper", 4, 6))

        sources = iter_pointcloud_sources(sample, 10)

        assert {source.name for source in sources} == {"LIDAR_FRONT_UPPER", "LIDAR_REAR_UPPER"}
        assert sorted((s.point_slice.start, s.point_slice.stop) for s in sources) == [
            (0, 4),
            (4, 10),
        ]

    def test_accepts_a_source_that_contributed_nothing(self):
        # A configured LiDAR absent from the recorded ranges is fine as long as the rest still
        # account for every point.
        sample = concat_sample(("front_upper", 0, 10))
        sample["lidar_sources"]["LIDAR_REAR_UPPER"] = {
            "sensor_token": "rear_upper",
            "translation": [0.0, 0.0, 0.0],
            "rotation": np.eye(3).tolist(),
        }

        assert len(iter_pointcloud_sources(sample, 10)) == 1

    def test_rejects_a_cloud_longer_than_the_recorded_ranges(self):
        # The shape of a multi-sweep cloud: sweeps appended to points, metadata still describing
        # only the keyframe. Masking those rows against no source would silently discard them.
        sample = concat_sample(("front_upper", 0, 4), ("rear_upper", 4, 6))

        with pytest.raises(ValueError, match="covers 10 of 20 points"):
            iter_pointcloud_sources(sample, 20)

    def test_names_appended_sweeps_as_the_likely_cause(self):
        sample = concat_sample(("front_upper", 0, 5))

        with pytest.raises(ValueError, match="appended sweeps"):
            iter_pointcloud_sources(sample, 15)

    def test_rejects_a_gap_between_ranges(self):
        sample = concat_sample(("front_upper", 0, 4), ("rear_upper", 6, 4))

        with pytest.raises(ValueError, match="covers 8 of 10 points"):
            iter_pointcloud_sources(sample, 10)

    def test_rejects_overlapping_ranges(self):
        sample = concat_sample(("front_upper", 0, 6), ("rear_upper", 4, 6))

        with pytest.raises(ValueError, match="overlap at index 4"):
            iter_pointcloud_sources(sample, 10)

    def test_rejects_a_range_past_the_end_of_the_cloud(self):
        sample = concat_sample(("front_upper", 0, 4), ("rear_upper", 4, 20))

        with pytest.raises(ValueError, match="does not fit a cloud of 10 points"):
            iter_pointcloud_sources(sample, 10)

    def test_rejects_missing_extrinsics(self):
        # Without these the caller cannot reach the sensor frame; numpy would otherwise turn the
        # absent values into 0-d NaN arrays that fail obscurely inside a matmul.
        sample = concat_sample(("front_upper", 0, 10), extrinsics=False)

        with pytest.raises(KeyError, match="missing its extrinsics"):
            iter_pointcloud_sources(sample, 10)

    def test_rejects_metadata_matching_no_configured_source(self):
        sample = concat_sample(("front_upper", 0, 10))
        sample["lidar_sources_info"]["sources"][0]["sensor_token"] = "unconfigured"

        with pytest.raises(ValueError, match="did not match any configured"):
            iter_pointcloud_sources(sample, 10)

    def test_single_source_samples_need_no_ranges(self):
        sources = iter_pointcloud_sources({"source_name": "LIDAR_REAR_LOWER"}, 7)

        assert len(sources) == 1
        assert sources[0].point_slice == slice(0, 7)
        assert sources[0].translation is None

    def test_single_source_name_can_come_from_the_sample_name(self):
        sources = iter_pointcloud_sources({"name": "scene_0_left_lower_00001"}, 3)

        assert sources[0].name == "left_lower"

    def test_requires_some_way_to_name_the_source(self):
        with pytest.raises(KeyError, match="require concat 'lidar_sources' metadata"):
            iter_pointcloud_sources({"name": "no-position-in-here"}, 3)


class TestNaming:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("LIDAR_FRONT_UPPER", "front_upper"), ("front_upper", "front_upper"), ("REAR", "rear")],
    )
    def test_normalize_lidar_name(self, raw, expected):
        assert normalize_lidar_name(raw) == expected

    def test_infer_lidar_name_from_text(self):
        assert infer_lidar_name_from_text("path/to/LIDAR_RIGHT_LOWER/0.bin") == "right_lower"
        assert infer_lidar_name_from_text("nothing here") is None
