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

"""Tests for loading Nebula masks and Hesai calibration from disk."""

import numpy as np
import pytest

from autoware_ml.transforms.point_cloud.nebula_mask_assets import (
    dither_mask,
    load_calibration,
    load_lidar_masks,
    normalize_model_name,
)


@pytest.fixture
def mask_tree(tmp_path):
    """A minimal on-disk asset tree: one 2-channel mask plus its calibration."""
    cv2 = pytest.importorskip("cv2")
    masks = tmp_path / "masks"
    (masks / "front_upper").mkdir(parents=True)
    # Row 0 fully kept (255), row 1 fully dropped (0).
    image = np.zeros((2, 4), dtype=np.uint8)
    image[0, :] = 255
    cv2.imwrite(str(masks / "front_upper" / "mask.png"), image)

    calibrations = tmp_path / "calibrations"
    calibrations.mkdir()
    # Hesai ships metadata rows ahead of the real header; the loader must skip them.
    (calibrations / "model.csv").write_text(
        "Hesai\nsome,metadata\nChannel,Elevation,Azimuth\n1,10.0,0.5\n2,-10.0,-0.5\n"
    )
    return masks, calibrations


class TestNebulaMaskAssets:
    def test_load_lidar_masks_pairs_masks_with_calibration(self, mask_tree):
        masks, calibrations = mask_tree

        loaded = load_lidar_masks(
            masks,
            calibration_root=calibrations,
            lidar_name_to_mask={"front_upper": "front_upper/mask.png"},
            lidar_name_to_model={"front_upper": "pandar_qt128"},
            lidar_name_to_calibration={"front_upper": "model.csv"},
        )

        assert set(loaded) == {"front_upper"}
        entry = loaded["front_upper"]
        assert entry.keep.shape == (2, 4)
        # Fully-white row is kept everywhere, fully-black row nowhere.
        assert entry.keep[0].all()
        assert not entry.keep[1].any()
        np.testing.assert_allclose(entry.elevation_rad, np.deg2rad([10.0, -10.0]), atol=1e-6)
        np.testing.assert_allclose(entry.azimuth_deg, [0.5, -0.5], atol=1e-6)

    def test_load_lidar_masks_rejects_channel_count_mismatch(self, mask_tree):
        masks, calibrations = mask_tree
        # Calibration with three channels against a two-row mask.
        (calibrations / "three.csv").write_text(
            "Channel,Elevation,Azimuth\n1,1.0,0.0\n2,0.0,0.0\n3,-1.0,0.0\n"
        )

        with pytest.raises(ValueError, match="rows, but .* calibration has"):
            load_lidar_masks(
                masks,
                calibration_root=calibrations,
                lidar_name_to_mask={"front_upper": "front_upper/mask.png"},
                lidar_name_to_model={"front_upper": "pandar_qt128"},
                lidar_name_to_calibration={"front_upper": "three.csv"},
            )

    def test_load_lidar_masks_reports_a_missing_mask_file(self, mask_tree):
        masks, calibrations = mask_tree

        with pytest.raises(FileNotFoundError, match="Could not read Nebula downsample mask"):
            load_lidar_masks(
                masks,
                calibration_root=calibrations,
                lidar_name_to_mask={"front_upper": "front_upper/absent.png"},
                lidar_name_to_model={"front_upper": "pandar_qt128"},
                lidar_name_to_calibration={"front_upper": "model.csv"},
            )

    def test_load_calibration_requires_a_header(self, tmp_path):
        path = tmp_path / "headerless.csv"
        path.write_text("1,2.0,3.0\n")

        with pytest.raises(ValueError, match="no Elevation/Azimuth header"):
            load_calibration(path)

    def test_bundled_platform_masks_load(self):
        # Pins the bundled assets and their packaging: eight LiDARs, 128 channels each.
        loaded = load_lidar_masks("aip_x2_gen2")

        assert len(loaded) == 8
        assert loaded["front_upper"].keep.shape == (128, 3600)
        assert loaded["front_lower"].keep.shape == (128, 900)
        # The masks downsample, so a sensible fraction of cells survive.
        assert 0.05 < loaded["front_upper"].keep.mean() < 0.95

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("OT128", "pandar128e4x"),
            ("PANDAR128", "pandar128e4x"),
            ("QT128", "pandar_qt128"),
            ("pandarqt128", "pandar_qt128"),
            ("pandar-qt128", "pandar_qt128"),
            ("something_else", "something_else"),
        ],
    )
    def test_normalize_model_name(self, alias, expected):
        assert normalize_model_name(alias) == expected

    def test_dither_mask_spreads_partial_keep_ratios(self):
        # A uniform mid-grey image keeps roughly half the cells, spread out rather than clustered.
        image = np.full((10, 10), 128, dtype=np.uint8)

        kept = dither_mask(image, "pandar_qt128")

        assert 0.3 < kept.mean() < 0.7
        # Every row sees some survivors, i.e. the pattern is not confined to a few rows.
        assert kept.any(axis=1).all()
