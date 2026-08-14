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

from autoware_ml.transforms.point_cloud.assets import resolve_asset_path
from autoware_ml.transforms.point_cloud.nebula_mask_assets import (
    dither_mask,
    load_calibration,
    load_lidar_masks,
    normalize_model_name,
    read_mask_manifest,
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
    # Hesai ships metadata rows ahead of the real header; the loader must skip them. The Azimuth
    # column is present in real files and deliberately unused.
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
            lidars={
                "front_upper": {
                    "mask": "front_upper/mask.png",
                    "model": "pandar_qt128",
                    "calibration": "model.csv",
                }
            },
        )

        assert set(loaded) == {"front_upper"}
        entry = loaded["front_upper"]
        assert entry.keep.shape == (2, 4)
        # Fully-white row is kept everywhere, fully-black row nowhere.
        assert entry.keep[0].all()
        assert not entry.keep[1].any()
        np.testing.assert_allclose(entry.elevation_rad, np.deg2rad([10.0, -10.0]), atol=1e-6)

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
                lidars={
                    "front_upper": {
                        "mask": "front_upper/mask.png",
                        "model": "pandar_qt128",
                        "calibration": "three.csv",
                    }
                },
            )

    def test_load_lidar_masks_reports_a_missing_mask_file(self, mask_tree):
        masks, calibrations = mask_tree

        with pytest.raises(FileNotFoundError, match="Could not read Nebula downsample mask"):
            load_lidar_masks(
                masks,
                calibration_root=calibrations,
                lidars={
                    "front_upper": {
                        "mask": "front_upper/absent.png",
                        "model": "pandar_qt128",
                        "calibration": "model.csv",
                    }
                },
            )

    def test_load_lidar_masks_defaults_calibration_to_the_configured_model(self, mask_tree):
        # Omitting the calibration must follow the model, never a table left over from some other
        # mapping: both bundled tables have 128 channels, so a shape check would not notice.
        masks, calibrations = mask_tree
        (calibrations / "pandar_qt128.csv").write_text(
            "Channel,Elevation,Azimuth\n1,20.0,0.5\n2,-20.0,-0.5\n"
        )

        loaded = load_lidar_masks(
            masks,
            calibration_root=calibrations,
            lidars={"front_upper": {"mask": "front_upper/mask.png", "model": "qt128"}},
        )

        np.testing.assert_allclose(
            loaded["front_upper"].elevation_rad, np.deg2rad([20.0, -20.0]), atol=1e-6
        )

    def test_load_lidar_masks_requires_a_manifest(self, mask_tree):
        # A directory of masks says nothing about which sensor sits where, and there is no
        # platform-neutral default to guess with.
        masks, _ = mask_tree

        with pytest.raises(FileNotFoundError, match="No lidar_masks.param.yaml"):
            load_lidar_masks(masks)

    def test_bundled_manifest_covers_every_bundled_mask(self):
        # The manifest is the platform's own record of its layout, so it has to name every mask
        # directory shipped beside it.
        mask_dir = resolve_asset_path("aip_x2_gen2")
        entries = read_mask_manifest(mask_dir)

        assert set(entries) == {path.name for path in mask_dir.iterdir() if path.is_dir()}
        for name, entry in entries.items():
            assert (mask_dir / entry["mask"]).is_file(), name
            assert normalize_model_name(entry["model"]) in {"pandar128e4x", "pandar_qt128"}, name

    def test_load_calibration_ignores_the_azimuth_column(self, tmp_path):
        # Azimuth corrections are never applied by Autoware-ML, so the loader does not return them
        # and a file without that column still reads cleanly.
        path = tmp_path / "no_azimuth.csv"
        path.write_text("Channel,Elevation\n1,5.0\n2,-5.0\n")

        np.testing.assert_allclose(load_calibration(path), np.deg2rad([5.0, -5.0]), atol=1e-6)

    def test_load_calibration_requires_a_header(self, tmp_path):
        path = tmp_path / "headerless.csv"
        path.write_text("1,2.0,3.0\n")

        with pytest.raises(ValueError, match="no Elevation header"):
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
        ],
    )
    def test_normalize_model_name(self, alias, expected):
        assert normalize_model_name(alias) == expected

    def test_normalize_model_name_rejects_an_unknown_model(self):
        # Accepting it would pick a dither pattern and an elevation table by accident, and the
        # 128-row channel check passes for either bundled sensor, so nothing downstream would object.
        with pytest.raises(ValueError, match="Unsupported Hesai model"):
            normalize_model_name("pandar_xt32")

    def test_dither_mask_rejects_an_unknown_model(self):
        with pytest.raises(ValueError, match="Unsupported Hesai model"):
            dither_mask(np.full((4, 4), 128, dtype=np.uint8), "pandar_xt32")

    def test_dither_mask_spreads_partial_keep_ratios(self):
        # A uniform mid-grey image keeps roughly half the cells, spread out rather than clustered.
        image = np.full((10, 10), 128, dtype=np.uint8)

        kept = dither_mask(image, "pandar_qt128")

        assert 0.3 < kept.mean() < 0.7
        # Every row sees some survivors, i.e. the pattern is not confined to a few rows.
        assert kept.any(axis=1).all()
