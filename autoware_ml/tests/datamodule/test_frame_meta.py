"""Tests for the per-frame evaluation metadata helpers."""

from __future__ import annotations

import pytest

from autoware_ml.datamodule.common.frame_meta import scene_dir_fragment

ROOT = "/data/t4"


def test_scene_dir_fragment() -> None:
    path = "db_j6gen2_v2/13cabeac-a81b/0/data/LIDAR_CONCAT/00000.pcd.bin"
    assert scene_dir_fragment(path, ROOT) == "db_j6gen2_v2/13cabeac-a81b/0"


def test_scene_dir_fragment_strips_data_root_prefix() -> None:
    path = "/data/t4/db_j6gen2_v2/13cabeac-a81b/0/data/LIDAR_CONCAT/00000.pcd.bin"
    assert scene_dir_fragment(path, ROOT) == "db_j6gen2_v2/13cabeac-a81b/0"
    relative_root = "data/t4"
    prefixed = "data/t4/db_j6gen2_v2/13cabeac-a81b/0/data/LIDAR_CONCAT/00000.pcd.bin"
    assert scene_dir_fragment(prefixed, relative_root) == "db_j6gen2_v2/13cabeac-a81b/0"


def test_scene_dir_fragment_rejects_absolute_path_outside_root() -> None:
    with pytest.raises(ValueError, match="absolute path must live under data_root"):
        scene_dir_fragment("/elsewhere/db/uuid/0/data/00000.pcd.bin", ROOT)


def test_scene_dir_fragment_rejects_short_paths() -> None:
    with pytest.raises(ValueError, match="scene directory"):
        scene_dir_fragment("no_scene.bin", ROOT)
