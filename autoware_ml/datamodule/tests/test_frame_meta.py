"""Tests for the scene directory fragment the frame metadata is keyed by."""

from __future__ import annotations

import pytest

from autoware_ml.datamodule.t4dataset.frame_meta import scene_dir_fragment


def test_scene_dir_fragment() -> None:
    path = "db_j6gen2_v2/13cabeac-a81b/0/data/LIDAR_CONCAT/00000.pcd.bin"
    assert scene_dir_fragment(path) == "db_j6gen2_v2/13cabeac-a81b/0"


def test_scene_dir_fragment_rejects_absolute_paths() -> None:
    # The records store their paths relative to the database root, so a rooted path was not
    # written by this version of the generators
    with pytest.raises(ValueError, match="relative to the database root"):
        scene_dir_fragment("/data/t4/db_j6gen2_v2/13cabeac-a81b/0/data/LIDAR_CONCAT/00000.pcd.bin")


def test_scene_dir_fragment_rejects_short_paths() -> None:
    with pytest.raises(ValueError, match="scene directory"):
        scene_dir_fragment("no_scene.bin")
