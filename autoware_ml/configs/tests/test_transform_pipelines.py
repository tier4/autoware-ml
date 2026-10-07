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

"""Invariants every transform pipeline config keeps."""

from pathlib import Path

import pytest
import yaml

_TRANSFORMS = Path(__file__).resolve().parents[1] / "datamodule" / "transforms"
_LIDAR_AUGMENTATIONS = (
    "point_cloud.geometry.GlobalRotScaleTrans",
    "point_cloud.geometry.GlobalBEVRandomFlip",
    "point_cloud.geometry.RandomRotateTargetAngle",
)


def targets(path: Path) -> list[str]:
    """Transform class paths of a pipeline config, in order."""
    return [step["_target_"] for step in yaml.safe_load(path.read_text())["pipeline"]]


@pytest.mark.parametrize("path", sorted(_TRANSFORMS.glob("*.yaml")), ids=lambda p: p.stem)
def test_cameras_load_before_the_lidar_augmentations_move_their_calibration(path: Path) -> None:
    steps = targets(path)
    camera = [i for i, t in enumerate(steps) if t.endswith("camera.loading.LoadImagesFromFile")]
    lidar = [i for i, t in enumerate(steps) if t.endswith(_LIDAR_AUGMENTATIONS)]
    if camera and lidar:
        assert camera[0] < lidar[0]


@pytest.mark.parametrize(
    "path",
    [
        p
        for p in sorted(_TRANSFORMS.glob("*_test_transforms.yaml"))
        if "detection3d" in p.stem or "segdet3d" in p.stem
    ],
    ids=lambda p: p.stem,
)
def test_detection_evaluation_drops_the_boxes_of_ignored_classes(path: Path) -> None:
    assert any(t.endswith("boxes3d.filters.BBoxesLabelNameFilter") for t in targets(path))
