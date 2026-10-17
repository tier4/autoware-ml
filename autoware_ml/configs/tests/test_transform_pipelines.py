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

"""Invariants every transform pipeline of an experiment config keeps."""

from pathlib import Path

import pytest
import yaml

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "tasks"
_LIDAR_AUGMENTATIONS = (
    "point_cloud.geometry.GlobalRotScaleTrans",
    "point_cloud.geometry.GlobalBEVRandomFlip",
    "point_cloud.geometry.RandomRotateTargetAngle",
)


def pipelines() -> list[tuple[str, str, list[str]]]:
    """Transform class paths of every pipeline an experiment config writes, in order."""
    found = []
    for path in sorted(_EXPERIMENTS.rglob("*.yaml")):
        datasets = (yaml.safe_load(path.read_text()) or {}).get("datasets") or {}
        for split in ("train", "test"):
            transforms = (datasets.get(split) or {}).get("transforms")
            if transforms:
                steps = [step["_target_"] for step in transforms["pipeline"]]
                found.append((str(path.relative_to(_EXPERIMENTS).with_suffix("")), split, steps))
    return found


_PIPELINES = pipelines()


def test_every_concrete_lidar_model_writes_its_pipelines() -> None:
    assert len(_PIPELINES) >= 30


@pytest.mark.parametrize(
    "config, split, steps", _PIPELINES, ids=[f"{c}:{s}" for c, s, _ in _PIPELINES]
)
def test_cameras_load_before_the_lidar_augmentations_move_their_calibration(
    config: str, split: str, steps: list[str]
) -> None:
    camera = [i for i, t in enumerate(steps) if t.endswith("camera.loading.LoadImagesFromFile")]
    lidar = [i for i, t in enumerate(steps) if t.endswith(_LIDAR_AUGMENTATIONS)]
    if camera and lidar:
        assert camera[0] < lidar[0]


_DETECTION_TEST = [
    (c, s, steps)
    for c, s, steps in _PIPELINES
    if s == "test" and c.startswith(("detection3d/", "multi/"))
]


@pytest.mark.parametrize(
    "config, split, steps", _DETECTION_TEST, ids=[c for c, _, _ in _DETECTION_TEST]
)
def test_detection_evaluation_drops_the_boxes_of_ignored_classes(
    config: str, split: str, steps: list[str]
) -> None:
    assert any(t.endswith("boxes3d.filters.BBoxesLabelNameFilter") for t in steps)
