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
    "camera_lidar.geometry.GlobalRotScaleTrans",
    "camera_lidar.geometry.GlobalBEVRandomFlip",
    "camera_lidar.geometry.RandomRotateTargetAngle",
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


_GEOMETRIC_AUGMENTATIONS = _LIDAR_AUGMENTATIONS + ("camera.resize.ResizeCropFlipRotImage",)
_ANNOTATIONS_2D = "camera.annotations2d.LoadAnnotations2DFromBoxes3D"


@pytest.mark.parametrize(
    "config, split, steps", _PIPELINES, ids=[f"{c}:{s}" for c, s, _ in _PIPELINES]
)
def test_2d_annotations_are_projected_after_every_geometric_augmentation(
    config: str, split: str, steps: list[str]
) -> None:
    """
    Input: every pipeline an experiment config writes.
    Expected: the 2D projection reads the augmented boxes and calibration, so every geometric
    augmentation, of the images and of the global 3D frame, runs before it; a later one would
    leave the 2D targets silently wrong. Where the range filter goes is a recipe choice (see
    test_streampetr_configs.py).
    Check: in a pipeline that projects 2D boxes, every geometric augmentation comes before
    LoadAnnotations2DFromBoxes3D.
    """
    annotations = [i for i, t in enumerate(steps) if t.endswith(_ANNOTATIONS_2D)]
    geometric = [i for i, t in enumerate(steps) if t.endswith(_GEOMETRIC_AUGMENTATIONS)]
    if annotations:
        assert all(i < annotations[0] for i in geometric)


_DETECTION3D_TEST = [
    (c, s, steps)
    for c, s, steps in _PIPELINES
    if s == "test" and c.startswith(("detection3d/", "multi/"))
]


@pytest.mark.parametrize(
    "config, split, steps", _DETECTION3D_TEST, ids=[c for c, _, _ in _DETECTION3D_TEST]
)
def test_detection3d_evaluation_drops_the_boxes_of_ignored_classes(
    config: str, split: str, steps: list[str]
) -> None:
    assert any(t.endswith("boxes3d.filters.BBoxesLabelNameFilter") for t in steps)
