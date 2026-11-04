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


"""Tests for the class donors of the motion anchors."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import polars as pl
import torch

from autoware_ml.anchors import trajectories
from autoware_ml.datamodule.common.motion import MotionTask
from autoware_ml.types.tasks import TaskType
from autoware_ml.anchors.trajectories import TrajectoryAnchorRecipe
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary
from autoware_ml.dataclasses.batch.motion import AgentTracksSample, MotionGTSample
from autoware_ml.datamodule.tests.test_detection3d import box
from autoware_ml.types.geometry import Box3DFieldIndex

_CLASS_NAMES = ["car", "traffic_cone", "debris"]
# Every class moves by its own constant offset, so the modes tell where they come from
_OFFSETS = {"car": 10.0, "traffic_cone": 0.0, "debris": -5.0}


def _recipe_taxonomy() -> LabelTaxonomy:
    """Taxonomy of the three classes."""
    return LabelTaxonomy(
        LabelVocabulary({name: name for name in _CLASS_NAMES}),
        _CLASS_NAMES,
        {name: name for name in _CLASS_NAMES},
        -1,
    )


def _recipe(tmp_path: Path, class_donors: dict[str, str]) -> TrajectoryAnchorRecipe:
    """Recipe of two modes over the three classes."""
    return TrajectoryAnchorRecipe(
        cache_path=str(tmp_path),
        taxonomy=_recipe_taxonomy(),
        num_modes=2,
        record_stride=1,
        max_agent_distance=55.0,
        iterations=5,
        seed=0,
        class_donors=class_donors,
        filter_attributes=[],
    )


def _build(recipe: TrajectoryAnchorRecipe, counts: dict[str, int]) -> np.ndarray:
    """Build the modes of a split where some classes have a few complete futures, the others 20."""
    rng = np.random.default_rng(0)
    futures = [
        [
            np.full((12, 2), _OFFSETS[name]) + rng.normal(0, 0.01, (12, 2))
            for _ in range(counts.get(name, 20))
        ]
        for name in _CLASS_NAMES
    ]
    motion_task = MotionTask.__new__(MotionTask)
    training = SimpleNamespace(
        source=SimpleNamespace(database=SimpleNamespace(version="db")),
        dataset=SimpleNamespace(dataset_tasks={TaskType.MOTION: motion_task}),
    )
    split = SimpleNamespace(datasets=lambda task_types: [training])
    with mock.patch.object(trajectories, "collect_agent_futures", lambda *args: futures):
        return recipe.build(split)["anchors"]


def _motion_task(
    boxes: list[list[dict]],
    key_frames: list[bool],
    read: list[int],
    positions: list[torch.Tensor] | None = None,
) -> SimpleNamespace:
    """Motion task over records with the given boxes, every agent with a complete future.

    The agents of a record stay at the origin unless ``positions`` gives their 13 steps.
    """
    column = DatasetTableSchema.BOXES_3D.name
    records = pl.DataFrame(
        {
            column: boxes,
            DatasetTableSchema.LIDAR_FRAMES.name: [[{"lidar_keyframe": key}] for key in key_frames],
        },
        schema_overrides={column: DatasetTableSchema.to_polars_schema()[column]},
    )

    def sample(index: int) -> SimpleNamespace:
        read.append(index)
        count = len(boxes[index])
        agents = AgentTracksSample(
            positions=torch.zeros((count, 13, 3)) if positions is None else positions[index],
            headings=torch.zeros((count, 13)),
            velocities=torch.zeros((count, 13, 2)),
            valid=torch.ones((count, 13), dtype=torch.bool),
            agent_types=torch.zeros(count, dtype=torch.int64),
            agent_roles=torch.zeros(count, dtype=torch.int64),
            track_ids=[str(row) for row in range(count)],
            current_step_index=0,
            step_seconds=0.5,
            ignore_index=-1,
        )
        return SimpleNamespace(motion_gt_sample=MotionGTSample(agents=agents))

    return SimpleNamespace(dataset_records_dataframe=records, get_data_sample=sample)


class TestTrajectoryAnchors(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_a_class_without_futures_takes_the_modes_of_its_donor(self) -> None:
        """
        Input: debris without complete futures, donated by the traffic cone.
        Expected: the debris modes are the traffic cone modes.
        Check: the anchors of both classes.
        """
        anchors = _build(_recipe(self.tmp_path, {"debris": "traffic_cone"}), {"debris": 0})

        np.testing.assert_array_equal(anchors[2], anchors[1])
        self.assertLess(abs(float(anchors[1].mean())), 0.1)

    def test_a_class_with_fewer_futures_than_modes_takes_its_donor(self) -> None:
        """
        Input: debris with one complete future for two modes, donated by the traffic cone.
        Expected: the debris modes are the traffic cone modes, one future clusters no two modes.
        Check: the anchors of both classes.
        """
        anchors = _build(_recipe(self.tmp_path, {"debris": "traffic_cone"}), {"debris": 1})

        np.testing.assert_array_equal(anchors[2], anchors[1])

    def test_a_class_with_futures_for_its_modes_takes_no_donor(self) -> None:
        """
        Input: debris with two complete futures for two modes, donated by the traffic cone.
        Expected: its own futures would be thrown away.
        Check: ValueError naming the class and the count of its futures.
        """
        with self.assertRaisesRegex(
            ValueError, "debris has 2 complete futures, enough for its 2 modes"
        ):
            _build(_recipe(self.tmp_path, {"debris": "traffic_cone"}), {"debris": 2})

    def test_chained_donors_are_refused(self) -> None:
        """
        Input: debris donated by the traffic cone, which is donated by the car.
        Expected: the debris modes would depend on the order of the donor pairs.
        Check: ValueError naming the traffic cone.
        """
        with self.assertRaisesRegex(ValueError, "traffic_cone"):
            _recipe(self.tmp_path, {"debris": "traffic_cone", "traffic_cone": "car"})

    def test_the_stride_counts_key_frame_records(self) -> None:
        """
        Input: six records whose odd ones lie between key frames, read with a stride of 2.
        Expected: every second key frame is read and no frame between key frames.
        Check: the read records.
        """
        read: list[int] = []
        motion_task = _motion_task(
            [[] for _ in range(6)], [index % 2 == 0 for index in range(6)], read
        )

        trajectories.collect_agent_futures(motion_task, _recipe_taxonomy(), 2, 55.0, frozenset())

        self.assertEqual(sorted(set(read)), [0, 4])

    def test_the_boxes_the_pipeline_drops_shape_no_mode(self) -> None:
        """
        Input: a key frame with a moving and a parked car, both with a complete future, and the
          exclusion rule of parked cars of the training pipeline.
        Expected: only the moving car gives a future.
        """
        cars = [
            box("moving", "car", fields={Box3DFieldIndex.X: 10.0}),
            box(
                "parked",
                "car",
                attributes=["vehicle_state.parked"],
                fields={Box3DFieldIndex.X: 20.0},
            ),
        ]
        motion_task = _motion_task([cars], [True], [])

        agents = trajectories.collect_agent_futures(
            motion_task, _recipe_taxonomy(), 1, 55.0, frozenset({("car", "vehicle_state.parked")})
        )

        self.assertEqual([len(class_agents) for class_agents in agents], [1, 0, 0])

    def test_a_donor_with_too_few_futures_is_named(self) -> None:
        """
        Input: debris without futures, donated by the traffic cone with one future for two modes.
        Expected: ValueError naming the donor, not the class it serves.
        """
        with self.assertRaisesRegex(
            ValueError, "Donor traffic_cone of class debris has 1 complete"
        ):
            _build(
                _recipe(self.tmp_path, {"debris": "traffic_cone"}),
                {"debris": 0, "traffic_cone": 1},
            )

    def test_a_class_with_fewer_futures_than_modes_and_no_donor_is_refused(self) -> None:
        """
        Input: debris with one complete future for two modes and no donor.
        Expected: ValueError naming the class, its modes cannot be clustered.
        """
        with self.assertRaisesRegex(ValueError, "debris has 1 samples, 2 modes need more"):
            _build(_recipe(self.tmp_path, {}), {"debris": 1})

    def test_a_donor_outside_the_taxonomy_is_refused(self) -> None:
        """
        Input: debris donated by a class the taxonomy does not hold.
        Expected: ValueError naming the unknown class.
        """
        with self.assertRaisesRegex(ValueError, r"outside the taxonomy: \['bus'\]"):
            _recipe(self.tmp_path, {"debris": "bus"})

    def test_a_future_is_read_in_the_frame_of_its_agent(self) -> None:
        """
        Input: a car at (10, 0) heading along y, its future steps at (10, 1), (10, 2) and on.
        Expected: the future runs along the heading of the agent, (1, 0), (2, 0) and on.
        """
        car = box("car", "car", fields={Box3DFieldIndex.X: 10.0, Box3DFieldIndex.YAW: math.pi / 2})
        positions = torch.zeros((1, 13, 3))
        positions[0, :, 0] = 10.0
        positions[0, :, 1] = torch.arange(13.0)
        motion_task = _motion_task([[car]], [True], [], positions=[positions])

        agents = trajectories.collect_agent_futures(
            motion_task, _recipe_taxonomy(), 1, 55.0, frozenset()
        )

        expected = np.stack([np.arange(1.0, 13.0), np.zeros(12)], axis=-1)
        np.testing.assert_allclose(agents[0][0], expected, atol=1e-5)
