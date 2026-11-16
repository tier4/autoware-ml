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

"""Tests for the motion samples and their batch."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.motion import (
    PADDED_INDEX,
    AgentTracksBatch,
    AgentTracksSample,
    MapPolylinesBatch,
    MapPolylinesSample,
    MotionGTBatch,
    MotionGTSample,
)


def _tracks(num_agents: int = 2, num_steps: int = 5, **kwargs) -> AgentTracksSample:
    """Build a zero filled agent tracks sample with the given fields overridden."""
    fields = dict(
        positions=torch.zeros(num_agents, num_steps, 3),
        headings=torch.zeros(num_agents, num_steps),
        velocities=torch.zeros(num_agents, num_steps, 2),
        valid=torch.ones(num_agents, num_steps, dtype=torch.bool),
        agent_types=torch.zeros(num_agents, dtype=torch.int64),
        agent_roles=torch.zeros(num_agents, dtype=torch.int64),
        track_ids=[str(i) for i in range(num_agents)],
        current_step_index=2,
        step_seconds=0.1,
        ignore_index=-1,
    )
    return AgentTracksSample(**(fields | kwargs))


def _polylines(num_polylines: int = 3, num_points: int = 4) -> MapPolylinesSample:
    """Build a map polylines sample of consecutive point coordinates."""
    return MapPolylinesSample(
        points=torch.arange(num_polylines * num_points * 3, dtype=torch.float32).reshape(
            num_polylines, num_points, 3
        ),
        valid=torch.ones(num_polylines, num_points, dtype=torch.bool),
        polyline_types=torch.arange(num_polylines, dtype=torch.int64),
        is_intersection=torch.ones(num_polylines, dtype=torch.bool),
        element_ids=[str(i) for i in range(num_polylines)],
    )


def _sample(num_agents: int = 2, num_polylines: int | None = 3, **kwargs) -> MotionGTSample:
    """Build a GT sample of the tracks and, unless None, the polylines."""
    return MotionGTSample(
        agents=_tracks(num_agents=num_agents, **kwargs),
        map_polylines=None if num_polylines is None else _polylines(num_polylines),
    )


class TestAgentTracksSample(unittest.TestCase):
    """The agent tracks sample counts its agents and steps and carries optional sizes."""

    def test_agent_tracks_sizes_default_to_none(self) -> None:
        """
        Input: a sample of two agents over five steps built without sizes.
        Expected: the counts follow the tensor shapes and the sizes default to None, since
        not every dataset records a box size.
        Check: num_agents, num_steps and sizes.
        """
        tracks = _tracks()

        self.assertEqual(tracks.num_agents, 2)
        self.assertEqual(tracks.num_steps, 5)
        self.assertIsNone(tracks.sizes)

    def test_agent_tracks_carry_one_size_per_agent(self) -> None:
        """
        Input: the sample built with a float32 size per agent.
        Expected: the sizes are stored as given, one row per agent, and can be dropped again
        through _replace.
        Check: the presence, shape and dtype of sizes and the sizes of the replaced sample.
        """
        sizes = torch.tensor([[4.5, 1.8, 1.5], [0.6, 0.6, 1.7]])

        tracks = _tracks(sizes=sizes)

        self.assertIsNotNone(tracks.sizes)
        self.assertEqual(tracks.sizes.shape, (tracks.num_agents, 3))
        self.assertEqual(tracks.sizes.dtype, torch.float32)
        self.assertIsNone(tracks._replace(sizes=None).sizes)


class TestAgentTracksBatch(unittest.TestCase):
    """The agent tracks batch collates the tracks of the scenarios on its own."""

    def test_collate_pads_the_agents_of_the_tracks(self) -> None:
        """
        Input: agent tracks of one and of two agents, collated without a GT sample around them.
        Expected: the tracks are padded to two agents per scenario, and the mask and the batch
        size follow the scenarios.
        Check: batch_size, the position shape, the mask and the padded role.
        """
        batch = AgentTracksBatch.collate([_tracks(num_agents=1), _tracks(num_agents=2)])

        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.positions.shape, (2, 2, 5, 3))
        self.assertEqual(batch.mask.tolist(), [[True, False], [True, True]])
        self.assertEqual(batch.agent_roles[0, 1].item(), PADDED_INDEX)


class TestMapPolylinesBatch(unittest.TestCase):
    """The map polylines batch collates the vector maps of the scenarios on its own."""

    def test_collate_pads_the_polylines_of_the_maps(self) -> None:
        """
        Input: maps of two and of one polyline, collated without a GT sample around them.
        Expected: the maps are padded to two polylines per scenario with the padded type.
        Check: the point shape, the mask and the padded polyline type.
        """
        batch = MapPolylinesBatch.collate(
            [_polylines(num_polylines=2), _polylines(num_polylines=1)]
        )

        self.assertEqual(batch.points.shape, (2, 2, 4, 3))
        self.assertEqual(batch.mask.tolist(), [[True, True], [True, False]])
        self.assertEqual(batch.polyline_types[1, 1].item(), PADDED_INDEX)


class TestMotionGTBatch(unittest.TestCase):
    """The GT batch pads the samples to the widest one and masks the padding."""

    def test_collate_pads_the_agents_and_masks_the_padding(self) -> None:
        """
        Input: a sample of one agent at position 1.0 and a sample of three agents.
        Expected: the agents are padded to three per sample, the padded slots carry zeros,
        invalid steps, the ignore index and the padded role, and the window settings are
        shared by the batch.
        Check: the batch size, the tensor shapes, the agent mask, the values of the real and
        the padded slots, the track ids, the window settings and the absent sizes.
        """
        first = _sample(num_agents=1)
        first = first._replace(agents=first.agents._replace(positions=first.agents.positions + 1.0))

        batch = MotionGTBatch.collate_gt_samples([first, _sample(num_agents=3)])

        self.assertIsNotNone(batch)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.agents.positions.shape, (2, 3, 5, 3))
        self.assertEqual(batch.agents.headings.shape, (2, 3, 5))
        self.assertEqual(batch.agents.velocities.shape, (2, 3, 5, 2))
        self.assertEqual(batch.agents.valid.shape, (2, 3, 5))
        self.assertEqual(batch.agents.mask.tolist(), [[True, False, False], [True, True, True]])
        self.assertTrue(batch.agents.positions[0, 0].eq(1.0).all().item())
        self.assertTrue(batch.agents.positions[0, 1:].eq(0.0).all().item())
        self.assertFalse(batch.agents.valid[0, 1:].any().item())
        self.assertTrue(batch.agents.valid[0, 0].all().item())
        self.assertEqual(batch.agents.agent_types[0].tolist(), [0, -1, -1])
        self.assertEqual(batch.agents.agent_roles[0].tolist(), [0, PADDED_INDEX, PADDED_INDEX])
        self.assertEqual(batch.agents.track_ids, [["0"], ["0", "1", "2"]])
        self.assertEqual(batch.agents.current_step_index, 2)
        self.assertEqual(batch.agents.step_seconds, 0.1)
        self.assertEqual(batch.agents.ignore_index, -1)
        self.assertIsNone(batch.agents.sizes)

    def test_collate_pads_the_polylines_and_masks_the_padding(self) -> None:
        """
        Input: a sample with one polyline and a sample with three.
        Expected: the polylines are padded to three per sample, the padded slots carry zeros,
        invalid points, the padded type and no intersection flag.
        Check: has_map, the tensor shapes, the polyline mask, the values of the padded slots,
        the types and intersection flags per sample and the element ids.
        """
        batch = MotionGTBatch.collate_gt_samples(
            [_sample(num_polylines=1), _sample(num_polylines=3)]
        )

        self.assertIsNotNone(batch)
        self.assertTrue(batch.has_map)
        self.assertEqual(batch.map_polylines.points.shape, (2, 3, 4, 3))
        self.assertEqual(batch.map_polylines.valid.shape, (2, 3, 4))
        self.assertEqual(
            batch.map_polylines.mask.tolist(), [[True, False, False], [True, True, True]]
        )
        self.assertFalse(batch.map_polylines.valid[0, 1:].any().item())
        self.assertTrue(batch.map_polylines.points[0, 1:].eq(0.0).all().item())
        self.assertEqual(
            batch.map_polylines.polyline_types[0].tolist(), [0, PADDED_INDEX, PADDED_INDEX]
        )
        self.assertEqual(batch.map_polylines.polyline_types[1].tolist(), [0, 1, 2])
        self.assertEqual(batch.map_polylines.is_intersection[0].tolist(), [True, False, False])
        self.assertEqual(batch.map_polylines.element_ids, [["0"], ["0", "1", "2"]])

    def test_collate_without_maps_leaves_them_none(self) -> None:
        """
        Input: two samples without a vector map.
        Expected: the batch has no map, so the whole map batch is None.
        Check: has_map and map_polylines.
        """
        batch = MotionGTBatch.collate_gt_samples([_sample(num_polylines=None)] * 2)

        self.assertIsNotNone(batch)
        self.assertFalse(batch.has_map)
        self.assertIsNone(batch.map_polylines)

    def test_collate_stacks_the_sizes_when_every_sample_carries_them(self) -> None:
        """
        Input: two samples with sizes, of two and three agents, and then a sample with sizes
        next to one without.
        Expected: the sizes are stacked and padded with zeros like the other agent tensors,
        and a batch where only some samples carry sizes is refused.
        Check: the shape of agent_sizes, the padded row, and ValueError naming the sizes.
        """
        sizes = torch.ones(2, 3)
        with_sizes = _sample(sizes=sizes)

        batch = MotionGTBatch.collate_gt_samples(
            [with_sizes, _sample(num_agents=3, sizes=torch.ones(3, 3))]
        )

        self.assertIsNotNone(batch)
        self.assertEqual(batch.agents.sizes.shape, (2, 3, 3))
        self.assertEqual(batch.agents.sizes[0, 2].tolist(), [0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "agent sizes"):
            MotionGTBatch.collate_gt_samples([with_sizes, _sample()])

    def test_collate_rejects_mismatched_windows_and_partial_maps(self) -> None:
        """
        Input: pairs of samples that differ in step count, current step, step period, ignore
        index, map presence or points per polyline.
        Expected: a batch shares one window shape, one taxonomy and one map layout, so each
        pair is refused.
        Check: ValueError naming the shared setting for every pair.
        """
        with self.assertRaisesRegex(ValueError, "shares the window shape"):
            MotionGTBatch.collate_gt_samples([_sample(), _sample(num_steps=4)])
        with self.assertRaisesRegex(ValueError, "shares the window shape"):
            MotionGTBatch.collate_gt_samples([_sample(), _sample(current_step_index=1)])
        with self.assertRaisesRegex(ValueError, "shares the window shape"):
            MotionGTBatch.collate_gt_samples([_sample(), _sample(step_seconds=0.5)])
        with self.assertRaisesRegex(ValueError, "shares the taxonomy"):
            MotionGTBatch.collate_gt_samples([_sample(), _sample(ignore_index=255)])
        with self.assertRaisesRegex(ValueError, "vector map or none"):
            MotionGTBatch.collate_gt_samples([_sample(), _sample(num_polylines=None)])
        mismatched = _sample()._replace(map_polylines=_polylines(num_points=7))
        with self.assertRaisesRegex(ValueError, "points per polyline"):
            MotionGTBatch.collate_gt_samples([_sample(), mismatched])

    def test_collate_of_nothing_is_none(self) -> None:
        """
        Input: an empty list of samples.
        Expected: there is no batch to build, so the result is None.
        Check: the return value.
        """
        self.assertIsNone(MotionGTBatch.collate_gt_samples([]))

    def test_to_device_moves_every_tensor_and_keeps_none(self) -> None:
        """
        Input: a batch with sizes and a map, and a batch without a map, moved to the CPU.
        Expected: every tensor field lands on the device, the python fields are kept, and a
        None map stays None.
        Check: the device of the agent, size and map tensors, the track ids and the map of
        the batch without a map.
        """
        batch = MotionGTBatch.collate_gt_samples([_sample(sizes=torch.ones(2, 3))])
        self.assertIsNotNone(batch)

        moved = batch.to_device(torch.device("cpu"))

        self.assertEqual(moved.agents.positions.device.type, "cpu")
        self.assertEqual(moved.agents.sizes.device.type, "cpu")
        self.assertEqual(moved.map_polylines.points.device.type, "cpu")
        self.assertEqual(moved.map_polylines.mask.device.type, "cpu")
        self.assertEqual(moved.agents.track_ids, batch.agents.track_ids)
        without_map = MotionGTBatch.collate_gt_samples([_sample(num_polylines=None)])
        self.assertIsNone(without_map.to_device(torch.device("cpu")).map_polylines)
