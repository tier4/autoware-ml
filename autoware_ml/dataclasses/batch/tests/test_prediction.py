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

from __future__ import annotations

import pytest
import torch

from autoware_ml.dataclasses.batch.prediction import (
    PADDED_INDEX,
    AgentTracksSample,
    MapPolylinesSample,
    PredictionGTBatch,
    PredictionGTSample,
)


def _tracks(num_agents: int = 2, num_steps: int = 5, **kwargs) -> AgentTracksSample:
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


def test_agent_tracks_sizes_default_to_none() -> None:
    tracks = _tracks()

    assert tracks.num_agents == 2
    assert tracks.num_steps == 5
    assert tracks.sizes is None


def test_agent_tracks_carry_one_size_per_agent() -> None:
    sizes = torch.tensor([[4.5, 1.8, 1.5], [0.6, 0.6, 1.7]])

    tracks = _tracks(sizes=sizes)

    assert tracks.sizes is not None
    assert tracks.sizes.shape == (tracks.num_agents, 3)
    assert tracks.sizes.dtype == torch.float32
    assert tracks._replace(sizes=None).sizes is None


def _polylines(num_polylines: int = 3, num_points: int = 4) -> MapPolylinesSample:
    return MapPolylinesSample(
        points=torch.arange(num_polylines * num_points * 3, dtype=torch.float32).reshape(
            num_polylines, num_points, 3
        ),
        valid=torch.ones(num_polylines, num_points, dtype=torch.bool),
        polyline_types=torch.arange(num_polylines, dtype=torch.int64),
        is_intersection=torch.ones(num_polylines, dtype=torch.bool),
        element_ids=[str(i) for i in range(num_polylines)],
    )


def _sample(num_agents: int = 2, num_polylines: int | None = 3, **kwargs) -> PredictionGTSample:
    return PredictionGTSample(
        agents=_tracks(num_agents=num_agents, **kwargs),
        map_polylines=None if num_polylines is None else _polylines(num_polylines),
    )


def test_collate_pads_the_agents_and_masks_the_padding() -> None:
    first = _sample(num_agents=1)
    first = first._replace(agents=first.agents._replace(positions=first.agents.positions + 1.0))

    batch = PredictionGTBatch.collate_gt_samples([first, _sample(num_agents=3)])

    assert batch is not None
    assert batch.batch_size == 2
    assert batch.agent_positions.shape == (2, 3, 5, 3)
    assert batch.agent_headings.shape == (2, 3, 5)
    assert batch.agent_velocities.shape == (2, 3, 5, 2)
    assert batch.agent_valid.shape == (2, 3, 5)
    assert batch.agent_mask.tolist() == [[True, False, False], [True, True, True]]
    # The padded slots carry zeros, invalid steps, the ignore index and the padded role
    assert batch.agent_positions[0, 0].eq(1.0).all()
    assert batch.agent_positions[0, 1:].eq(0.0).all()
    assert not batch.agent_valid[0, 1:].any()
    assert batch.agent_valid[0, 0].all()
    assert batch.agent_types[0].tolist() == [0, -1, -1]
    assert batch.agent_roles[0].tolist() == [0, PADDED_INDEX, PADDED_INDEX]
    assert batch.track_ids == [["0"], ["0", "1", "2"]]
    assert batch.current_step_index == 2
    assert batch.step_seconds == 0.1
    assert batch.ignore_index == -1
    assert batch.agent_sizes is None


def test_collate_pads_the_polylines_and_masks_the_padding() -> None:
    batch = PredictionGTBatch.collate_gt_samples(
        [_sample(num_polylines=1), _sample(num_polylines=3)]
    )

    assert batch is not None
    assert batch.has_map
    assert batch.map_points.shape == (2, 3, 4, 3)
    assert batch.map_valid.shape == (2, 3, 4)
    assert batch.polyline_mask.tolist() == [[True, False, False], [True, True, True]]
    assert not batch.map_valid[0, 1:].any()
    assert batch.map_points[0, 1:].eq(0.0).all()
    assert batch.map_polyline_types[0].tolist() == [0, PADDED_INDEX, PADDED_INDEX]
    assert batch.map_polyline_types[1].tolist() == [0, 1, 2]
    assert batch.map_is_intersection[0].tolist() == [True, False, False]
    assert batch.element_ids == [["0"], ["0", "1", "2"]]


def test_collate_without_maps_leaves_them_none() -> None:
    batch = PredictionGTBatch.collate_gt_samples([_sample(num_polylines=None)] * 2)

    assert batch is not None
    assert not batch.has_map
    assert batch.map_points is None
    assert batch.polyline_mask is None
    assert batch.element_ids is None


def test_collate_stacks_the_sizes_when_every_sample_carries_them() -> None:
    sizes = torch.ones(2, 3)
    with_sizes = _sample(sizes=sizes)

    batch = PredictionGTBatch.collate_gt_samples(
        [with_sizes, _sample(num_agents=3, sizes=torch.ones(3, 3))]
    )

    assert batch is not None
    assert batch.agent_sizes.shape == (2, 3, 3)
    assert batch.agent_sizes[0, 2].tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="agent sizes"):
        PredictionGTBatch.collate_gt_samples([with_sizes, _sample()])


def test_collate_rejects_mismatched_windows_and_partial_maps() -> None:
    with pytest.raises(ValueError, match="shares the window shape"):
        PredictionGTBatch.collate_gt_samples([_sample(), _sample(num_steps=4)])
    with pytest.raises(ValueError, match="shares the window shape"):
        PredictionGTBatch.collate_gt_samples([_sample(), _sample(current_step_index=1)])
    with pytest.raises(ValueError, match="shares the window shape"):
        PredictionGTBatch.collate_gt_samples([_sample(), _sample(step_seconds=0.5)])
    with pytest.raises(ValueError, match="shares the taxonomy"):
        PredictionGTBatch.collate_gt_samples([_sample(), _sample(ignore_index=255)])
    with pytest.raises(ValueError, match="vector map or none"):
        PredictionGTBatch.collate_gt_samples([_sample(), _sample(num_polylines=None)])
    mismatched = _sample()._replace(map_polylines=_polylines(num_points=7))
    with pytest.raises(ValueError, match="points per polyline"):
        PredictionGTBatch.collate_gt_samples([_sample(), mismatched])


def test_collate_of_nothing_is_none() -> None:
    assert PredictionGTBatch.collate_gt_samples([]) is None


def test_to_device_moves_every_tensor_and_keeps_none() -> None:
    batch = PredictionGTBatch.collate_gt_samples([_sample(sizes=torch.ones(2, 3))])
    assert batch is not None

    moved = batch.to_device(torch.device("cpu"))

    assert moved.agent_positions.device.type == "cpu"
    assert moved.agent_sizes.device.type == "cpu"
    assert moved.map_points.device.type == "cpu"
    assert moved.polyline_mask.device.type == "cpu"
    assert moved.track_ids == batch.track_ids
    without_map = PredictionGTBatch.collate_gt_samples([_sample(num_polylines=None)])
    assert without_map.to_device(torch.device("cpu")).map_points is None
