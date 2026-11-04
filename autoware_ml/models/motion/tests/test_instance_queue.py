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

"""Tests for the history the motion head reads per query."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.motion.instance_queue import (
    QueueState,
    advance_queue,
    push_queue,
    queue_history,
)

QUERIES, DIMS = 4, 3
IDENTITY = torch.eye(4)[None]


def frame_feature(value: float) -> torch.Tensor:
    """Features of every query of a frame, the query index plus a frame value."""
    return (torch.arange(QUERIES, dtype=torch.float32)[:, None] + value).expand(-1, DIMS)[None]


def step(
    state: QueueState,
    value: float,
    carried: torch.Tensor,
    stride: int,
    continues: bool = True,
    prev2current: torch.Tensor = IDENTITY,
    anchor: torch.Tensor | None = None,
) -> QueueState:
    """Advance one frame of one lane and push its features."""
    state = advance_queue(state, torch.tensor([continues]), carried, prev2current, stride)
    anchor = torch.zeros(1, QUERIES, 10) if anchor is None else anchor
    return push_queue(state, frame_feature(value), anchor, stride)


class TestInstanceQueue(unittest.TestCase):
    def test_a_query_reads_the_entry_of_the_slot_it_was_carried_from(self) -> None:
        """
        Input: frame 0 pushed, frame 1 carries slot 0 from query 2 and slot 1 from query 0.
        Expected: query 0 reads query 2 of frame 0, query 1 reads query 0, fresh queries read
          nothing.
        """
        state = step(
            QueueState.empty(1, 2, QUERIES, DIMS, 1, torch.device("cpu")),
            100.0,
            torch.zeros(1, 0, dtype=torch.long),
            1,
        )
        state = advance_queue(state, torch.tensor([True]), torch.tensor([[2, 0]]), IDENTITY, 1)
        feature, _, valid = queue_history(state)
        self.assertEqual(feature[0, 0, 0, 0].item(), 102.0)
        self.assertEqual(feature[0, 1, 0, 0].item(), 100.0)
        self.assertEqual(valid[0, :, 0].tolist(), [True, True, False, False])

    def test_entries_are_pushed_every_stride_frames_and_cleared_on_restart(self) -> None:
        """
        Input: stride 2, three frames carrying every query to itself, then a lane restart.
        Expected: frames 0 and 2 are kept, newest first, the restart clears the history.
        """
        identity = torch.arange(2)[None]
        state = QueueState.empty(1, 3, QUERIES, DIMS, 2, torch.device("cpu"))
        for value in (0.0, 10.0, 20.0):
            state = step(state, value, identity, 2)
        state = advance_queue(state, torch.tensor([True]), identity, IDENTITY, 2)
        feature, _, valid = queue_history(state)
        self.assertEqual(feature[0, 0, :2, 0].tolist(), [20.0, 0.0])
        self.assertEqual(valid[0, 0].tolist(), [True, True, False])
        state = advance_queue(state, torch.tensor([False]), identity, IDENTITY, 2)
        self.assertFalse(queue_history(state)[2].any())

    def test_entry_anchors_follow_the_ego_motion(self) -> None:
        """
        Input: an entry pushed with a query at x=10, then a frame 2 m further along x.
        Expected: the entry anchor reads x=8 in the new frame.
        """
        anchor = torch.zeros(1, QUERIES, 10)
        anchor[0, 0, 0], anchor[0, :, 7] = 10.0, 1.0
        state = step(
            QueueState.empty(1, 1, QUERIES, DIMS, 1, torch.device("cpu")),
            0.0,
            torch.zeros(1, 0, dtype=torch.long),
            1,
            anchor=anchor,
        )
        forward = IDENTITY.clone()
        forward[0, 0, 3] = -2.0
        state = advance_queue(state, torch.tensor([True]), torch.arange(2)[None], forward, 1)
        self.assertEqual(queue_history(state)[1][0, 0, 0, 0].item(), 8.0)
