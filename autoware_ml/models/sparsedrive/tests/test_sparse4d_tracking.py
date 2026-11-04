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

"""Tracks, observed targets and tracklet-aware assignment of the Sparse4D head."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.sparsedrive.sparse4d_head import (
    ANCHOR_DIM,
    InstanceTracker,
    encode_ground_truth,
)
from autoware_ml.models.sparsedrive.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_recipe,
)


def _continue(batch_size: int) -> torch.Tensor:
    return torch.ones(batch_size, dtype=torch.bool)


class TestSparse4DTracking(unittest.TestCase):
    def test_an_instance_stays_a_target_once_observed(self) -> None:
        tracker = InstanceTracker(
            num_temp_instances=2, min_observed_points=2, track_birth_score=0.5
        )
        ids = [torch.tensor([7, 8])]
        point_counts = [[0, 5], [0, 5], [3, 5], [0, 0], [0, 1]]
        observed = []
        for frame_index, counts in enumerate(point_counts):
            tracker.begin_frame(
                _continue(1), None if frame_index == 0 else torch.zeros((1, 2), dtype=torch.long), 3
            )
            observed.append(tracker.observe(ids, [torch.tensor(counts)])[0].tolist())
            tracker.assign_track_ids(torch.zeros((1, 3)))

        # Instance 7 is a target from its first frame with 2 points on, also without points later
        self.assertEqual([mask[0] for mask in observed], [False, False, True, True, True])
        self.assertEqual([mask[1] for mask in observed], [True] * 5)

    def test_a_lane_starting_a_stream_forgets_its_instances_and_tracks(self) -> None:
        tracker = InstanceTracker(
            num_temp_instances=2, min_observed_points=1, track_birth_score=0.5
        )
        tracker.begin_frame(_continue(2), None, 3)
        tracker.observe(
            [torch.tensor([1]), torch.tensor([2])], [torch.tensor([4]), torch.tensor([4])]
        )
        tracker.assign_track_ids(torch.tensor([[0.9, 0.1, 0.9], [0.9, 0.1, 0.1]]))

        tracker.begin_frame(torch.tensor([True, False]), torch.tensor([[0, 2], [0, 1]]), 3)
        masks = tracker.observe(
            [torch.tensor([1]), torch.tensor([2])], [torch.tensor([0]), torch.tensor([0])]
        )

        self.assertEqual([mask.tolist() for mask in masks], [[True], [False]])
        self.assertEqual(tracker._slot_track_id.tolist(), [[0, 1], [-1, -1]])

    def test_a_state_made_in_inference_mode_restarts_a_stream_outside_it(self) -> None:
        # trainer.validate or test builds the state in inference mode, a later fit restarts lanes
        tracker = InstanceTracker(
            num_temp_instances=2, min_observed_points=1, track_birth_score=0.5
        )
        with torch.inference_mode():
            tracker.begin_frame(_continue(2), None, 3)
            tracker.assign_track_ids(torch.zeros((2, 3)))

        tracker.begin_frame(torch.tensor([True, False]), torch.zeros((2, 2), dtype=torch.long), 3)

        self.assertEqual(tracker._slot_owner[1].tolist(), [-1, -1])

    def test_slots_carry_the_owner_and_the_track_of_their_query(self) -> None:
        tracker = InstanceTracker(
            num_temp_instances=2, min_observed_points=1, track_birth_score=0.5
        )
        tracker.begin_frame(_continue(1), None, 4)
        first = tracker.assign_track_ids(torch.tensor([[0.2, 0.9, 0.1, 0.8]]))
        self.assertEqual(first.tolist(), [[-1, 0, -1, 1]])
        tracker.set_query_owners(
            [0], torch.tensor([[-1, 30, -1, 31]]), torch.tensor([[0.0, 0.9, 0.0, 0.8]])
        )

        # The bank cached query 3 into slot 0 and query 1 into slot 1
        tracker.begin_frame(_continue(1), torch.tensor([[3, 1]]), 4)
        self.assertEqual(tracker.slot_owners([0]).tolist(), [[31, 30]])
        second = tracker.assign_track_ids(torch.tensor([[0.1, 0.1, 0.7, 0.2]]))
        # Slots keep their tracks whatever their score, a new confident query starts track 2
        self.assertEqual(second.tolist(), [[1, 0, 2, -1]])

    def test_assignment_keeps_the_followed_instance_and_frees_the_rest(self) -> None:
        head = build_sparse4d_head(make_anchor_recipe())
        boxes = torch.tensor(
            [
                [5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0],
                [-5.0, 3.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0],
            ]
        )
        ground_truth = encode_ground_truth(
            boxes, torch.tensor([1, 1]), torch.device("cpu"), torch.float32
        )
        pred_anchor = torch.randn(1, 8, ANCHOR_DIM) * 20.0
        # Query 0 (a slot) sits far off, queries 4 and 5 sit exactly on the two boxes
        pred_anchor[0, 4] = ground_truth.anchor[0]
        pred_anchor[0, 5] = ground_truth.anchor[1]
        cls_logits = torch.full((1, 8, 2), -6.0)
        cls_logits[0, 4:6, 1] = 6.0
        instance_ids = [torch.tensor([10, 11])]

        # Slot 0 follows instance 10, slot 1 follows instance 99 that left the scene
        targets = head._assign_targets(
            cls_logits, pred_anchor, [ground_truth], instance_ids, torch.tensor([[10, 99]])
        )

        self.assertEqual(targets.assigned_instance[0].tolist(), [10, -1, -1, -1, -1, 11, -1, -1])
        self.assertFalse(targets.matched[0, 1])
        self.assertEqual(targets.num_pos, 2)

    def test_assignment_without_ground_truth_sends_followed_slots_to_background(self) -> None:
        head = build_sparse4d_head(make_anchor_recipe())
        ground_truth = encode_ground_truth(
            torch.zeros((0, 9)),
            torch.zeros((0,), dtype=torch.long),
            torch.device("cpu"),
            torch.float32,
        )

        targets = head._assign_targets(
            torch.zeros((1, 8, 2)),
            torch.zeros((1, 8, ANCHOR_DIM)),
            [ground_truth],
            [torch.zeros((0,), dtype=torch.long)],
            torch.tensor([[10, -1]]),
        )

        self.assertEqual(targets.num_pos, 0)
        self.assertTrue((targets.assigned_instance == -1).all())

    def test_an_unconfident_match_takes_no_instance(self) -> None:
        tracker = InstanceTracker(
            num_temp_instances=2, min_observed_points=1, track_birth_score=0.5
        )
        tracker.begin_frame(_continue(1), None, 3)
        tracker.assign_track_ids(torch.zeros((1, 3)))
        tracker.set_query_owners([0], torch.tensor([[30, 31, -1]]), torch.tensor([[0.9, 0.1, 0.0]]))

        tracker.begin_frame(_continue(1), torch.tensor([[0, 1]]), 3)

        # Query 1 matched instance 31 below the birth score, so its slot stays free
        self.assertEqual(tracker.slot_owners([0]).tolist(), [[30, -1]])

    def test_slots_keep_their_owner_through_frames_without_assignment(self) -> None:
        """
        Input: two lanes whose slots follow instances, then a frame where lane 0 is assigned no box
          while lane 1, between its key frames, gets no assignment, then frames without any.
        Expected: lane 0 frees its slots, lane 1 keeps its owners in the slots the bank moves them
          to, until it starts a new stream.
        Check: the slot owners of every frame.
        """
        tracker = InstanceTracker(
            num_temp_instances=2, min_observed_points=0, track_birth_score=0.3
        )
        confident = torch.full((2, 5), 0.9)
        tracker.begin_frame(torch.tensor([False, False]), None, 5)
        tracker.assign_track_ids(confident)
        owners = torch.tensor([[-1, -1, -1, 30, 31], [-1, -1, -1, 40, 41]])
        tracker.set_query_owners([0, 1], owners, confident)
        # The bank caches queries 3 and 4 into the two slots of both lanes
        tracker.begin_frame(_continue(2), torch.tensor([[3, 4], [3, 4]]), 5)
        tracker.assign_track_ids(confident)
        self.assertEqual(tracker.slot_owners([0, 1]).tolist(), [[30, 31], [40, 41]])

        tracker.set_query_owners([0], torch.full((1, 5), -1), confident[:1])
        # Slot 1 moves to slot 0 and a fresh query takes slot 1
        tracker.begin_frame(_continue(2), torch.tensor([[1, 4], [1, 4]]), 5)
        tracker.assign_track_ids(confident)
        self.assertEqual(tracker.slot_owners([0, 1]).tolist(), [[-1, -1], [41, -1]])

        tracker.begin_frame(torch.tensor([False, True]), torch.tensor([[0, 1], [0, 1]]), 5)
        self.assertEqual(tracker.slot_owners([0, 1]).tolist(), [[-1, -1], [41, -1]])
        tracker.assign_track_ids(confident)
        tracker.begin_frame(torch.tensor([True, False]), torch.tensor([[0, 1], [0, 1]]), 5)
        self.assertEqual(tracker.slot_owners([0, 1]).tolist(), [[-1, -1], [-1, -1]])
