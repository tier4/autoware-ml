"""Tracks, observed targets and tracklet-aware assignment of the Sparse4D head."""

from __future__ import annotations

import torch

from autoware_ml.models.detection3d.heads.sparse4d import (
    ANCHOR_DIM,
    InstanceTracker,
    encode_ground_truth,
)
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)


def _continue(batch_size: int) -> torch.Tensor:
    return torch.ones(batch_size, dtype=torch.bool)


def test_an_instance_stays_a_target_once_observed() -> None:
    tracker = InstanceTracker(num_temp_instances=2, min_observed_points=2, track_birth_score=0.5)
    ids = [torch.tensor([7, 8])]
    point_counts = [[0, 5], [0, 5], [3, 5], [0, 0], [0, 1]]
    observed = []
    for frame_index, counts in enumerate(point_counts):
        tracker.begin_frame(
            _continue(1), None if frame_index == 0 else torch.zeros((1, 2), dtype=torch.long)
        )
        observed.append(tracker.observe(ids, [torch.tensor(counts)])[0].tolist())

    # Instance 7 is a target from its first frame with 2 points on, also without points later
    assert [mask[0] for mask in observed] == [False, False, True, True, True]
    assert [mask[1] for mask in observed] == [True] * 5


def test_a_lane_starting_a_stream_forgets_its_instances_and_tracks() -> None:
    tracker = InstanceTracker(num_temp_instances=2, min_observed_points=1, track_birth_score=0.5)
    tracker.begin_frame(_continue(2), None)
    tracker.observe([torch.tensor([1]), torch.tensor([2])], [torch.tensor([4]), torch.tensor([4])])
    tracker.assign_track_ids(torch.tensor([[0.9, 0.1, 0.9], [0.9, 0.1, 0.1]]))
    first_streams = tracker.stream_ids.clone()

    tracker.begin_frame(torch.tensor([True, False]), torch.tensor([[0, 2], [0, 1]]))
    masks = tracker.observe(
        [torch.tensor([1]), torch.tensor([2])], [torch.tensor([0]), torch.tensor([0])]
    )

    assert [mask.tolist() for mask in masks] == [[True], [False]]
    assert tracker._slot_track_id.tolist() == [[0, 1], [-1, -1]]
    assert tracker.stream_ids[0] == first_streams[0]
    assert tracker.stream_ids[1] != first_streams[1]


def test_slots_carry_the_owner_and_the_track_of_their_query() -> None:
    tracker = InstanceTracker(num_temp_instances=2, min_observed_points=1, track_birth_score=0.5)
    tracker.begin_frame(_continue(1), None)
    first = tracker.assign_track_ids(torch.tensor([[0.2, 0.9, 0.1, 0.8]]))
    assert first.tolist() == [[-1, 0, -1, 1]]
    tracker.set_query_owners(
        [0], torch.tensor([[-1, 30, -1, 31]]), torch.tensor([[0.0, 0.9, 0.0, 0.8]])
    )

    # The bank cached query 3 into slot 0 and query 1 into slot 1
    tracker.begin_frame(_continue(1), torch.tensor([[3, 1]]))
    assert tracker.slot_owners([0]).tolist() == [[31, 30]]
    second = tracker.assign_track_ids(torch.tensor([[0.1, 0.1, 0.7, 0.2]]))
    # Slots keep their tracks whatever their score, a new confident query starts track 2
    assert second.tolist() == [[1, 0, 2, -1]]


def test_assignment_keeps_the_followed_instance_and_frees_the_rest(tmp_path) -> None:
    head = build_sparse4d_head(make_anchor_file(tmp_path))
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

    assert targets.assigned_instance[0].tolist() == [10, -1, -1, -1, -1, 11, -1, -1]
    assert not targets.matched[0, 1]
    assert targets.num_pos == 2


def test_assignment_without_ground_truth_sends_followed_slots_to_background(tmp_path) -> None:
    head = build_sparse4d_head(make_anchor_file(tmp_path))
    ground_truth = encode_ground_truth(
        torch.zeros((0, 9)), torch.zeros((0,), dtype=torch.long), torch.device("cpu"), torch.float32
    )

    targets = head._assign_targets(
        torch.zeros((1, 8, 2)),
        torch.zeros((1, 8, ANCHOR_DIM)),
        [ground_truth],
        [torch.zeros((0,), dtype=torch.long)],
        torch.tensor([[10, -1]]),
    )

    assert targets.num_pos == 0
    assert (targets.assigned_instance == -1).all()


def test_an_unconfident_match_takes_no_instance() -> None:
    tracker = InstanceTracker(num_temp_instances=2, min_observed_points=1, track_birth_score=0.5)
    tracker.begin_frame(_continue(1), None)
    tracker.assign_track_ids(torch.zeros((1, 3)))
    tracker.set_query_owners([0], torch.tensor([[30, 31, -1]]), torch.tensor([[0.9, 0.1, 0.0]]))

    tracker.begin_frame(_continue(1), torch.tensor([[0, 1]]))

    # Query 1 matched instance 31 below the birth score, so its slot stays free
    assert tracker.slot_owners([0]).tolist() == [[30, -1]]
