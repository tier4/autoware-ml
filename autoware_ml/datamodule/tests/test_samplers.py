"""Tests for the scene-streaming sampler."""

from __future__ import annotations

import logging

import pytest

from autoware_ml.datamodule.base import DataLoaderConfig
from autoware_ml.datamodule.common.multiview_detection3d import build_streaming_dataloader
from autoware_ml.datamodule.samplers import GroupStreamingSampler


class _FakeScenesDataset:
    """Minimal dataset stub exposing scene_index_groups()."""

    def __init__(self, scene_lengths: list[int]) -> None:
        self._groups = []
        start = 0
        for length in scene_lengths:
            self._groups.append(list(range(start, start + length)))
            start += length
        self.scene_starts = {group[0] for group in self._groups}
        self.total = start

    def scene_index_groups(self) -> list[list[int]]:
        return [list(group) for group in self._groups]

    def __len__(self) -> int:
        return sum(len(group) for group in self._groups)

    def __getitem__(self, index: int) -> int:
        return index


def test_lanes_stay_scene_contiguous_across_batches() -> None:
    # Scenes 0-9, 10-14, 15-19; two lanes. Each scene goes to the currently
    # shortest lane: lane0 gets scene 0, lane1 gets scenes 1 and 2, so both
    # lanes hold 10 frames and batch position k always continues the same
    # scene across consecutive batches.
    sampler = GroupStreamingSampler(_FakeScenesDataset([10, 5, 5]), batch_size=2, shuffle=False)
    expected = [index for pair in zip(range(0, 10), range(10, 20)) for index in pair]
    assert list(sampler) == expected


def test_per_rank_length_is_a_whole_number_of_batches() -> None:
    sampler = GroupStreamingSampler(_FakeScenesDataset([7, 3, 5, 2]), batch_size=3, shuffle=False)
    indices = list(sampler)
    assert len(indices) == len(sampler)
    assert len(indices) % 3 == 0


def test_trimming_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    sampler = GroupStreamingSampler(_FakeScenesDataset([10, 5]), batch_size=2, shuffle=False)
    with caplog.at_level(logging.WARNING, logger="autoware_ml.datamodule.samplers"):
        list(sampler)
    assert any("trimmed" in record.message for record in caplog.records)


def test_ranks_agree_on_length_and_shard_disjoint_scenes() -> None:
    dataset = _FakeScenesDataset([6, 6, 6, 6, 6, 6])
    rank_indices = []
    for rank in range(2):
        sampler = GroupStreamingSampler(dataset, batch_size=2, shuffle=False)
        sampler.world_size = 2
        sampler.rank = rank
        sampler._cached_epoch = None
        rank_indices.append(list(sampler))
    assert len(rank_indices[0]) == len(rank_indices[1])
    assert not set(rank_indices[0]) & set(rank_indices[1])


def test_set_epoch_reshuffles_deterministically() -> None:
    dataset = _FakeScenesDataset([4] * 8)

    def epoch_indices(epoch: int) -> list[int]:
        sampler = GroupStreamingSampler(dataset, batch_size=2, shuffle=True, seed=0)
        sampler.set_epoch(epoch)
        return list(sampler)

    assert epoch_indices(0) == epoch_indices(0)
    assert epoch_indices(0) != epoch_indices(1)


def test_raises_when_lanes_cannot_be_filled() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        list(GroupStreamingSampler(_FakeScenesDataset([5]), batch_size=2, shuffle=False))


def test_build_streaming_dataloader_rejects_dataloader_shuffle() -> None:
    with pytest.raises(ValueError, match="own the sample order"):
        build_streaming_dataloader(
            _FakeScenesDataset([4, 4]),
            DataLoaderConfig(batch_size=2, shuffle=True),
            collate_fn=None,
            shuffle_scenes=False,
        )


def _lanes(sampler: GroupStreamingSampler) -> list[list[int]]:
    indices = list(iter(sampler))
    return [indices[lane :: sampler.batch_size] for lane in range(sampler.batch_size)]


class TestFixedLengthTrainingEpochs:
    """With ``shuffle=True`` every epoch has the same length (Lightning assumes it)."""

    def test_length_is_constant_across_epochs(self) -> None:
        dataset = _FakeScenesDataset([7, 5, 18, 3, 8, 9])
        sampler = GroupStreamingSampler(dataset, batch_size=2, shuffle=True)

        lengths = []
        for epoch in range(6):
            sampler.set_epoch(epoch)
            lengths.append(len(sampler))
            assert len(list(iter(sampler))) == lengths[-1]

        assert len(set(lengths)) == 1
        assert lengths[0] == sampler.rounds_per_epoch * 2
        assert lengths[0] >= dataset.total

    def test_every_frame_is_served_when_lanes_balance(self) -> None:
        # Balanced scenes: shortest-lane assignment keeps every lane at the
        # fixed length, so padding never displaces a frame.
        dataset = _FakeScenesDataset([10, 10, 10, 10])
        sampler = GroupStreamingSampler(dataset, batch_size=2, shuffle=True)

        for epoch in range(4):
            sampler.set_epoch(epoch)
            assert set(iter(sampler)) == set(range(dataset.total))

    def test_lanes_only_advance_within_a_scene_or_restart_at_a_scene_start(self) -> None:
        dataset = _FakeScenesDataset([7, 5, 18, 3, 8, 9])
        sampler = GroupStreamingSampler(dataset, batch_size=3, shuffle=True)

        for epoch in range(4):
            sampler.set_epoch(epoch)
            for lane in _lanes(sampler):
                for previous, current in zip(lane, lane[1:]):
                    # Either the next frame of the same scene, or the first frame
                    # of a scene (prev_exists == 0 there resets the temporal memory).
                    assert current == previous + 1 or current in dataset.scene_starts

    def test_raises_when_scenes_cannot_fill_every_lane(self) -> None:
        sampler = GroupStreamingSampler(_FakeScenesDataset([5, 5]), batch_size=3, shuffle=True)

        with pytest.raises(ValueError, match="cannot fill"):
            len(sampler)


class TestEvaluationTrimsInsteadOfPadding:
    def test_single_lane_serves_every_frame_exactly_once_in_file_order(self) -> None:
        dataset = _FakeScenesDataset([7, 5, 18])
        sampler = GroupStreamingSampler(dataset, batch_size=1, shuffle=False)

        assert list(iter(sampler)) == list(range(dataset.total))

    def test_wider_batches_trim_without_repeating_frames(self) -> None:
        dataset = _FakeScenesDataset([7, 5, 18])
        sampler = GroupStreamingSampler(dataset, batch_size=2, shuffle=False)

        indices = list(iter(sampler))
        assert len(indices) == len(set(indices))
        assert len(indices) < dataset.total
