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

    def scene_index_groups(self) -> list[list[int]]:
        return [list(group) for group in self._groups]

    def __len__(self) -> int:
        return sum(len(group) for group in self._groups)

    def __getitem__(self, index: int) -> int:
        return index


def test_lanes_stay_scene_contiguous_across_batches() -> None:
    # Scenes 0-9, 10-14, 15-19; two lanes: lane0 gets scenes 0 and 2, lane1
    # scene 1. rounds = min(15, 5) = 5, so batch position k always continues
    # the same scene across consecutive batches.
    sampler = GroupStreamingSampler(
        _FakeScenesDataset([10, 5, 5]), batch_size=2, shuffle=False
    )
    assert list(sampler) == [0, 10, 1, 11, 2, 12, 3, 13, 4, 14]


def test_per_rank_length_is_a_whole_number_of_batches() -> None:
    sampler = GroupStreamingSampler(
        _FakeScenesDataset([7, 3, 5, 2]), batch_size=3, shuffle=False
    )
    indices = list(sampler)
    assert len(indices) == len(sampler)
    assert len(indices) % 3 == 0


def test_trimming_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    sampler = GroupStreamingSampler(
        _FakeScenesDataset([10, 5]), batch_size=2, shuffle=False
    )
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
