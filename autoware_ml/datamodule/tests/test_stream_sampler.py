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

"""Unit tests for the stream slots the streaming sampler hands the frames of a scene."""

from __future__ import annotations

from collections.abc import Sequence
import pickle
import unittest

from autoware_ml.datamodule.base_dataset import PaddingSlot, StreamPaddingSlot, StreamSlot
from autoware_ml.datamodule.samplers import GroupStreamingSampler


class _ScenesDataset:
    """Minimal dataset stub exposing scene_index_groups()."""

    def __init__(self, scene_lengths: Sequence[int]) -> None:
        self._groups = []
        start = 0
        for length in scene_lengths:
            self._groups.append(list(range(start, start + length)))
            start += length

    def scene_index_groups(self) -> Sequence[Sequence[int]]:
        return [list(group) for group in self._groups]

    def __len__(self) -> int:
        return sum(len(group) for group in self._groups)

    def __getitem__(self, index: int) -> int:
        return index


class TestStreamSamplerConsistentAugmentation(unittest.TestCase):
    """The stream slots a streaming sampler yields with ``consistent_augmentation``."""

    def build_stream_sampler(
        self,
        scene_lengths: Sequence[int],
        batch_size: int,
        seed: int = 0,
        shuffle: bool = False,
        consistent_augmentation: bool = True,
    ) -> GroupStreamingSampler:
        """Build a streaming sampler over scenes of the given lengths.

        Args:
            scene_lengths: Number of frames of every scene.
            batch_size: Number of lanes of the sampler.
            seed: Seed of the scene shuffle and of the stream seeds.
            shuffle: Whether the scenes are shuffled every epoch.
            consistent_augmentation: Whether the frames of a scene carry the seed of it.

        Returns:
            The sampler over a stub dataset holding the scenes.
        """
        return GroupStreamingSampler(
            _ScenesDataset(scene_lengths),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            consistent_augmentation=consistent_augmentation,
        )

    def lanes(self, sampler: GroupStreamingSampler) -> Sequence[Sequence[int]]:
        """Split the indices of one epoch into the lanes they feed, batch position by position."""
        indices = list(iter(sampler))
        return [indices[lane :: sampler.batch_size] for lane in range(sampler.batch_size)]

    def stream_seeds(self, sampler: GroupStreamingSampler, epoch: int) -> Sequence[tuple[int, ...]]:
        """The stream seed carried by every slot of one epoch."""
        sampler.set_epoch(epoch)
        return [slot.stream_seed for slot in sampler]

    def stream_seeds_per_scene(
        self, sampler: GroupStreamingSampler
    ) -> dict[int, set[tuple[int, ...]]]:
        """The stream seeds the slots of every scene carry in the current epoch."""
        scene_of = {
            index: scene for scene, group in enumerate(sampler.scene_groups) for index in group
        }
        seeds: dict[int, set[tuple[int, ...]]] = {}
        for slot in sampler:
            seeds.setdefault(scene_of[int(slot)], set()).add(slot.stream_seed)
        return seeds

    def test_frames_of_a_scene_carry_one_stream_seed_and_scenes_differ(self) -> None:
        """
        Input: scenes of 6, 3 and 3 frames through two lanes, with and without the consistent
        augmentation.
        Expected: the same frames in the same order either way, every frame a stream slot
        carrying the seed of its scene.
        Check: the indices match the plain sampler, every scene carries one seed and the
        three scenes carry three different ones.
        """
        sampler = self.build_stream_sampler([6, 3, 3], batch_size=2)
        plain = self.build_stream_sampler([6, 3, 3], batch_size=2, consistent_augmentation=False)

        indices = list(sampler)
        seeds = self.stream_seeds_per_scene(sampler)

        self.assertEqual([int(index) for index in indices], list(plain))
        self.assertTrue(all(isinstance(index, StreamSlot) for index in indices))
        self.assertFalse(any(isinstance(index, StreamSlot) for index in plain))
        self.assertEqual([len(seeds[scene]) for scene in range(3)], [1, 1, 1])
        self.assertEqual(len(seeds[0] | seeds[1] | seeds[2]), 3)

    def test_a_scene_gets_a_new_seed_every_epoch_and_the_same_in_every_process(self) -> None:
        """
        Input: two samplers built alike, as the ranks or a resumed run build them, and a
        third one with another seed.
        Expected: the stream seed depends only on the seed of the run, the epoch and the
        scene.
        Check: the two samplers carry the same seeds in epoch 0, and neither the next epoch
        nor the other run repeats any of them.
        """
        first = self.build_stream_sampler([4, 4], batch_size=2)
        second = self.build_stream_sampler([4, 4], batch_size=2)
        other_run = self.build_stream_sampler([4, 4], batch_size=2, seed=1)

        self.assertEqual(self.stream_seeds(first, 0), self.stream_seeds(second, 0))
        self.assertTrue(set(self.stream_seeds(first, 0)).isdisjoint(self.stream_seeds(first, 1)))
        self.assertTrue(
            set(self.stream_seeds(first, 0)).isdisjoint(self.stream_seeds(other_run, 0))
        )

    def test_a_scene_keeps_its_stream_seed_when_shuffled_and_cut_into_lanes(self) -> None:
        """
        Input: scenes of 5, 4, 3 and 2 frames shuffled through three lanes, so scenes are
        cut across lanes.
        Expected: a scene keeps one seed wherever its frames land.
        Check: every one of the four scenes carries exactly one seed.
        """
        sampler = self.build_stream_sampler([5, 4, 3, 2], batch_size=3, shuffle=True)

        seeds = self.stream_seeds_per_scene(sampler)

        self.assertEqual(sorted(seeds), [0, 1, 2, 3])
        self.assertTrue(all(len(scene_seeds) == 1 for scene_seeds in seeds.values()))

    def test_a_padding_slot_keeps_the_stream_seed_of_the_frame_it_repeats(self) -> None:
        """
        Input: scenes of 3 and 2 frames through two lanes, cut into lanes of two and three
        frames, so the shorter lane ends with a padding slot.
        Expected: the padding slot repeats the last frame of its lane under the same seed.
        Check: the slot is a StreamPaddingSlot and a PaddingSlot, with the index and the
        seed of the frame before it.
        """
        sampler = self.build_stream_sampler([3, 2], batch_size=2)

        short_lane = self.lanes(sampler)[0]
        padding = short_lane[-1]

        self.assertIsInstance(padding, StreamPaddingSlot)
        self.assertIsInstance(padding, PaddingSlot)
        self.assertEqual(padding, short_lane[-2])
        self.assertEqual(padding.stream_seed, short_lane[-2].stream_seed)

    def test_a_stream_slot_keeps_its_seed_through_a_worker_and_a_split(self) -> None:
        """
        Input: a stream slot and a stream padding slot, pickled as for a dataloader worker
        and moved to another index as a concatenated split does.
        Expected: both keep their seed and their kind.
        Check: the unpickled and the moved slots carry the seed, and the padding slot stays
        a StreamPaddingSlot.
        """
        stream_seed = (7, 2, 5)
        slot = StreamSlot(3, stream_seed)
        padding = StreamPaddingSlot(3, stream_seed)

        self.assertEqual(pickle.loads(pickle.dumps(slot)).stream_seed, stream_seed)
        self.assertIsInstance(pickle.loads(pickle.dumps(padding)), StreamPaddingSlot)
        self.assertEqual(slot.at(1), 1)
        self.assertEqual(slot.at(1).stream_seed, stream_seed)
        self.assertIsInstance(padding.at(1), StreamPaddingSlot)


if __name__ == "__main__":
    unittest.main()
