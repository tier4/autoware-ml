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

"""Unit tests for the augmentation the streaming sampler draws once per scene."""

from __future__ import annotations

from collections.abc import Sequence
import pickle
import unittest

import numpy as np

from autoware_ml.datamodule.base_dataset import PaddingSlot, StreamPaddingSlot, StreamSlot
from autoware_ml.datamodule.samplers import GroupStreamingSampler
from autoware_ml.transforms.base import StreamParameters


class _AugmentedScenesDataset:
    """Minimal dataset stub exposing its scenes and the draw of a stream augmentation."""

    def __init__(self, scene_lengths: Sequence[int]) -> None:
        self._groups = []
        start = 0
        for length in scene_lengths:
            self._groups.append(list(range(start, start + length)))
            start += length

    def scene_index_groups(self) -> Sequence[Sequence[int]]:
        return [list(group) for group in self._groups]

    def sample_stream_augmentation(self, index: int) -> tuple[float]:
        """One value drawn as a pipeline draws the rotation of a stream."""
        return (float(np.random.uniform()),)

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
            seed: Seed of the scene shuffle and of the augmentation draws.
            shuffle: Whether the scenes are shuffled every epoch.
            consistent_augmentation: Whether a scene keeps one augmentation.

        Returns:
            The sampler over a stub dataset holding the scenes.
        """
        return GroupStreamingSampler(
            _AugmentedScenesDataset(scene_lengths),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            consistent_augmentation=consistent_augmentation,
        )

    def lanes(self, sampler: GroupStreamingSampler) -> Sequence[Sequence[int]]:
        """Split the indices of one epoch into the lanes they feed, batch position by position."""
        indices = list(iter(sampler))
        return [indices[lane :: sampler.batch_size] for lane in range(sampler.batch_size)]

    def drawn(self, sampler: GroupStreamingSampler, epoch: int) -> Sequence[tuple]:
        """The augmentation carried by every slot of one epoch."""
        sampler.set_epoch(epoch)
        return [slot.augmentation for slot in sampler]

    def drawn_per_scene(self, sampler: GroupStreamingSampler) -> dict[int, set[tuple]]:
        """The augmentations the slots of every scene carry in the current epoch."""
        scene_of = {
            index: scene for scene, group in enumerate(sampler.scene_groups) for index in group
        }
        drawn: dict[int, set[tuple]] = {}
        for slot in sampler:
            drawn.setdefault(scene_of[int(slot)], set()).add(slot.augmentation)
        return drawn

    def test_frames_of_a_scene_carry_one_augmentation_and_scenes_differ(self) -> None:
        """
        Input: scenes of 6, 3 and 3 frames through two lanes, with and without the consistent
        augmentation.
        Expected: the same frames in the same order either way, every frame a stream slot
        whose augmentation is the one drawn for its scene.
        Check: the indices match the plain sampler, every scene carries one augmentation
        and the three scenes carry three different ones.
        """
        sampler = self.build_stream_sampler([6, 3, 3], batch_size=2)
        plain = self.build_stream_sampler([6, 3, 3], batch_size=2, consistent_augmentation=False)

        indices = list(sampler)
        drawn = self.drawn_per_scene(sampler)

        self.assertEqual([int(index) for index in indices], list(plain))
        self.assertTrue(all(isinstance(index, StreamSlot) for index in indices))
        self.assertEqual([len(drawn[scene]) for scene in range(3)], [1, 1, 1])
        self.assertEqual(len(drawn[0] | drawn[1] | drawn[2]), 3)

    def test_a_scene_is_augmented_anew_every_epoch_and_alike_in_every_process(self) -> None:
        """
        Input: two samplers built alike, as the ranks or a resumed run build them, and a
        third one with another seed.
        Expected: the draw depends only on the seed, the epoch and the scene.
        Check: the two samplers draw the same in epoch 0, and neither the next epoch nor
        the other seed repeats any of those draws.
        """
        first = self.build_stream_sampler([4, 4], batch_size=2)
        second = self.build_stream_sampler([4, 4], batch_size=2)
        other_run = self.build_stream_sampler([4, 4], batch_size=2, seed=1)

        self.assertEqual(self.drawn(first, 0), self.drawn(second, 0))
        self.assertTrue(set(self.drawn(first, 0)).isdisjoint(self.drawn(first, 1)))
        self.assertTrue(set(self.drawn(first, 0)).isdisjoint(self.drawn(other_run, 0)))

    def test_drawing_the_augmentation_leaves_the_random_state_as_it_was(self) -> None:
        """
        Input: a seeded NumPy random state, read once directly and once after the sampler
        drew the augmentation of an epoch.
        Expected: the sampler restores the random state it seeded for its draws.
        Check: the three values drawn after the epoch are the ones drawn without it.
        """
        sampler = self.build_stream_sampler([4, 4], batch_size=2)
        np.random.seed(11)
        expected = np.random.uniform(size=3).tolist()

        np.random.seed(11)
        list(sampler)

        self.assertEqual(np.random.uniform(size=3).tolist(), expected)

    def test_a_scene_keeps_its_augmentation_when_shuffled_and_cut_into_lanes(self) -> None:
        """
        Input: scenes of 5, 4, 3 and 2 frames shuffled through three lanes, so scenes are
        cut across lanes.
        Expected: a scene keeps one augmentation wherever its frames land.
        Check: every one of the four scenes carries exactly one augmentation.
        """
        sampler = self.build_stream_sampler([5, 4, 3, 2], batch_size=3, shuffle=True)

        drawn = self.drawn_per_scene(sampler)

        self.assertEqual(sorted(drawn), [0, 1, 2, 3])
        self.assertTrue(all(len(scene_drawn) == 1 for scene_drawn in drawn.values()))

    def test_a_padding_slot_keeps_the_augmentation_of_the_frame_it_repeats(self) -> None:
        """
        Input: scenes of 3 and 2 frames through two lanes, cut into lanes of two and three
        frames, so the shorter lane ends with a padding slot.
        Expected: the padding slot repeats the last frame of its lane under the same
        augmentation.
        Check: the slot is a StreamPaddingSlot and a PaddingSlot, with the index and the
        augmentation of the frame before it.
        """
        sampler = self.build_stream_sampler([3, 2], batch_size=2)

        short_lane = self.lanes(sampler)[0]
        padding = short_lane[-1]

        self.assertIsInstance(padding, StreamPaddingSlot)
        self.assertIsInstance(padding, PaddingSlot)
        self.assertEqual(padding, short_lane[-2])
        self.assertEqual(padding.augmentation, short_lane[-2].augmentation)

    def test_a_stream_slot_keeps_its_augmentation_through_a_worker_and_a_split(self) -> None:
        """
        Input: a stream slot and a stream padding slot, pickled as for a dataloader worker
        and moved to another index as a concatenated split does.
        Expected: both keep their augmentation and their kind.
        Check: the unpickled and the moved slots carry the augmentation, and the padding
        slot stays a StreamPaddingSlot.
        """
        augmentation = (StreamParameters(True, (0.1, 1.02, (0.0, 0.1, 0.0))), None)
        slot = StreamSlot(3, augmentation)
        padding = StreamPaddingSlot(3, augmentation)

        self.assertEqual(pickle.loads(pickle.dumps(slot)).augmentation, augmentation)
        self.assertIsInstance(pickle.loads(pickle.dumps(padding)), StreamPaddingSlot)
        self.assertEqual(slot.at(1), 1)
        self.assertEqual(slot.at(1).augmentation, augmentation)
        self.assertIsInstance(padding.at(1), StreamPaddingSlot)


if __name__ == "__main__":
    unittest.main()
