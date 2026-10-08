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

"""Equivalence tests for the batched window padding of serialized attention."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from autoware_ml.models.segmentation3d.encoders.ptv3 import SerializedAttention, WindowPadding
from autoware_ml.utils.point_cloud.batching import offset_to_bincount
from autoware_ml.utils.point_cloud.structures import Point


def _reference_padding(offset: torch.Tensor, patch_size: int) -> WindowPadding:
    """The per sample loop the batched implementation replaced, with flash enabled."""
    bincount = offset_to_bincount(offset)
    padded_bincount = (
        torch.maximum(
            torch.div(bincount + patch_size - 1, patch_size, rounding_mode="trunc"),
            torch.ones_like(bincount),
        )
        * patch_size
    )
    mask = bincount > patch_size
    padded_bincount = (~mask).long() * bincount + mask.long() * padded_bincount

    offset = nn.functional.pad(offset, (1, 0))
    padded_offset = nn.functional.pad(torch.cumsum(padded_bincount, dim=0), (1, 0))
    pad = torch.arange(padded_offset[-1], device=offset.device)
    unpad = torch.arange(offset[-1], device=offset.device)
    cu_seqlens = []
    for batch_index in range(offset.numel() - 1):
        unpad[offset[batch_index] : offset[batch_index + 1]] += (
            padded_offset[batch_index] - offset[batch_index]
        )
        if bincount[batch_index] != padded_bincount[batch_index]:
            pad[
                padded_offset[batch_index + 1]
                - patch_size
                + (bincount[batch_index] % patch_size) : padded_offset[batch_index + 1]
            ] = pad[
                padded_offset[batch_index + 1]
                - 2 * patch_size
                + (bincount[batch_index] % patch_size) : padded_offset[batch_index + 1] - patch_size
            ]
        pad[padded_offset[batch_index] : padded_offset[batch_index + 1]] -= (
            padded_offset[batch_index] - offset[batch_index]
        )
        cu_seqlens.append(
            torch.arange(
                padded_offset[batch_index],
                padded_offset[batch_index + 1],
                step=patch_size,
                dtype=torch.int32,
                device=offset.device,
            )
        )
    return WindowPadding(
        pad, unpad, nn.functional.pad(torch.cat(cu_seqlens), (0, 1), value=padded_offset[-1])
    )


def _offsets(counts: list[int]) -> torch.Tensor:
    """Cumulative offsets of per sample point counts."""
    return torch.cumsum(torch.tensor(counts, dtype=torch.long), dim=0)


def _attention(enable_flash: bool) -> SerializedAttention:
    """Serialized attention with a window of 8 tokens."""
    attention = SerializedAttention(
        channels=8,
        num_heads=2,
        patch_size=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=enable_flash,
        upcast_attention=False,
        upcast_softmax=False,
    )
    # Without flash attention the forward pass sizes the window per batch; set it here.
    attention.patch_size = 8
    return attention


class TestWindowPadding(unittest.TestCase):
    """The batched window padding reproduces the per sample loop it replaced."""

    _CASES = (
        ("single_sample_below_window", [5], 8),
        ("single_sample_exact_window", [8], 8),
        ("single_sample_one_over", [9], 8),
        ("mixed_including_empty", [17, 3, 16, 0, 25], 8),
        ("all_empty", [0, 0], 8),
        ("exact_multiples", [32, 64, 96], 32),
        ("remainders", [33, 1, 31, 65], 32),
        ("training_window", [1000, 1024, 1025, 7], 1024),
    )

    def _assert_matches_reference(self, offset: torch.Tensor, patch_size: int) -> None:
        """Assert that the batched padding equals the reference loop."""
        padding = SerializedAttention._batched_window_padding(offset, patch_size)
        reference = _reference_padding(offset, patch_size)
        self.assertTrue(torch.equal(padding.pad, reference.pad))
        self.assertTrue(torch.equal(padding.unpad, reference.unpad))
        self.assertEqual(padding.cu_seqlens.dtype, torch.int32)
        self.assertTrue(torch.equal(padding.cu_seqlens, reference.cu_seqlens))

    def test_batched_padding_matches_the_reference_loop(self) -> None:
        """Test the batched padding against the reference loop on hand-picked batches."""
        for name, counts, patch_size in self._CASES:
            with self.subTest(name):
                self._assert_matches_reference(_offsets(counts), patch_size)

    def test_batched_padding_matches_the_reference_loop_on_random_batches(self) -> None:
        """Test the batched padding against the reference loop on random batches."""
        generator = torch.Generator().manual_seed(1234)
        for _ in range(200):
            patch_size = int(torch.randint(1, 12, (1,), generator=generator))
            num_samples = int(torch.randint(1, 6, (1,), generator=generator))
            counts = torch.randint(0, 4 * patch_size + 1, (num_samples,), generator=generator)
            with self.subTest(counts=counts.tolist(), patch_size=patch_size):
                self._assert_matches_reference(torch.cumsum(counts, dim=0), patch_size)

    def test_padding_round_trips_every_real_token(self) -> None:
        """Test that every token lands in its own slot and every slot holds a real token."""
        offset = _offsets([17, 3, 25])
        padding = SerializedAttention._batched_window_padding(offset, 8)
        self.assertTrue(torch.equal(padding.pad[padding.unpad], torch.arange(int(offset[-1]))))
        self.assertGreaterEqual(int(padding.pad.min()), 0)
        self.assertLess(int(padding.pad.max()), int(offset[-1]))

    def test_attention_caches_one_padding_per_window_size(self) -> None:
        """Test that one cached entry serves attention with and without flash."""
        point = Point(offset=_offsets([17, 3]))
        padding = _attention(enable_flash=False)._get_padding_and_inverse(point)
        self.assertIsInstance(padding, WindowPadding)
        self.assertIs(point["window_padding_8"], padding)
        self.assertIsNotNone(padding.cu_seqlens)
        self.assertIs(_attention(enable_flash=False)._get_padding_and_inverse(point), padding)
        self.assertIs(_attention(enable_flash=True)._get_padding_and_inverse(point), padding)
