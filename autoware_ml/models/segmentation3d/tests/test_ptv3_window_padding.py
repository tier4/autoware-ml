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

import pytest
import torch
from torch import nn

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    SerializedAttention,
    serialized_window_padding,
)
from autoware_ml.utils.point_cloud.batching import offset_to_bincount
from autoware_ml.utils.point_cloud.structures import Point


def _reference_padding(
    offset: torch.Tensor, patch_size: int, enable_flash: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The per sample loop the batched implementation replaced, kept verbatim."""
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
    cu_seqlens = [] if enable_flash else None
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
        if cu_seqlens is not None:
            cu_seqlens.append(
                torch.arange(
                    padded_offset[batch_index],
                    padded_offset[batch_index + 1],
                    step=patch_size,
                    dtype=torch.int32,
                    device=offset.device,
                )
            )
    if cu_seqlens is None:
        return pad, unpad, None
    return pad, unpad, nn.functional.pad(torch.cat(cu_seqlens), (0, 1), value=padded_offset[-1])


def _offsets(counts: list[int]) -> torch.Tensor:
    return torch.cumsum(torch.tensor(counts, dtype=torch.long), dim=0)


_CASES = [
    pytest.param([5], 8, id="single_sample_below_window"),
    pytest.param([8], 8, id="single_sample_exact_window"),
    pytest.param([9], 8, id="single_sample_one_over"),
    pytest.param([17, 3, 16, 0, 25], 8, id="mixed_including_empty"),
    pytest.param([0, 0], 8, id="all_empty"),
    pytest.param([32, 64, 96], 32, id="exact_multiples"),
    pytest.param([33, 1, 31, 65], 32, id="remainders"),
    pytest.param([1000, 1024, 1025, 7], 1024, id="training_window"),
]


@pytest.mark.parametrize(("counts", "patch_size"), _CASES)
@pytest.mark.parametrize("enable_flash", [True, False], ids=["flash", "dense"])
def test_batched_padding_matches_the_reference_loop(
    counts: list[int], patch_size: int, enable_flash: bool
) -> None:
    offset = _offsets(counts)
    pad, unpad, cu_seqlens = serialized_window_padding(offset, patch_size, enable_flash)
    ref_pad, ref_unpad, ref_cu_seqlens = _reference_padding(offset, patch_size, enable_flash)
    assert torch.equal(pad, ref_pad)
    assert torch.equal(unpad, ref_unpad)
    if enable_flash:
        assert ref_cu_seqlens is not None and cu_seqlens is not None
        assert cu_seqlens.dtype == ref_cu_seqlens.dtype == torch.int32
        assert torch.equal(cu_seqlens, ref_cu_seqlens)
    else:
        assert cu_seqlens is None and ref_cu_seqlens is None


def test_batched_padding_matches_the_reference_loop_on_random_batches() -> None:
    generator = torch.Generator().manual_seed(1234)
    for _ in range(200):
        patch_size = int(torch.randint(1, 12, (1,), generator=generator))
        num_samples = int(torch.randint(1, 6, (1,), generator=generator))
        counts = torch.randint(0, 4 * patch_size + 1, (num_samples,), generator=generator)
        offset = torch.cumsum(counts, dim=0)
        for enable_flash in (True, False):
            pad, unpad, cu_seqlens = serialized_window_padding(offset, patch_size, enable_flash)
            ref = _reference_padding(offset, patch_size, enable_flash)
            assert torch.equal(pad, ref[0]), (counts.tolist(), patch_size)
            assert torch.equal(unpad, ref[1]), (counts.tolist(), patch_size)
            if enable_flash:
                assert torch.equal(cu_seqlens, ref[2]), (counts.tolist(), patch_size)


def test_padding_round_trips_every_real_token() -> None:
    offset = _offsets([17, 3, 25])
    pad, unpad, _ = serialized_window_padding(offset, 8, True)
    # Every token lands in its own slot, and every slot holds a real token.
    assert torch.equal(pad[unpad], torch.arange(int(offset[-1])))
    assert pad.min() >= 0 and pad.max() < int(offset[-1])


def test_attention_caches_the_padding_on_the_point() -> None:
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
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    )
    # Without flash attention the forward pass sizes the window per batch; set it here.
    attention.patch_size = 8
    point = Point(offset=_offsets([17, 3]))
    pad, unpad, cu_seqlens = attention._get_padding_and_inverse(point)
    assert cu_seqlens is None
    assert "pad_8" in point and "unpad_8" in point and "cu_seqlens_8" not in point
    again = attention._get_padding_and_inverse(point)
    assert again[0] is pad and again[1] is unpad
