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

"""Sublayers that stay in fp32 under autocast."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.segmentation3d.encoders.ptv3 import SerializedAttention
from autoware_ml.utils.point_cloud.structures import Point

requires_cuda_flash = pytest.mark.skipif(
    not torch.cuda.is_available() or pytest.importorskip("flash_attn") is None,
    reason="CUDA and flash attention required",
)


def _attention(fp32_projections: bool) -> SerializedAttention:
    return SerializedAttention(
        channels=64,
        num_heads=2,
        patch_size=16,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        fp32_projections=fp32_projections,
    ).cuda()


def _point(num_points: int = 64) -> Point:
    return Point(
        {
            "feat": torch.randn(num_points, 64, device="cuda"),
            "grid_coord": torch.randint(0, 8, (num_points, 3), dtype=torch.int32, device="cuda"),
            "serialized_order": torch.arange(num_points, device="cuda").reshape(1, -1),
            "serialized_inverse": torch.arange(num_points, device="cuda").reshape(1, -1),
            "offset": torch.tensor([num_points], dtype=torch.long, device="cuda"),
        }
    )


@requires_cuda_flash
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp32_projections_keep_the_output_in_fp32_under_autocast(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    with torch.autocast("cuda", dtype=dtype):
        low = _attention(fp32_projections=False)(_point()).feat
        high = _attention(fp32_projections=True)(_point()).feat
    assert low.dtype == dtype
    assert high.dtype == torch.float32
    assert torch.isfinite(high).all()


@requires_cuda_flash
def test_attention_without_autocast_is_unchanged_by_the_flag() -> None:
    torch.manual_seed(1)
    plain, guarded = _attention(False), _attention(True)
    guarded.load_state_dict(plain.state_dict())
    point = _point()
    out_plain = plain(Point(dict(point))).feat
    out_guarded = guarded(Point(dict(point))).feat
    torch.testing.assert_close(out_plain, out_guarded)
