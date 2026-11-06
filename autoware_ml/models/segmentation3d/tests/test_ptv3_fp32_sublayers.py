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

import importlib.util
import unittest

import torch

from autoware_ml.models.segmentation3d.encoders.ptv3 import SerializedAttention
from autoware_ml.utils.point_cloud.structures import Point

_CUDA_AND_FLASH_AVAILABLE = (
    torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None
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


@unittest.skipUnless(_CUDA_AND_FLASH_AVAILABLE, "CUDA and flash attention required")
class TestSerializedAttentionFp32Projections(unittest.TestCase):
    """Unit tests for the fp32 attention projections."""

    def test_fp32_projections_keep_the_output_in_fp32_under_autocast(self) -> None:
        """Test that the output stays in fp32 under autocast while the plain path does not."""
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                with torch.autocast("cuda", dtype=dtype):
                    low = _attention(fp32_projections=False)(_point()).feat
                    high = _attention(fp32_projections=True)(_point()).feat
                self.assertEqual(low.dtype, dtype)
                self.assertEqual(high.dtype, torch.float32)
                self.assertTrue(torch.isfinite(high).all())

    def test_attention_without_autocast_is_unchanged_by_the_flag(self) -> None:
        """Test that the flag does not change the output outside autocast."""
        torch.manual_seed(1)
        plain, guarded = _attention(False), _attention(True)
        guarded.load_state_dict(plain.state_dict())
        point = _point()
        out_plain = plain(Point(dict(point))).feat
        out_guarded = guarded(Point(dict(point))).feat
        torch.testing.assert_close(out_plain, out_guarded)


if __name__ == "__main__":
    unittest.main()
