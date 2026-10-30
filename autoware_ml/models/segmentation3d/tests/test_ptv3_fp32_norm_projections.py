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

"""The projections that feed a BatchNorm stay in fp32 under autocast."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    SerializedUnpooling,
    _LinearBeforeNorm,
    linear_before_norm,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast on CUDA")


@requires_cuda
def test_projection_survives_inputs_that_overflow_fp16() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(64, 32).cuda()
    with torch.no_grad():
        linear.weight.mul_(40.0)
    feat = torch.randn(1000, 64, device="cuda") * 3000.0  # products exceed 65,504
    with torch.autocast("cuda", dtype=torch.float16):
        naive = linear(feat)
        guarded = linear_before_norm(linear, feat)
    assert not torch.isfinite(naive).all(), "the test input must overflow the plain fp16 path"
    assert guarded.dtype == torch.float32
    assert torch.isfinite(guarded).all()
    torch.testing.assert_close(guarded, linear(feat.float()), rtol=1e-4, atol=1e-2)


def test_linear_before_norm_is_a_plain_linear_outside_autocast() -> None:
    torch.manual_seed(1)
    layer = _LinearBeforeNorm(8, 4)
    plain = nn.Linear(8, 4)
    plain.load_state_dict(layer.state_dict())
    x = torch.randn(5, 8)
    torch.testing.assert_close(layer(x), plain(x))
    assert set(layer.state_dict()) == {"weight", "bias"}


def test_unpooling_state_dict_layout_is_unchanged() -> None:
    module = SerializedUnpooling(16, 8, 12)
    keys = sorted(module.state_dict())
    assert "proj.0.weight" in keys and "proj_skip.0.weight" in keys
    assert "proj.1.running_mean" in keys and "proj_skip.1.running_mean" in keys
