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

"""Unit tests for the multiview image encoder shared by the camera detectors."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from autoware_ml.models.common.multiview_image_encoder import (
    MultiviewImageEncoder,
    encode_multiview_images,
)


class _TwoScaleBackbone(nn.Module):
    """Stride-4 and stride-8 feature maps, as a tuple like the real multi-scale backbones."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.stage2 = nn.Conv2d(8, 16, kernel_size=2, stride=2)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stride4 = self.stage1(images)
        return stride4, self.stage2(stride4)


class _ProjectionNeck(nn.Module):
    """Projects every map to 4 channels; the first output is the primary feature."""

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(8, 4, 1), nn.Conv2d(16, 4, 1)])

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(project(feature) for project, feature in zip(self.projections, features))


def test_encodes_every_camera_and_keeps_the_first_neck_output() -> None:
    torch.manual_seed(0)
    encoder = MultiviewImageEncoder(_TwoScaleBackbone(), _ProjectionNeck())
    images = torch.randn(2, 3, 3, 32, 48)

    features = encoder(images)

    assert features.shape == (2, 3, 4, 8, 12)
    # Every (sample, camera) slot holds the primary feature of exactly that image.
    expected = encoder.img_neck(encoder.img_backbone(images[1, 2:3]))[0][0]
    torch.testing.assert_close(features[1, 2], expected)


class _FirstMapNeck(nn.Module):
    """Returns the first backbone map as a bare tensor."""

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return features[0]


class _TupleNeck(nn.Module):
    """Returns the backbone maps as a tuple."""

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(features)


def test_a_single_tensor_from_the_backbone_or_the_neck_is_accepted() -> None:
    """Stub modules may return one map instead of a tuple; both forms give the same feature."""
    torch.manual_seed(0)
    # A bare convolution returns one tensor, which reaches the neck as a one-map tuple.
    backbone = nn.Conv2d(3, 8, kernel_size=4, stride=4)
    images = torch.randn(1, 2, 3, 16, 16)

    from_tensor = encode_multiview_images(backbone, _FirstMapNeck(), images)
    from_tuple = encode_multiview_images(backbone, _TupleNeck(), images)

    assert from_tensor.shape == (1, 2, 8, 4, 4)
    torch.testing.assert_close(from_tensor, from_tuple)


def test_rejects_a_batch_without_the_camera_axis() -> None:
    encoder = MultiviewImageEncoder(_TwoScaleBackbone(), _ProjectionNeck())

    with pytest.raises(ValueError, match=r"\(B, N, C, H, W\)"):
        encoder(torch.randn(3, 3, 32, 48))


def test_checkpoint_keys_name_the_backbone_and_the_neck() -> None:
    encoder = MultiviewImageEncoder(_TwoScaleBackbone(), _ProjectionNeck())

    prefixes = {key.split(".")[0] for key in encoder.state_dict()}

    assert prefixes == {"img_backbone", "img_neck"}
