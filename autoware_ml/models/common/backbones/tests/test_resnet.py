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

"""Unit tests for the ResNet backbones: the single-scale ResNet-18, the four-level feature
pyramids of the multi-scale variants, the stem rebuilt for other input widths, and the parameter
names shared with torchvision."""

from __future__ import annotations

import unittest

import torch
from torchvision.models import resnet18, resnet50

from autoware_ml.models.common.backbones.resnet import (
    ResNet18,
    ResNet18MultiScale,
    ResNet50MultiScale,
)


class _ResNetTestCase(unittest.TestCase):
    """Shared inputs for the ResNet test cases: a two-image batch of 128x128 RGB inputs."""

    def setUp(self) -> None:
        """Set up the batch on the available device."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.image_size = 128
        self.images = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size, device=self.device
        )


class TestResNetMultiScale(_ResNetTestCase):
    """Unit tests for the multi-scale ResNet feature pyramids."""

    def test_resnet18_returns_four_feature_levels(self) -> None:
        """Test that ResNet-18 returns ``layer1`` to ``layer4`` at strides 4, 8, 16 and 32."""
        backbone = ResNet18MultiScale(in_channels=3).to(self.device).eval()

        with torch.no_grad():
            c2, c3, c4, c5 = backbone(self.images)

        self.assertEqual(c2.shape, (self.batch_size, 64, 32, 32))
        self.assertEqual(c3.shape, (self.batch_size, 128, 16, 16))
        self.assertEqual(c4.shape, (self.batch_size, 256, 8, 8))
        self.assertEqual(c5.shape, (self.batch_size, 512, 4, 4))

    def test_resnet50_returns_bottleneck_channels(self) -> None:
        """Test that ResNet-50 returns the four-times wider bottleneck channels at the same strides."""
        backbone = ResNet50MultiScale(in_channels=3).to(self.device).eval()

        with torch.no_grad():
            c2, c3, c4, c5 = backbone(self.images)

        self.assertEqual(c2.shape, (self.batch_size, 256, 32, 32))
        self.assertEqual(c3.shape, (self.batch_size, 512, 16, 16))
        self.assertEqual(c4.shape, (self.batch_size, 1024, 8, 8))
        self.assertEqual(c5.shape, (self.batch_size, 2048, 4, 4))

    def test_in_channels_rebuilds_the_stem(self) -> None:
        """Test that a non-RGB input width is accepted by the stem convolution."""
        backbone = ResNet18MultiScale(in_channels=1).to(self.device).eval()
        images = torch.randn(
            self.batch_size, 1, self.image_size, self.image_size, device=self.device
        )

        with torch.no_grad():
            c2, _, _, _ = backbone(images)

        self.assertEqual(backbone.conv1.in_channels, 1)
        self.assertEqual(c2.shape, (self.batch_size, 64, 32, 32))

    def test_classification_layers_are_removed(self) -> None:
        """Test that the average pooling and the classifier head are dropped from the backbone."""
        backbone = ResNet18MultiScale(in_channels=3)

        self.assertFalse(hasattr(backbone, "avgpool"))
        self.assertFalse(hasattr(backbone, "fc"))


class TestResNet18(_ResNetTestCase):
    """Unit tests for the single-scale ResNet-18 backbone."""

    def test_returns_the_last_stage(self) -> None:
        """Test that the output is the ``layer4`` feature map, equal to the last pyramid level."""
        backbone = ResNet18(in_channels=3).to(self.device).eval()
        pyramid = ResNet18MultiScale(in_channels=3).to(self.device).eval()
        pyramid.load_state_dict(backbone.state_dict())

        with torch.no_grad():
            features = backbone(self.images)
            _, _, _, c5 = pyramid(self.images)

        self.assertEqual(features.shape, (self.batch_size, 512, 4, 4))
        torch.testing.assert_close(features, c5)


class TestTorchvisionCompatibility(unittest.TestCase):
    """The backbones keep the parameter names of a torchvision ResNet, minus the classifier."""

    def test_resnet18_state_dict_keys_match_torchvision(self) -> None:
        """Test that an ImageNet ResNet-18 checkpoint maps onto the backbone without the head."""
        expected = {k for k in resnet18().state_dict() if not k.startswith("fc.")}

        self.assertEqual(set(ResNet18(in_channels=3).state_dict()), expected)
        self.assertEqual(set(ResNet18MultiScale(in_channels=3).state_dict()), expected)

    def test_resnet50_state_dict_keys_match_torchvision(self) -> None:
        """Test that an ImageNet ResNet-50 checkpoint maps onto the backbone without the head."""
        expected = {k for k in resnet50().state_dict() if not k.startswith("fc.")}

        self.assertEqual(set(ResNet50MultiScale(in_channels=3).state_dict()), expected)

    def test_torchvision_weights_load_strictly(self) -> None:
        """Test that the torchvision stage weights load with ``strict=True`` once the head is dropped."""
        weights = {k: v for k, v in resnet18().state_dict().items() if not k.startswith("fc.")}

        ResNet18MultiScale(in_channels=3).load_state_dict(weights, strict=True)


if __name__ == "__main__":
    unittest.main()
