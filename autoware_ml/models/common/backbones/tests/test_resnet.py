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

"""Unit tests for the multi-scale ResNet backbones."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.common.backbones.resnet import ResNet18MultiScale, ResNet50MultiScale


class TestResNetMultiScale(unittest.TestCase):
    """Unit tests for the multi-scale ResNet feature pyramids."""

    def setUp(self) -> None:
        """Set up a two-image batch of 128x128 RGB inputs."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.image_size = 128
        self.images = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size, device=self.device
        )

    def test_resnet18_returns_three_feature_levels(self) -> None:
        """Test that ResNet-18 returns ``layer2`` to ``layer4`` at strides 8, 16 and 32."""
        backbone = ResNet18MultiScale(in_channels=3).to(self.device).eval()

        with torch.no_grad():
            c3, c4, c5 = backbone(self.images)

        self.assertEqual(c3.shape, (self.batch_size, 128, 16, 16))
        self.assertEqual(c4.shape, (self.batch_size, 256, 8, 8))
        self.assertEqual(c5.shape, (self.batch_size, 512, 4, 4))

    def test_resnet50_returns_bottleneck_channels(self) -> None:
        """Test that ResNet-50 returns the four-times wider bottleneck channels at the same strides."""
        backbone = ResNet50MultiScale(in_channels=3).to(self.device).eval()

        with torch.no_grad():
            c3, c4, c5 = backbone(self.images)

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
            c3, _, _ = backbone(images)

        self.assertEqual(backbone.conv1.in_channels, 1)
        self.assertEqual(c3.shape, (self.batch_size, 128, 16, 16))

    def test_classification_layers_are_removed(self) -> None:
        """Test that the average pooling and the classifier head are dropped from the backbone."""
        backbone = ResNet18MultiScale(in_channels=3)

        self.assertFalse(hasattr(backbone, "avgpool"))
        self.assertFalse(hasattr(backbone, "fc"))


if __name__ == "__main__":
    unittest.main()
