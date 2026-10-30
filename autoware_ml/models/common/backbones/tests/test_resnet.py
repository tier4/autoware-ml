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

"""Unit tests for the ResNet backbones: the single-scale ResNet-18, the stage selection of the
multi-scale variants, the stem rebuilt for other input widths, and the parameter names shared
with torchvision."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
from torchvision.models import ResNet50_Weights, resnet18, resnet50

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

    def test_resnet18_returns_three_feature_levels_by_default(self) -> None:
        """Test that ResNet-18 returns ``layer2`` to ``layer4`` at strides 8, 16 and 32 by default."""
        backbone = ResNet18MultiScale(in_channels=3).to(self.device).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(backbone.out_indices, (1, 2, 3))
        self.assertEqual(len(features), 3)
        c3, c4, c5 = features
        self.assertEqual(c3.shape, (self.batch_size, 128, 16, 16))
        self.assertEqual(c4.shape, (self.batch_size, 256, 8, 8))
        self.assertEqual(c5.shape, (self.batch_size, 512, 4, 4))

    def test_resnet18_returns_four_feature_levels_when_requested(self) -> None:
        """Test that ``out_indices=[0, 1, 2, 3]`` adds the ``layer1`` output at stride 4."""
        backbone = (
            ResNet18MultiScale(in_channels=3, out_indices=[0, 1, 2, 3]).to(self.device).eval()
        )

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(backbone.out_indices, (0, 1, 2, 3))
        self.assertEqual(len(features), 4)
        c2, c3, c4, c5 = features
        self.assertEqual(c2.shape, (self.batch_size, 64, 32, 32))
        self.assertEqual(c3.shape, (self.batch_size, 128, 16, 16))
        self.assertEqual(c4.shape, (self.batch_size, 256, 8, 8))
        self.assertEqual(c5.shape, (self.batch_size, 512, 4, 4))

    def test_out_indices_selects_a_subset_of_stages(self) -> None:
        """Test that a sparse selection returns only the chosen stages in increasing order."""
        backbone = ResNet18MultiScale(in_channels=3, out_indices=(0, 3)).to(self.device).eval()

        with torch.no_grad():
            features = backbone(self.images)

        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].shape, (self.batch_size, 64, 32, 32))
        self.assertEqual(features[1].shape, (self.batch_size, 512, 4, 4))

    def test_out_indices_matches_the_four_level_pyramid(self) -> None:
        """Test that a selected stage equals the same stage of the full pyramid."""
        full = ResNet18MultiScale(in_channels=3, out_indices=[0, 1, 2, 3]).to(self.device).eval()
        subset = ResNet18MultiScale(in_channels=3, out_indices=[1, 3]).to(self.device).eval()
        subset.load_state_dict(full.state_dict())

        with torch.no_grad():
            full_features = full(self.images)
            subset_features = subset(self.images)

        torch.testing.assert_close(subset_features[0], full_features[1])
        torch.testing.assert_close(subset_features[1], full_features[3])

    def test_out_indices_rejects_an_empty_selection(self) -> None:
        """Test that at least one stage must be selected."""
        with self.assertRaises(ValueError):
            ResNet18MultiScale(in_channels=3, out_indices=[])

    def test_out_indices_rejects_an_index_out_of_range(self) -> None:
        """Test that indices outside the four stages are rejected."""
        with self.assertRaises(ValueError):
            ResNet18MultiScale(in_channels=3, out_indices=[1, 4])
        with self.assertRaises(ValueError):
            ResNet50MultiScale(in_channels=3, out_indices=[-1, 0])

    def test_out_indices_rejects_a_non_increasing_selection(self) -> None:
        """Test that duplicated or descending indices are rejected."""
        with self.assertRaises(ValueError):
            ResNet18MultiScale(in_channels=3, out_indices=[1, 1, 2])
        with self.assertRaises(ValueError):
            ResNet18MultiScale(in_channels=3, out_indices=[3, 2, 1])

    def test_resnet50_returns_bottleneck_channels(self) -> None:
        """Test that ResNet-50 returns the four-times wider bottleneck channels at the same strides."""
        backbone = (
            ResNet50MultiScale(in_channels=3, out_indices=[0, 1, 2, 3]).to(self.device).eval()
        )

        with torch.no_grad():
            c2, c3, c4, c5 = backbone(self.images)

        self.assertEqual(c2.shape, (self.batch_size, 256, 32, 32))
        self.assertEqual(c3.shape, (self.batch_size, 512, 16, 16))
        self.assertEqual(c4.shape, (self.batch_size, 1024, 8, 8))
        self.assertEqual(c5.shape, (self.batch_size, 2048, 4, 4))

    def test_in_channels_rebuilds_the_stem(self) -> None:
        """Test that a non-RGB input width is accepted by the stem convolution."""
        backbone = ResNet18MultiScale(in_channels=1, out_indices=[0]).to(self.device).eval()
        images = torch.randn(
            self.batch_size, 1, self.image_size, self.image_size, device=self.device
        )

        with torch.no_grad():
            (c2,) = backbone(images)

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
            c5 = pyramid(self.images)[-1]

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


class TestTorchvisionWeights(unittest.TestCase):
    """The backbones start from torchvision weights when asked."""

    def setUp(self) -> None:
        """Save a torchvision ResNet-18 state dict, classifier included, to a temporary file."""
        torch.manual_seed(0)
        self.reference = resnet18()
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "resnet18.pth"
        torch.save(self.reference.state_dict(), self.path)

    def tearDown(self) -> None:
        """Remove the temporary file."""
        self.directory.cleanup()

    def test_state_dict_file_loads_the_stages_and_drops_the_classifier(self) -> None:
        """
        Input: a state dict file of a torchvision ResNet-18 with its classifier.
        Expected: every stage tensor equals the file's, and the classifier is ignored.
        Check: the stem and a deep stage weight match the reference.
        """
        backbone = ResNet18MultiScale(in_channels=3, weights=self.path)

        torch.testing.assert_close(backbone.conv1.weight, self.reference.conv1.weight)
        torch.testing.assert_close(
            backbone.layer4[1].conv2.weight, self.reference.layer4[1].conv2.weight
        )
        self.assertFalse(hasattr(backbone, "fc"))

    def test_other_input_width_keeps_its_own_stem(self) -> None:
        """
        Input: the same file into a five-channel backbone.
        Expected: the stages load but the three-channel stem is skipped.
        Check: a stage weight matches and the stem keeps five input channels.
        """
        backbone = ResNet18(in_channels=5, weights=self.path)

        torch.testing.assert_close(
            backbone.layer1[0].conv1.weight, self.reference.layer1[0].conv1.weight
        )
        self.assertEqual(backbone.conv1.weight.shape[1], 5)

    def test_truncated_state_dict_is_refused(self) -> None:
        """
        Input: the reference state dict without its third and fourth stages, saved to a file.
        Expected: refused instead of leaving those stages at their random draw.
        Check: ValueError naming the file and a missing key.
        """
        truncated = {
            key: value
            for key, value in self.reference.state_dict().items()
            if not key.startswith(("layer3.", "layer4."))
        }
        path = Path(self.directory.name) / "truncated.pth"
        torch.save(truncated, path)

        with self.assertRaisesRegex(ValueError, "(?s)truncated.pth.*layer3.0.conv1.weight"):
            ResNet18MultiScale(in_channels=3, weights=path)

    def test_truncated_state_dict_is_refused_when_the_stem_is_skipped(self) -> None:
        """
        Input: the truncated file into a five-channel backbone, whose stem is skipped.
        Expected: still refused, the skipped stem does not loosen the check.
        Check: ValueError naming a missing key.
        """
        truncated = {
            key: value
            for key, value in self.reference.state_dict().items()
            if not key.startswith("layer4.")
        }
        path = Path(self.directory.name) / "truncated.pth"
        torch.save(truncated, path)

        with self.assertRaisesRegex(ValueError, "layer4.0.conv1.weight"):
            ResNet18(in_channels=5, weights=path)

    def test_wrapped_state_dict_is_refused_with_the_file_name(self) -> None:
        """
        Input: a Lightning-style file wrapping the state dict under ``state_dict``.
        Expected: refused with the file name instead of a bare KeyError.
        Check: ValueError naming the file and the stray key.
        """
        path = Path(self.directory.name) / "wrapped.ckpt"
        torch.save({"state_dict": self.reference.state_dict()}, path)

        with self.assertRaisesRegex(ValueError, "(?s)wrapped.ckpt.*state_dict"):
            ResNet18MultiScale(in_channels=3, weights=path)

    def test_weight_set_names_resolve_through_torchvision(self) -> None:
        """
        Input: an unknown weight set name, and the ImageNet names of ResNet-50.
        Expected: the unknown name is refused with the available names, and the ImageNet
            names exist in torchvision (not downloaded here).
        Check: ValueError naming IMAGENET1K_V1, and the enum members.
        """
        with self.assertRaisesRegex(ValueError, "IMAGENET1K_V1"):
            ResNet50MultiScale(in_channels=3, weights="NOT_A_WEIGHT_SET")
        self.assertIn("IMAGENET1K_V1", ResNet50_Weights.__members__)
        self.assertIn("IMAGENET1K_V2", ResNet50_Weights.__members__)

    def test_load_without_weights_is_unchanged(self) -> None:
        """
        Input: a backbone without weights.
        Expected: the stem keeps its own draw, which differs from the saved reference.
        Check: the stem weights differ.
        """
        backbone = ResNet18MultiScale(in_channels=3)

        self.assertFalse(torch.equal(backbone.conv1.weight, self.reference.conv1.weight))


if __name__ == "__main__":
    unittest.main()
