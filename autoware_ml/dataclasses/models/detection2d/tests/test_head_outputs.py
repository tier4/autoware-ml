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

"""Unit tests for the 2D detection head output dataclasses."""

from __future__ import annotations

import unittest

from pydantic import ValidationError
import torch

from autoware_ml.dataclasses.models.detection2d.head_outputs import FocalHead2DOutputs


class TestFocalHead2DOutputs(unittest.TestCase):
    """Unit tests for the dense outputs of the auxiliary 2D head."""

    def setUp(self) -> None:
        """Build the outputs of 4 images of 6 tokens on a 2 x 3 grid and 3 classes."""
        self.outputs = self.build()

    def build(
        self,
        num_tokens_of_boxes: int = 6,
        feature_height: int = 2,
        feature_width: int = 3,
        heatmap_channels: int = 3,
    ) -> FocalHead2DOutputs:
        """The outputs of 4 images of 6 tokens, with one field changed from the valid ones."""
        return FocalHead2DOutputs(
            cls_scores=torch.zeros(4, 6, 3),
            bbox_preds=torch.zeros(4, num_tokens_of_boxes, 4),
            centers_2d=torch.zeros(4, 6, 2),
            heatmap=torch.zeros(4, 6, heatmap_channels),
            feature_height=feature_height,
            feature_width=feature_width,
            image_height=96,
            image_width=160,
        )

    def test_the_heatmap_must_have_one_channel_per_class(self) -> None:
        """Test that a heatmap with other channels than the classes is refused.

        Input: 2D head outputs of 3 classes with a one-channel heatmap.
        Expected: the heatmap channels are weighted by the annotated class mask of the sample,
        one per class, so a heatmap of other channels is rejected when the outputs are built.
        Check: a ValidationError is raised.
        """
        with self.assertRaises(ValidationError):
            self.build(heatmap_channels=1)

    def test_the_token_grid_must_hold_the_tokens(self) -> None:
        """Test that outputs whose token grid does not hold the tokens are refused.

        Input: 2D head outputs of 6 tokens per image with a 2 x 2 token grid.
        Expected: the heatmap targets are drawn on the grid, one cell per token, so a grid of
        another size is rejected when the outputs are built.
        Check: a ValidationError is raised.
        """
        with self.assertRaises(ValidationError):
            self.build(feature_width=2)

    def test_every_prediction_covers_the_same_tokens(self) -> None:
        """Test that outputs whose predictions cover other tokens are refused.

        Input: 2D head outputs whose boxes cover 5 tokens of every image while the logits,
        centers and heatmap cover 6.
        Expected: the predictions of a token are read at the same index of every tensor, so
        outputs covering other tokens are rejected when they are built.
        Check: a ValidationError is raised.
        """
        with self.assertRaises(ValidationError):
            self.build(num_tokens_of_boxes=5)
