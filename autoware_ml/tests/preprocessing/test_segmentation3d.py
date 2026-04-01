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

"""Unit tests for segmentation preprocessing."""

import torch

from autoware_ml.preprocessing.segmentation3d import FrustumRangePreprocessor


class TestFrustumRangePreprocessor:
    """Tests for FRNet frustum/range preprocessing."""

    def test_forward_builds_sparse_frustum_targets(self) -> None:
        """Project points, merge duplicate frustum cells, and keep point labels."""
        preprocessor = FrustumRangePreprocessor(
            height=2,
            width=4,
            fov_up=10.0,
            fov_down=-10.0,
            ignore_index=255,
            num_classes=4,
        )
        batch_inputs = {
            "points": [
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.1],
                        [2.0, 0.0, 0.0, 0.2],
                        [1.0, 1.0, 0.0, 0.3],
                    ],
                    dtype=torch.float32,
                )
            ],
            "pts_semantic_mask": [torch.tensor([3, 3, 1], dtype=torch.long)],
        }

        outputs = preprocessor(batch_inputs)

        assert outputs["points"].shape == (3, 4)
        assert outputs["coors"].shape == (3, 3)
        assert outputs["voxel_coors"].shape == (2, 3)
        assert outputs["inverse_map"].shape == (3,)
        assert torch.equal(outputs["pts_semantic_mask"], torch.tensor([3, 3, 1]))
        assert outputs["semantic_seg"].shape == (1, 2, 4)
        assert outputs["semantic_seg"][0, 1, 2].item() == 3
        assert outputs["semantic_seg"][0, 1, 1].item() == 1
        assert outputs["semantic_seg"][0, 0, 0].item() == 255

    def test_forward_handles_batch_of_two_samples(self) -> None:
        """Multi-sample batches should produce concatenated point arrays and stacked seg maps."""
        preprocessor = FrustumRangePreprocessor(
            height=2,
            width=4,
            fov_up=10.0,
            fov_down=-10.0,
            ignore_index=255,
            num_classes=3,
        )
        sample_a = torch.tensor([[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2]], dtype=torch.float32)
        sample_b = torch.tensor([[1.0, 1.0, 0.0, 0.3]], dtype=torch.float32)
        batch_inputs = {
            "points": [sample_a, sample_b],
            "pts_semantic_mask": [
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor([2], dtype=torch.long),
            ],
        }

        outputs = preprocessor(batch_inputs)

        assert outputs["points"].shape == (3, 4)
        assert outputs["pts_semantic_mask"].shape == (3,)
        assert outputs["semantic_seg"].shape == (2, 2, 4)
        assert outputs["batch_size"] == 2

    def test_forward_predict_mode_produces_no_label_keys(self) -> None:
        """When pts_semantic_mask is absent, semantic_seg should not appear in output."""
        preprocessor = FrustumRangePreprocessor(
            height=2,
            width=4,
            fov_up=10.0,
            fov_down=-10.0,
            ignore_index=255,
            num_classes=4,
        )
        batch_inputs = {
            "points": [torch.tensor([[1.0, 0.0, 0.0, 0.1]], dtype=torch.float32)],
        }

        outputs = preprocessor(batch_inputs)

        assert "pts_semantic_mask" not in outputs
        assert "semantic_seg" not in outputs
        assert "points" in outputs
        assert "voxel_coors" in outputs

    def test_forward_masks_negative_ignore_labels_before_majority_vote(self) -> None:
        """Ignore labels should be excluded before one-hot voting."""
        preprocessor = FrustumRangePreprocessor(
            height=2,
            width=4,
            fov_up=10.0,
            fov_down=-10.0,
            ignore_index=-1,
            num_classes=3,
        )
        batch_inputs = {
            "points": [
                torch.tensor([[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2]], dtype=torch.float32)
            ],
            "pts_semantic_mask": [torch.tensor([-1, 2], dtype=torch.long)],
        }

        outputs = preprocessor(batch_inputs)

        assert outputs["semantic_seg"].shape == (1, 2, 4)
        assert (outputs["semantic_seg"] == 2).any()
        assert (outputs["semantic_seg"] == -1).any()
