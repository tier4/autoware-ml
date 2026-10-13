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

from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_point_cloud_batch,
)
from autoware_ml.preprocessing.segmentation3d.frustum_range import FrustumRangePreprocessor


def _make_batch(
    sample_points: list[torch.Tensor],
    sample_labels: list[torch.Tensor] | None = None,
) -> ModelBatchInputs:
    """Build model inputs from per-sample tensors."""
    point_cloud = build_point_cloud_batch(sample_points)
    segmentation = (
        Segmentation3DGTBatch(
            gt_semantic_masks=torch.cat(sample_labels, dim=0),
            batch_indices=point_cloud.batch_indices,
        )
        if sample_labels is not None
        else None
    )
    return build_batch_inputs(point_cloud=point_cloud, segmentation=segmentation)


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
        batch_inputs = _make_batch(
            sample_points=[
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.1],
                        [2.0, 0.0, 0.0, 0.2],
                        [1.0, 1.0, 0.0, 0.3],
                    ],
                    dtype=torch.float32,
                )
            ],
            sample_labels=[torch.tensor([3, 3, 1], dtype=torch.long)],
        )

        outputs = preprocessor(batch_inputs, is_training=False).range_view_data
        assert outputs is not None

        assert outputs.coors.shape == (3, 3)
        assert outputs.voxel_coors.shape == (2, 3)
        assert outputs.inverse_map.shape == (3,)
        assert outputs.semantic_labels.shape == (1, 2, 4)
        assert outputs.semantic_labels[0, 1, 2].item() == 3
        assert outputs.semantic_labels[0, 1, 1].item() == 1
        assert outputs.semantic_labels[0, 0, 0].item() == 255

    def test_forward_handles_batch_of_two_samples(self) -> None:
        """Multi-sample batches should produce per-sample coordinates and stacked seg maps."""
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
        batch_inputs = _make_batch(
            sample_points=[sample_a, sample_b],
            sample_labels=[
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor([2], dtype=torch.long),
            ],
        )

        outputs = preprocessor(batch_inputs, is_training=False).range_view_data
        assert outputs is not None

        assert outputs.coors[:, 0].tolist() == [0, 0, 1]
        assert outputs.semantic_labels.shape == (2, 2, 4)

    def test_forward_predict_mode_produces_no_label_keys(self) -> None:
        """When the labels are absent, no semantic target image is built."""
        preprocessor = FrustumRangePreprocessor(
            height=2,
            width=4,
            fov_up=10.0,
            fov_down=-10.0,
            ignore_index=255,
            num_classes=4,
        )
        batch_inputs = _make_batch(
            sample_points=[torch.tensor([[1.0, 0.0, 0.0, 0.1]], dtype=torch.float32)]
        )

        outputs = preprocessor(batch_inputs, is_training=False).range_view_data
        assert outputs is not None

        assert outputs.semantic_labels is None
        assert outputs.voxel_coors.shape == (1, 3)

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
        batch_inputs = _make_batch(
            sample_points=[
                torch.tensor([[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2]], dtype=torch.float32)
            ],
            sample_labels=[torch.tensor([-1, 2], dtype=torch.long)],
        )

        outputs = preprocessor(batch_inputs, is_training=False).range_view_data
        assert outputs is not None

        assert outputs.semantic_labels.shape == (1, 2, 4)
        assert (outputs.semantic_labels == 2).any()
        assert (outputs.semantic_labels == -1).any()

    def test_forward_keeps_ignore_only_cells_at_ignore_index(self) -> None:
        """Cells containing only ignored labels should remain ignored."""
        preprocessor = FrustumRangePreprocessor(
            height=2,
            width=4,
            fov_up=10.0,
            fov_down=-10.0,
            ignore_index=255,
            num_classes=3,
        )
        batch_inputs = _make_batch(
            sample_points=[
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2], [1.0, 1.0, 0.0, 0.3]],
                    dtype=torch.float32,
                )
            ],
            sample_labels=[torch.tensor([255, 255, 1], dtype=torch.long)],
        )

        outputs = preprocessor(batch_inputs, is_training=False).range_view_data
        assert outputs is not None

        assert outputs.semantic_labels[0, 1, 2].item() == 255
        assert outputs.semantic_labels[0, 1, 1].item() == 1
