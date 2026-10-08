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

"""Unit tests for the DataPreprocessing pipeline wrapper."""

import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.preprocessing.base import DataPreprocessing


class _ModeRecorder:
    """Minimal pipeline stage that records the mode it was called with."""

    def __init__(self) -> None:
        self.seen_modes: list[bool] = []

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        self.seen_modes.append(is_training)
        return batch_inputs


def _batch() -> ModelGTBatch:
    """Build a one sample batch holding two points."""
    return ModelGTBatch(
        point_cloud_gt_batch=PointCloudGTBatch(
            points=torch.zeros((2, 4), dtype=torch.float32),
            batch_indices=torch.zeros(2, dtype=torch.int32),
            batch_size=1,
            timestamp_difference_dim=-1,
        ),
        detection3d_gt_batch=None,
        segmentation3d_gt_batch=None,
        image_gt_batch=None,
    )


def test_call_forwards_is_training_to_every_layer():
    """The pipeline is not a registered submodule, so the owning model's mode reaches
    the stages only through the explicit is_training argument."""
    first, second = _ModeRecorder(), _ModeRecorder()
    pipeline = DataPreprocessing([first, second])

    pipeline(_batch(), is_training=True)
    pipeline(_batch(), is_training=False)

    assert first.seen_modes == [True, False]
    assert second.seen_modes == [True, False]


def test_call_requires_explicit_is_training():
    """The mode must be stated on every call; forgetting it is an immediate TypeError
    instead of silently running in the wrong mode."""
    pipeline = DataPreprocessing([_ModeRecorder()])

    with pytest.raises(TypeError):
        pipeline(_batch())  # type: ignore[call-arg]


def test_call_chains_the_layers_from_the_collated_batch():
    """Every layer receives the inputs the previous one returned, starting from the batch."""
    batch = _batch()
    voxels = VoxelsData(
        voxels=torch.zeros((1, 2, 4)),
        coords=torch.zeros((1, 3), dtype=torch.int32),
        num_points=torch.full((1,), 2, dtype=torch.int32),
        batch_indices=torch.zeros(1, dtype=torch.int32),
        point_voxel_indices=torch.zeros(2, dtype=torch.int64),
        num_dropped_voxels=torch.zeros((), dtype=torch.int64),
    )
    seen: list[ModelBatchInputs] = []

    def voxelize(batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        seen.append(batch_inputs)
        return batch_inputs.replace(voxels_data=voxels)

    def check(batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        seen.append(batch_inputs)
        return batch_inputs

    result = DataPreprocessing([voxelize, check])(batch, is_training=True)

    assert seen[0].multi_task_gt_batch is batch
    assert seen[0].voxels_data is None
    assert seen[1].voxels_data is voxels
    assert result is seen[1]
