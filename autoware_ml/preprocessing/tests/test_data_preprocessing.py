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

"""Unit tests for the DataPreprocessor pipeline wrapper."""

import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule


class _ModeRecorder(DataPreprocessorModule):
    """Minimal module that records the mode it was called with."""

    def __init__(self) -> None:
        self.seen_modes: list[bool] = []

    def __call__(self, batch: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        self.seen_modes.append(is_training)
        return batch


class _Voxelize(DataPreprocessorModule):
    """Module that attaches fixed voxels and records the inputs it saw."""

    def __init__(self, voxels: VoxelsData, seen: list[ModelBatchInputs]) -> None:
        self.voxels = voxels
        self.seen = seen

    def __call__(self, batch: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        self.seen.append(batch)
        return batch.replace(voxels_data=self.voxels)


class _Check(DataPreprocessorModule):
    """Module that records the inputs it saw and passes them through."""

    def __init__(self, seen: list[ModelBatchInputs]) -> None:
        self.seen = seen

    def __call__(self, batch: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        self.seen.append(batch)
        return batch


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


def test_call_forwards_is_training_to_every_module():
    """The preprocessor is not a registered submodule, so the owning model's mode reaches
    the modules only through the explicit is_training argument."""
    first, second = _ModeRecorder(), _ModeRecorder()
    preprocessor = DataPreprocessor(preprocessor_modules=[first, second])

    preprocessor(_batch(), is_training=True)
    preprocessor(_batch(), is_training=False)

    assert first.seen_modes == [True, False]
    assert second.seen_modes == [True, False]


def test_call_requires_explicit_is_training():
    """The mode must be stated on every call; forgetting it is an immediate TypeError
    instead of silently running in the wrong mode."""
    preprocessor = DataPreprocessor(preprocessor_modules=[_ModeRecorder()])

    with pytest.raises(TypeError):
        preprocessor(_batch())  # type: ignore[call-arg]


def test_call_chains_the_modules_from_the_collated_batch():
    """Every module receives the inputs the previous one returned, starting from the batch."""
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
    preprocessor = DataPreprocessor(preprocessor_modules=[_Voxelize(voxels, seen), _Check(seen)])

    result = preprocessor(batch, is_training=True)

    assert seen[0].multi_task_gt_batch is batch
    assert seen[0].voxels_data is None
    assert seen[1].voxels_data is voxels
    assert result is seen[1]


def test_rejects_a_module_outside_the_interface():
    """A callable that does not implement DataPreprocessorModule is refused up front."""

    def stage(batch: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        return batch

    with pytest.raises(TypeError, match="DataPreprocessorModule"):
        DataPreprocessor(preprocessor_modules=[stage])  # type: ignore[list-item]


def test_repr_names_the_modules_in_order():
    """The representation the model builder logs lists the module types in run order."""
    preprocessor = DataPreprocessor(preprocessor_modules=[_ModeRecorder(), _Check([])])

    assert repr(preprocessor) == "DataPreprocessor(preprocessor_modules=[_ModeRecorder, _Check])"
