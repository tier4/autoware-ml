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

"""Unit tests for the step and export contract of ``ModuleBaseModel``."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.module_base_model import ModuleBaseModel


class _PointSumModel(ModuleBaseModel):
    """Sum the points of the batch with one learned scale."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> torch.Tensor:
        point_batch = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        assert point_batch is not None
        return point_batch.points.sum() * self.scale

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: Any
    ) -> dict[str, Any]:
        del multi_task_batch_inputs
        return {"loss": multi_task_outputs}


def _batch_inputs() -> ModelBatchInputs:
    points = PointCloudGTBatch(
        points=torch.ones(3, 4),
        batch_indices=torch.tensor([0, 0, 1], dtype=torch.int32),
        batch_size=2,
        timestamp_difference_dim=-1,
    )
    return ModelBatchInputs.from_gt_batch(
        ModelGTBatch(
            point_cloud_gt_batch=points,
            detection3d_gt_batch=None,
            segmentation3d_gt_batch=None,
            image_gt_batch=None,
        )
    )


def test_step_runs_forward_on_the_batch_and_logs_the_sample_count() -> None:
    model = _PointSumModel()
    logged: dict[str, Any] = {}
    model.log_dict = lambda values, batch_size, **kwargs: logged.update(batch_size=batch_size)

    loss = model.training_step(_batch_inputs(), batch_idx=0)

    assert float(loss) == 12.0
    assert logged["batch_size"] == 2


def test_model_without_an_export_fails_loudly() -> None:
    with pytest.raises(NotImplementedError, match="build_export_spec"):
        _PointSumModel().build_export_specs(_batch_inputs())
