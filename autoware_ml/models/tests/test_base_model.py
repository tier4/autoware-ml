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

"""Unit tests for the step and export contract of ``BaseModel``."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.base import BaseModel


class _PointSumModel(BaseModel):
    """Sum the points of the batch with one learned scale."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return points.sum() * self.scale

    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        point_batch = batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        assert point_batch is not None
        return {"points": point_batch.points}

    def compute_metrics(self, batch_inputs: ModelBatchInputs, outputs: Any) -> dict[str, Any]:
        del batch_inputs
        return {"loss": outputs}


def _batch_inputs() -> ModelBatchInputs:
    points = PointCloudGTBatch(
        points=torch.ones(3, 4),
        batch_indices=torch.tensor([0, 0, 1], dtype=torch.int32),
        batch_size=2,
    )
    return ModelBatchInputs.from_gt_batch(
        ModelGTBatch(
            point_cloud_gt_batch=points,
            detection3d_gt_batch=None,
            segmentation3d_gt_batch=None,
            image_gt_batch=None,
        )
    )


def test_step_runs_forward_on_the_picked_inputs_and_logs_the_sample_count() -> None:
    model = _PointSumModel()
    logged: dict[str, Any] = {}
    model.log_dict = lambda values, batch_size, **kwargs: logged.update(batch_size=batch_size)

    loss = model.training_step(_batch_inputs(), batch_idx=0)

    assert float(loss) == 12.0
    assert logged["batch_size"] == 2


def test_model_without_an_export_fails_loudly() -> None:
    with pytest.raises(NotImplementedError, match="_PointSumModel"):
        _PointSumModel().build_export_specs(_batch_inputs())
