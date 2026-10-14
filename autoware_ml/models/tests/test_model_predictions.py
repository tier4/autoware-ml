"""Framework prediction-step contracts for ModuleBaseModel."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule


class _ShiftPoints(DataPreprocessorModule):
    """Preprocessing module shifting the points of the batch by one."""

    def __call__(self, batch_inputs: ModelBatchInputs, *, is_training: bool) -> ModelBatchInputs:
        del is_training
        gt_batch = batch_inputs.multi_task_gt_batch
        points = gt_batch.point_cloud_gt_batch
        assert points is not None
        shifted = points._replace(points=points.points + 1.0)
        return batch_inputs.replace(
            multi_task_gt_batch=gt_batch._replace(point_cloud_gt_batch=shifted)
        )


def _batch(value: float) -> ModelGTBatch:
    """Build a one sample batch holding one point whose first feature is the given value."""
    return ModelGTBatch(
        point_cloud_gt_batch=PointCloudGTBatch(
            points=torch.full((1, 4), value, dtype=torch.float32),
            batch_indices=torch.zeros(1, dtype=torch.int32),
            batch_size=1,
            timestamp_difference_dim=-1,
        ),
        detection3d_gt_batch=None,
        segmentation3d_gt_batch=None,
        image_gt_batch=None,
    )


def _shifting_model(model_type: type[_ToyModel]) -> _ToyModel:
    """Build a toy model whose preprocessing shifts the points by one."""
    return model_type(data_preprocessor=DataPreprocessor(preprocessor_modules=[_ShiftPoints()]))


class _ToyModel(ModuleBaseModel):
    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> torch.Tensor:
        points = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        assert points is not None
        return points.points[:, 0] * 2.0

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: Any
    ) -> dict[str, Any]:
        del multi_task_batch_inputs
        return {"loss": multi_task_outputs.sum()}

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: Any
    ) -> dict[str, Any]:
        return {"prediction": outputs, "batch_size": multi_task_batch_inputs.batch_size()}


class _MissingLossModel(_ToyModel):
    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: Any
    ) -> dict[str, Any]:
        del multi_task_batch_inputs, multi_task_outputs
        return {"accuracy": torch.tensor(1.0)}


def test_on_after_batch_transfer_applies_preprocessing_pipeline() -> None:
    model = _shifting_model(_ToyModel)

    batch = model.on_after_batch_transfer(_batch(1.0), dataloader_idx=0)

    points = batch.multi_task_gt_batch.point_cloud_gt_batch
    assert points is not None
    assert torch.equal(points.points[:, 0], torch.tensor([2.0]))


def test_predict_step_runs_forward_and_formats_predictions() -> None:
    model = _shifting_model(_ToyModel)
    batch = model.on_after_batch_transfer(_batch(1.0), dataloader_idx=0)

    predictions = model.predict_step(batch, batch_idx=0)

    assert torch.equal(predictions["prediction"], torch.tensor([4.0]))
    assert predictions["batch_size"] == 1


def test_core_step_requires_loss_metric() -> None:
    model = _MissingLossModel()

    with pytest.raises(ValueError, match="'loss' key"):
        model.training_step(ModelBatchInputs.from_gt_batch(_batch(1.0)), batch_idx=0)
