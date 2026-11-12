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

from types import SimpleNamespace
from typing import Any, ClassVar

import lightning as L
import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.datamodule.streaming_data_module import StreamingDataModule
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


class _PointSumModel(ModuleBaseModel):
    """Sum the points of the batch with one learned scale."""

    def __init__(self) -> None:
        super().__init__(
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        )
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

    def decode_outputs(self, multi_task_batch_inputs: ModelBatchInputs, outputs: Any) -> Any:
        del multi_task_batch_inputs
        return outputs


class _StreamingModel(_PointSumModel):
    """A model whose memory needs scene-contiguous batches."""

    requires_streaming: ClassVar[bool] = True


class _ForeignDataModule(L.LightningDataModule):
    """A Lightning datamodule from outside the framework that claims to stream."""

    streaming = True


def _without_sources(datamodule_type: type[DataModule]) -> DataModule:
    """An instance of a framework datamodule built without sources, which ``setup`` never reads."""
    return datamodule_type.__new__(datamodule_type)


def _attach_trainer(model: ModuleBaseModel, datamodule: object | None) -> None:
    """Attach a stand-in trainer carrying the given datamodule."""
    model.trainer = SimpleNamespace(datamodule=datamodule)


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
            padding_mask=torch.zeros(2, dtype=torch.bool),
        )
    )


def test_step_runs_forward_on_the_batch_and_logs_the_sample_count() -> None:
    model = _PointSumModel()
    logged: dict[str, Any] = {}
    model.log_dict = lambda values, batch_size, **kwargs: logged.update(batch_size=batch_size)

    loss = model.training_step(_batch_inputs(), batch_idx=0)

    assert float(loss) == 12.0
    assert logged["batch_size"] == 2


def test_the_logged_batch_size_leaves_out_the_padding_samples() -> None:
    """
    Input: the two-sample batch with its second sample flagged as padding.
    Expected: a padding sample only keeps the ranks in step, so the step logs one sample.
    Check: the batch size handed to the logger.
    """
    model = _PointSumModel()
    logged: dict[str, Any] = {}
    model.log_dict = lambda values, batch_size, **kwargs: logged.update(batch_size=batch_size)
    batch = _batch_inputs()
    gt_batch = batch.multi_task_gt_batch._replace(padding_mask=torch.tensor([False, True]))

    model.training_step(ModelBatchInputs.from_gt_batch(gt_batch), batch_idx=0)

    assert logged["batch_size"] == 1


def test_model_without_an_export_fails_loudly() -> None:
    with pytest.raises(NotImplementedError, match="build_export_spec"):
        _PointSumModel().build_export_specs(_batch_inputs())


def test_setup_accepts_a_streaming_datamodule_for_a_streaming_model() -> None:
    """
    Input: a streaming model set up with a StreamingDataModule.
    Expected: the StreamingDataModule serves the frames of every scene in time order to each
    batch position, which is all the memory of the model needs.
    Check: ``setup`` returns without raising.
    """
    model = _StreamingModel()
    _attach_trainer(model, _without_sources(StreamingDataModule))

    model.setup("fit")


@pytest.mark.parametrize(
    ("datamodule", "found"),
    [
        (_without_sources(DataModule), "DataModule"),
        (_ForeignDataModule(), "_ForeignDataModule"),
        (None, "no datamodule"),
    ],
    ids=["plain-datamodule", "foreign-datamodule", "bare-dataloaders"],
)
def test_setup_rejects_a_streaming_model_without_a_streaming_datamodule(
    datamodule: object | None, found: str
) -> None:
    """
    Input: a streaming model set up with the plain DataModule, with an outside Lightning
    datamodule that sets ``streaming = True``, and without a datamodule.
    Expected: ``setup`` refuses all three before any batch runs: only the StreamingDataModule
    type serves scene-contiguous frames, an attribute does not count.
    Check: a ValueError asking for a StreamingDataModule and naming what the trainer runs with.
    """
    model = _StreamingModel()
    _attach_trainer(model, datamodule)

    with pytest.raises(ValueError, match=f"use a StreamingDataModule, got {found}"):
        model.setup("fit")


def test_setup_ignores_the_datamodule_for_a_model_without_memory() -> None:
    """
    Input: a model without memory set up without a datamodule.
    Expected: only a model that sets ``requires_streaming`` needs a streaming datamodule, so the
    check is skipped.
    Check: ``setup`` returns without raising.
    """
    model = _PointSumModel()
    _attach_trainer(model, None)

    model.setup("fit")
