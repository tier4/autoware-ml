"""Unit tests for the precise BatchNorm callback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from autoware_ml.callbacks.precise_batchnorm import PreciseBatchNorm
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.distillation.tests.toy_models import IN_CHANNELS, NUM_CLASSES, build_gt_batch
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


class _NormalizedModel(ModuleBaseModel):
    """Classify every point after a BatchNorm over the point features."""

    def __init__(self) -> None:
        super().__init__(
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        )
        self.norm = nn.BatchNorm1d(IN_CHANNELS, momentum=0.01)
        self.classifier = nn.Linear(IN_CHANNELS, NUM_CLASSES)

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        points = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch.points
        return ModelOutputs(
            segmentation3d_head_outputs=Segmentation3DHeadOutputs(
                logits=self.classifier(self.norm(points))
            )
        )

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: Any
    ) -> dict:
        return {"loss": multi_task_outputs.segmentation3d().logits.sum()}

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: Any
    ) -> ModelPredictions:
        return ModelPredictions()


def _trainer(batches: list, sanity_checking: bool = False) -> SimpleNamespace:
    datamodule = SimpleNamespace(train_dataloader=lambda: iter(batches))
    return SimpleNamespace(datamodule=datamodule, sanity_checking=sanity_checking)


def test_recalibration_averages_the_statistics_over_the_batches_seen() -> None:
    model = _NormalizedModel()
    batches = [build_gt_batch(seed=seed) for seed in range(4)]
    # BatchNorm with momentum None averages the per-batch mean and unbiased variance.
    seen = [b.point_cloud_gt_batch.points for b in batches[:3]]
    expected_mean = torch.stack([p.mean(dim=0) for p in seen]).mean(dim=0)
    expected_var = torch.stack([p.var(dim=0, unbiased=True) for p in seen]).mean(dim=0)
    model.norm.running_mean.fill_(5.0)
    model.eval()

    PreciseBatchNorm(num_batches=3).on_validation_start(_trainer(batches), model)

    assert torch.allclose(model.norm.running_mean, expected_mean, atol=1e-5)
    assert torch.allclose(model.norm.running_var, expected_var, atol=1e-5)
    assert model.norm.momentum == 0.01
    assert not model.training and not model.norm.training


def test_the_sanity_check_is_left_alone() -> None:
    model = _NormalizedModel()
    model.norm.running_mean.fill_(5.0)

    PreciseBatchNorm().on_validation_start(
        _trainer([build_gt_batch()], sanity_checking=True), model
    )

    assert torch.all(model.norm.running_mean == 5.0)


def test_test_start_recalibrates_too() -> None:
    model = _NormalizedModel()
    model.norm.running_mean.fill_(5.0)

    PreciseBatchNorm(num_batches=1).on_test_start(_trainer([build_gt_batch()]), model)

    assert not torch.any(model.norm.running_mean == 5.0)


def test_a_module_without_batchnorm_needs_no_datamodule() -> None:
    model = _NormalizedModel()
    model.norm = nn.Identity()

    PreciseBatchNorm().on_validation_start(
        SimpleNamespace(datamodule=None, sanity_checking=False), model
    )


def test_a_missing_datamodule_is_a_misconfiguration() -> None:
    with pytest.raises(MisconfigurationException, match="datamodule"):
        PreciseBatchNorm().on_validation_start(
            SimpleNamespace(datamodule=None, sanity_checking=False), _NormalizedModel()
        )


def test_num_batches_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        PreciseBatchNorm(num_batches=0)
