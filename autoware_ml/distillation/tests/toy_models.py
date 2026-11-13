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

"""Small models and run layouts the distillation tests build on."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.distillation.teacher import Teacher, _resolve_autocast_dtype
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.mlflow_helpers import CONFIG_ARTIFACTS_DIRNAME, RESOLVED_CONFIG_FILENAME

NUM_CLASSES = 3
IN_CHANNELS = 4


class ToySegModel(ModuleBaseModel):
    """Classify every point through one hidden layer, which a module tap can read."""

    def __init__(
        self,
        hidden_channels: int = 6,
        data_preprocessor: DataPreprocessor | None = None,
        log_dict_configs: LogDictConfigs | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            log_dict_configs=log_dict_configs or LogDictConfigs(prog_bar=False),
            data_preprocessor=data_preprocessor or DataPreprocessor(preprocessor_modules=[]),
            **kwargs,
        )
        self.hidden = nn.Linear(IN_CHANNELS, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, NUM_CLASSES)

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        point_batch = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        if point_batch is None:
            raise ValueError("The toy model needs points.")
        logits = self.classifier(torch.relu(self.hidden(point_batch.points)))
        return ModelOutputs(segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=logits))

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: ModelOutputs
    ) -> dict[str, torch.Tensor]:
        label_batch = multi_task_batch_inputs.multi_task_gt_batch.segmentation3d_gt_batch
        if label_batch is None:
            raise ValueError("The toy model needs point labels.")
        loss = F.cross_entropy(
            multi_task_outputs.segmentation3d().logits, label_batch.gt_semantic_masks
        )
        return {"loss": loss, "loss_ce": loss}

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        del multi_task_batch_inputs, outputs
        return ModelPredictions()


class ToySegModelWithGroups(ToySegModel):
    """Group the own parameters structurally, the way the joint PTv3 model does."""

    def build_optimizer_groups(self) -> Mapping[str, Sequence[nn.Parameter]]:
        return {
            "hidden": list(self.hidden.parameters()),
            "classifier": list(self.classifier.parameters()),
        }


def build_gt_batch(num_points: int = 12, batch_size: int = 2, seed: int = 0) -> ModelGTBatch:
    """Build a labelled point cloud batch with ``num_points`` points per sample."""
    generator = torch.Generator().manual_seed(seed)
    total = num_points * batch_size
    points = torch.rand((total, IN_CHANNELS), generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (total,), generator=generator)
    batch_indices = torch.arange(total, dtype=torch.int32) // num_points
    return ModelGTBatch(
        point_cloud_gt_batch=PointCloudGTBatch(
            points=points,
            batch_indices=batch_indices,
            batch_size=batch_size,
            timestamp_difference_dim=-1,
        ),
        detection3d_gt_batch=None,
        segmentation3d_gt_batch=Segmentation3DGTBatch(
            gt_semantic_masks=labels, batch_indices=batch_indices
        ),
        image_gt_batch=None,
        padding_mask=torch.zeros(batch_size, dtype=torch.bool),
    )


def toy_model_config(hidden_channels: int, **extra: Any) -> dict[str, Any]:
    """Model node of a resolved config instantiating :class:`ToySegModel`."""
    return {
        "_target_": "autoware_ml.distillation.tests.toy_models.ToySegModel",
        "_convert_": "all",
        "hidden_channels": hidden_channels,
        **extra,
    }


def write_teacher_run(run_dir: Path, model: nn.Module, model_config: Mapping[str, Any]) -> Path:
    """Lay out a run the way training does and return its checkpoint.

    Args:
        run_dir: Directory of the run.
        model: Model whose weights the checkpoint holds.
        model_config: Model node of the resolved config.

    Returns:
        Path of ``artifacts/checkpoints/best.ckpt``, with ``artifacts/config/resolved.yaml``
        beside it.
    """
    artifacts = run_dir / "artifacts"
    config_dir = artifacts / CONFIG_ARTIFACTS_DIRNAME
    checkpoints_dir = artifacts / "checkpoints"
    config_dir.mkdir(parents=True)
    checkpoints_dir.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": dict(model_config),
                "data_preprocessor": {
                    "_target_": "autoware_ml.preprocessing.data_preprocessor.DataPreprocessor",
                    "preprocessor_modules": [],
                },
            }
        ),
        config_dir / RESOLVED_CONFIG_FILENAME,
    )
    checkpoint_path = checkpoints_dir / "best.ckpt"
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path


class InMemoryTeacher(Teacher):
    """A teacher built from a model in memory, for models that have no config of their own."""

    def __init__(
        self,
        model: ModuleBaseModel,
        autocast_dtype: str | None = None,
        share_student_inputs: bool = True,
    ) -> None:
        # Deliberately skips Teacher.__init__, which reads a checkpoint and its config.
        self._source = model
        self._model = None
        self.weight_paths = []
        self.config_path = None
        self.autocast_dtype = _resolve_autocast_dtype(autocast_dtype)
        self.share_student_inputs = share_student_inputs

    def build(self, device: torch.device) -> ModuleBaseModel:
        model = self._source.to(device)
        model.requires_grad_(False)
        model.eval()
        self._model = model
        return model
