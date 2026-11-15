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

"""A frozen, trained model whose outputs supervise a student."""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf, open_dict
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.utils.mlflow_helpers import find_run_config

if TYPE_CHECKING:
    from autoware_ml.models.module_base_model import ModuleBaseModel

logger = logging.getLogger(__name__)

# Training-only arguments of the teacher's model config. The teacher neither trains nor scores,
# and a teacher that was itself distilled must not rebuild its own teacher.
TEACHER_DROPPED_MODEL_KEYS = (
    "metrics",
    "optimizer",
    "scheduler",
    "optimizer_group_overrides",
    "scheduler_config",
    "distillation",
)


class Teacher:
    """A trained model, frozen, that runs on every batch the student trains on.

    The model is rebuilt from the resolved config a training run writes next to its checkpoints
    (``config/resolved.yaml``), so a checkpoint path is all a teacher needs. It is built at fit
    setup and is deliberately not a submodule of the student: it stays out of the student's state
    dict, optimizer groups, distributed wrapping and model summary, and never leaves eval mode.

    Args:
        weights: Checkpoints loaded into the teacher in order, later ones overriding earlier
            ones, as ``--weights`` does. Every teacher parameter must be covered. ``None``
            leaves the teacher unusable: the experiment config composes and its model
            instantiates, and the run fails at fit setup until ``teacher_weights`` names a
            checkpoint.
        config: Resolved config of the teacher. Found next to the last checkpoint when omitted.
        autocast_dtype: Autocast dtype of the teacher forward, ``bfloat16`` or ``float16``,
            None for the full precision of the parameters.
        share_student_inputs: Feed the teacher the preprocessed inputs of the student, so both
            see the same voxels and their outputs align index for index. Off, the teacher
            preprocesses the batch with the preprocessor of its own config.

    Raises:
        FileNotFoundError: If a checkpoint does not exist, or no config is found for it.
        ValueError: If the autocast dtype is not a floating dtype.
    """

    def __init__(
        self,
        weights: Sequence[str | Path] | None,
        config: str | Path | None = None,
        autocast_dtype: str | None = "bfloat16",
        share_student_inputs: bool = True,
    ) -> None:
        if isinstance(weights, (str, Path)):
            weights = [weights]
        self.weight_paths = [Path(path) for path in weights or ()]
        for path in self.weight_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Teacher checkpoint '{path}' does not exist.")
        self.config_path = Path(config) if config is not None else None
        if self.config_path is None and self.weight_paths:
            self.config_path = find_run_config(self.weight_paths[-1])
            if self.config_path is None:
                raise FileNotFoundError(
                    f"No resolved config found next to '{self.weight_paths[-1]}'. A run logged "
                    "to MLflow writes one under its artifacts; pass `config` for any other "
                    "checkpoint."
                )
        if self.config_path is not None and not self.config_path.is_file():
            raise FileNotFoundError(f"Teacher config '{self.config_path}' does not exist.")
        self.autocast_dtype = _resolve_autocast_dtype(autocast_dtype)
        self.share_student_inputs = bool(share_student_inputs)
        self._model: ModuleBaseModel | None = None

    @property
    def model(self) -> ModuleBaseModel:
        """The built teacher model.

        Raises:
            RuntimeError: If :meth:`build` has not run.
        """
        if self._model is None:
            raise RuntimeError("The teacher is built at fit setup; it has not been built yet.")
        return self._model

    def build(self, device: torch.device) -> ModuleBaseModel:
        """Instantiate the teacher from its config, load its weights and freeze it.

        Args:
            device: Device the teacher is placed on. It follows the batches afterwards.

        Returns:
            The frozen teacher model in eval mode.

        Raises:
            ValueError: If the teacher was given no checkpoint.
        """
        if not self.weight_paths or self.config_path is None:
            raise ValueError(
                "The teacher has no checkpoint. Pass teacher_weights=[<run>/artifacts/checkpoints/"
                "best.ckpt] (repeatable) when training the distilled experiment."
            )
        # Deferred: the model builder imports the base model, which imports the distillation.
        from autoware_ml.builders.model_builder import build_data_preprocessor, build_model

        cfg = OmegaConf.load(self.config_path)
        with open_dict(cfg.model):
            for key in TEACHER_DROPPED_MODEL_KEYS:
                cfg.model.pop(key, None)
        logger.info("Building the teacher from '%s'.", self.config_path)
        model = build_model(
            cfg,
            data_preprocessor=build_data_preprocessor(cfg),
            device=device,
            weights_path=self.weight_paths,
            resume_checkpoint_path=None,
            set_eval=True,
            enforce_full_coverage=True,
        )
        model.requires_grad_(False)
        model.eval()
        self._model = model
        return model

    def run(self, batch: ModelGTBatch, student_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the teacher on the batch of the student.

        Args:
            batch: Collated batch on the device of the step.
            student_inputs: The batch as preprocessed for the student.

        Returns:
            The raw teacher outputs, without gradients.
        """
        model = self.model
        device = batch.padding_mask.device
        if model.device != device:
            model.to(device)
        if self.share_student_inputs:
            inputs = student_inputs
        else:
            inputs = model._data_preprocessor(batch, is_training=False)
        autocast = (
            torch.autocast(device_type=device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast:
            return model(inputs)


def _resolve_autocast_dtype(name: str | None) -> torch.dtype | None:
    """Turn the configured dtype name into a torch dtype.

    Args:
        name: ``bfloat16``, ``float16``, or None.

    Returns:
        The dtype, or None for no autocast.

    Raises:
        ValueError: If the name is not a floating point torch dtype.
    """
    if name is None:
        return None
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"autocast_dtype must name a floating point torch dtype, got {name!r}.")
    return dtype
