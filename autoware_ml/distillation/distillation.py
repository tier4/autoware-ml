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

"""Knowledge distillation attached to a model.

The base model calls :meth:`Distillation.prepare_batch` after its runtime preprocessing, which
runs the teacher and stores its outputs on the batch as :class:`DistillationTargets`, and
:meth:`Distillation.compute` after its own losses, which adds one term per configured
:class:`DistillationTerm` and redefines ``loss`` as the ground-truth loss plus the weighted terms.
The teacher sees the very batch the student sees, augmentation included.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.models.distillation import DistillationTargets
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.distillation.taps import CaptureStore, register_taps
from autoware_ml.distillation.teacher import Teacher
from autoware_ml.distillation.term import DistillationTerm
from autoware_ml.types.dataset import SplitType

GT_LOSS_KEY = "gt_loss"
DISTILL_LOSS_KEY = "distill_loss"
TERM_KEY_PREFIX = "distill/"
DISTILLATION_STAGES = frozenset({SplitType.TRAIN.value, SplitType.VAL.value, SplitType.TEST.value})


class Distillation(nn.Module):
    """A teacher and the terms that pull the student towards it.

    The adapters of the terms are the only parameters of this module; they are saved with the
    student and trained with it. The teacher is held, not registered, see :class:`Teacher`.

    Args:
        teacher: The frozen teacher.
        terms: Terms added to the loss, each logged as ``distill/<name>``.
        stages: Splits the distillation runs at. By default train only, so ``val/loss`` stays
            the ground-truth loss the checkpoint and early-stopping callbacks monitor.

    Raises:
        ValueError: If two terms share a name, or a stage is not train, val or test.
    """

    def __init__(
        self,
        teacher: Teacher,
        terms: Sequence[DistillationTerm],
        stages: Sequence[str] = (SplitType.TRAIN.value,),
    ) -> None:
        super().__init__()
        names = [term.name for term in terms]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Distillation terms share the name(s) {duplicates}.")
        unknown = sorted(set(stages) - DISTILLATION_STAGES)
        if unknown:
            raise ValueError(
                f"Distillation stages must be among {sorted(DISTILLATION_STAGES)}, got {unknown}."
            )
        self.teacher = teacher
        self.terms = nn.ModuleList(terms)
        self.stages = frozenset(stages)
        self._student_store = CaptureStore()
        self._teacher_store = CaptureStore()
        self._hook_handles: list[RemovableHandle] = []

    def active(self, stage: str) -> bool:
        """Whether the distillation runs at the stage.

        Args:
            stage: Split name of the running step.

        Returns:
            True when the terms are added to the loss at this stage.
        """
        return stage in self.stages

    def setup(self, student: nn.Module, device: torch.device) -> None:
        """Build the teacher and hook the module taps of the terms on both models.

        Args:
            student: The model being trained.
            device: Device the teacher is built on.
        """
        teacher_model = self.teacher.build(device)
        self._hook_handles.extend(
            register_taps(student, (term.student_tap for term in self.terms), self._student_store)
        )
        self._hook_handles.extend(
            register_taps(
                teacher_model, (term.teacher_tap for term in self.terms), self._teacher_store
            )
        )

    def teardown(self) -> None:
        """Remove the hooks and drop what they captured."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._student_store.enabled = False
        self._student_store.clear()
        self._teacher_store.clear()

    def prepare_batch(
        self, batch: ModelGTBatch, batch_inputs: ModelBatchInputs
    ) -> ModelBatchInputs:
        """Run the teacher on the batch and attach its outputs to the student inputs.

        Args:
            batch: Collated batch on the device of the step.
            batch_inputs: The batch as preprocessed for the student.

        Returns:
            The student inputs carrying the teacher targets.
        """
        self._teacher_store.clear()
        self._teacher_store.enabled = True
        try:
            outputs = self.teacher.run(batch, batch_inputs)
        finally:
            self._teacher_store.enabled = False
        features = dict(self._teacher_store.tensors)
        self._teacher_store.clear()
        # The student forward comes next; its hooks capture until compute reads them.
        self._student_store.clear()
        self._student_store.enabled = True
        return batch_inputs.replace(
            distillation_targets=DistillationTargets(outputs=outputs, features=features)
        )

    def compute(
        self,
        batch_inputs: ModelBatchInputs,
        outputs: Any,
        metrics: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Add the distillation terms to the metrics of the step.

        Args:
            batch_inputs: Student inputs carrying the teacher targets.
            outputs: Outputs the student returned for the batch.
            metrics: Losses of the student against the ground truth, with ``loss``.

        Returns:
            The metrics plus ``gt_loss`` (the incoming ``loss``), ``distill/<name>`` per term,
            ``distill_loss`` (their weighted sum) and ``loss`` as the sum of both.

        Raises:
            ValueError: If the batch carries no teacher targets.
        """
        targets = batch_inputs.distillation_targets
        if targets is None:
            raise ValueError(
                "The batch carries no distillation targets; prepare_batch did not run."
            )
        self._student_store.enabled = False
        student_features = dict(self._student_store.tensors)
        self._student_store.clear()

        gt_loss = metrics["loss"]
        result: dict[str, torch.Tensor] = dict(metrics)
        result[GT_LOSS_KEY] = gt_loss
        distill_loss = torch.zeros_like(gt_loss)
        for term in self.terms:
            value = term(outputs, student_features, targets)
            result[f"{TERM_KEY_PREFIX}{term.name}"] = value
            distill_loss = distill_loss + term.weight * value
        result[DISTILL_LOSS_KEY] = distill_loss
        result["loss"] = gt_loss + distill_loss
        return result
