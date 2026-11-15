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

"""One distillation term: a student tensor, a teacher tensor and the loss between them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.distillation import DistillationTargets
from autoware_ml.distillation.taps import Tap


class DistillationTerm(nn.Module):
    """Compare one student tensor with one teacher tensor.

    Args:
        name: Name the term is logged under, ``distill/<name>``.
        student: Tap into the student, see :class:`Tap`.
        teacher: Tap into the teacher.
        loss: Loss module called as ``loss(student_tensor, teacher_tensor)``.
        weight: Factor of the term in the total loss. The logged value is unweighted.
        adapter: Trainable module applied to the student tensor before the loss, for a student
            whose channels differ from the teacher's. It trains with the student.

    Raises:
        ValueError: If the name is empty.
    """

    def __init__(
        self,
        name: str,
        student: str,
        teacher: str,
        loss: nn.Module,
        weight: float = 1.0,
        adapter: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not name:
            raise ValueError("A distillation term needs a name.")
        self.name = name
        self.student_tap = Tap(student)
        self.teacher_tap = Tap(teacher)
        self.loss = loss
        self.weight = float(weight)
        self.adapter = adapter

    def forward(
        self,
        student_outputs: Any,
        student_features: Mapping[str, torch.Tensor],
        targets: DistillationTargets,
    ) -> torch.Tensor:
        """Compute the unweighted term.

        Args:
            student_outputs: Outputs the student returned for the batch.
            student_features: Module outputs captured on the student for the batch.
            targets: Teacher outputs and captured features for the batch.

        Returns:
            The loss between the adapted student tensor and the teacher tensor.

        Raises:
            ValueError: If the two tensors differ in shape after the adapter.
        """
        student = self.student_tap.resolve(student_outputs, student_features)
        teacher = self.teacher_tap.resolve(targets.outputs, targets.features).detach()
        if self.adapter is not None:
            student = self.adapter(student)
        if student.shape != teacher.shape:
            raise ValueError(
                f"Distillation term {self.name!r} compares a student tensor of shape "
                f"{tuple(student.shape)} with a teacher tensor of shape {tuple(teacher.shape)}. "
                "Add an adapter, or feed the teacher the student inputs so both see the same "
                "voxels."
            )
        return self.loss(student, teacher)
