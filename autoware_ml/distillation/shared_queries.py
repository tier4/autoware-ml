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

"""Distill the per-query outputs of a TransFusion head through shared proposals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    TransFusionHeadOutputs,
    TransFusionSeparateHeadOutputs,
)
from autoware_ml.dataclasses.models.distillation import DistillationTargets
from autoware_ml.distillation.taps import Tap, walk_path


class TransFusionSharedQueriesTerm(nn.Module):
    """Run the student's TransFusion head at the teacher's proposals and match the queries.

    Two heads pick their own proposals, so their per-query outputs never line up. This term runs
    the student head once more, on the student's own BEV features but at the positions and
    classes the teacher chose, which pairs every query of the two models index for index. The
    class logits are matched with a binary cross entropy against the teacher's probabilities,
    and the box regressions with an L1 weighted by the teacher's confidence in each query, so
    boxes the teacher does not believe in do not pull. Both parts are returned separately and
    logged as ``distill/<name>/cls`` and ``distill/<name>/reg``.

    The dense heatmap stays a separate term: at inference the student still picks its own
    proposals, and the heatmap term is what makes it look where the teacher looks.

    Args:
        name: Name the term is logged under.
        student_features: Tap of the student's BEV features the head reads, a ``module:`` tap.
        head: Path of the student's TransFusion head within the student.
        teacher: Attribute path of the teacher's TransFusion outputs within its outputs.
        weight: Factor of the whole term in the total loss.
        classification_weight: Factor of the class part, as ``loss_cls_weight`` of the head.
        regression_weight: Factor of the box part, as ``loss_bbox_weight`` of the head.
    """

    def __init__(
        self,
        name: str = "det3d_queries",
        student_features: str = "module:bev_neck",
        head: str = "bbox_head",
        teacher: str = "detection3d_head_outputs.transfusion_head_outputs",
        weight: float = 1.0,
        classification_weight: float = 1.0,
        regression_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if not name:
            raise ValueError("A distillation term needs a name.")
        self.name = name
        self.weight = float(weight)
        self.student_tap = Tap(student_features)
        if not self.student_tap.is_module:
            raise ValueError("student_features must be a module tap, such as module:bev_neck.")
        self.teacher_tap = Tap(teacher)
        self.head_path = head
        self.classification_weight = float(classification_weight)
        self.regression_weight = float(regression_weight)
        # Held, not registered: the head belongs to the student, which already owns it.
        self._head: list[nn.Module] = []

    def bind(self, student: nn.Module) -> None:
        """Take hold of the student's head.

        Args:
            student: The model being trained.

        Raises:
            ValueError: If the student has no module at the head path, or that module cannot
                run at given proposals.
        """
        try:
            head = student.get_submodule(self.head_path)
        except AttributeError as error:
            raise ValueError(
                f"Shared-queries term {self.name!r}: {type(student).__name__} has no submodule "
                f"{self.head_path!r}."
            ) from error
        if not callable(getattr(head, "forward_at_proposals", None)):
            raise ValueError(
                f"Shared-queries term {self.name!r}: {type(head).__name__} cannot run at given "
                "proposals; it needs a forward_at_proposals method."
            )
        self._head = [head]

    def forward(
        self,
        student_outputs: Any,
        student_features: Mapping[str, torch.Tensor],
        targets: DistillationTargets,
    ) -> dict[str, torch.Tensor]:
        """Compute the class and the box part of the term.

        Args:
            student_outputs: Outputs the student returned for the batch, unused: the head runs
                again at the teacher's proposals.
            student_features: Module outputs captured on the student, holding the BEV features.
            targets: Teacher outputs for the batch, with the proposal positions.

        Returns:
            ``cls`` and ``reg``, each already multiplied by its weight.

        Raises:
            RuntimeError: If the term was not bound to the student.
            ValueError: If the teacher outputs carry no TransFusion outputs with proposal
                positions, or the two heads produce proposals of different counts.
        """
        if not self._head:
            raise RuntimeError("The shared-queries term is bound to the student at fit setup.")
        bev_features = self.student_tap.resolve(student_outputs, student_features)
        teacher = walk_path(targets.outputs, self.teacher_tap.output_path or "")
        if not isinstance(teacher, TransFusionHeadOutputs):
            raise ValueError(
                f"Shared-queries term {self.name!r}: the teacher tap leads to "
                f"{type(teacher).__name__}, not TransFusionHeadOutputs."
            )
        if teacher.query_positions is None:
            raise ValueError(
                f"Shared-queries term {self.name!r}: the teacher outputs carry no query positions."
            )
        student = self._head[0].forward_at_proposals(
            bev_features, teacher.query_positions, teacher.query_labels
        )
        student_branches = student.separate_head_outputs
        teacher_branches = teacher.separate_head_outputs
        if student_branches.heatmap.shape != teacher_branches.heatmap.shape:
            raise ValueError(
                f"Shared-queries term {self.name!r}: the student predicts "
                f"{tuple(student_branches.heatmap.shape)} per-query logits, the teacher "
                f"{tuple(teacher_branches.heatmap.shape)}. Both heads need the same proposal "
                "count and decoder layers."
            )
        teacher_probs = torch.sigmoid(teacher_branches.heatmap.float().detach())
        classification = F.binary_cross_entropy_with_logits(
            student_branches.heatmap.float(), teacher_probs
        )
        student_boxes = _box_regressions(student_branches)
        teacher_boxes = _box_regressions(teacher_branches).detach()
        # (batch_size, num_queries): how much the teacher believes in each query.
        query_weights = teacher_probs.amax(dim=1)
        l1 = (student_boxes - teacher_boxes).abs().sum(dim=1)
        regression = (l1 * query_weights).sum() / query_weights.sum().clamp(min=1.0)
        return {
            "cls": self.classification_weight * classification,
            "reg": self.regression_weight * regression,
        }


def _box_regressions(branches: TransFusionSeparateHeadOutputs) -> torch.Tensor:
    """Stack the box regressions of every query into one tensor.

    Args:
        branches: Per-query outputs of a head.

    Returns:
        Tensor of shape ``(batch_size, box_channels, num_queries)`` in float32.
    """
    parts = [branches.center, branches.height, branches.dim, branches.rot]
    if branches.vel is not None:
        parts.append(branches.vel)
    return torch.cat([part.float() for part in parts], dim=1)
