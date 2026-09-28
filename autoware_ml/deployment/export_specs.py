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

"""Build the per-module export specs from a stage graph.

:func:`build_export_specs_from_stages` is what :meth:`ModuleBaseModel.build_export_specs`
returns for a model that declares ``build_stages()``: the stage graph runs once in PyTorch
on the example batch, and every :class:`~autoware_ml.deployment.stages.GraphStage` becomes
one :class:`~autoware_ml.utils.deploy.ExportSpec` whose trace inputs are the run-state
tensors it declares as inputs. :meth:`DeploymentExport.export` consumes the result exactly
like a hand-written mapping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from autoware_ml.deployment.stages import (
    GraphStage,
    RunState,
    Stage,
    graph_stages,
    run_stages_in_torch,
)
from autoware_ml.utils.deploy import ExportSpec


def export_spec_for_stage(stage: GraphStage, state: RunState) -> ExportSpec:
    """Build one graph stage's :class:`ExportSpec` from a filled run state.

    Args:
        stage: The exportable stage.
        state: Run state of a completed PyTorch run of the whole stage graph; the
            stage's ``inputs`` are read from it as the trace arguments.

    Returns:
        Spec whose ``input_param_names`` / ``output_names`` are the stage's declared
        context names and whose ``dynamic_axes`` are the stage's intrinsic axes (``None``
        when it declares none, so the module's config entry applies).
    """
    dynamic_axes = (
        {name: dict(axes) for name, axes in stage.onnx_dynamic_axes.items()}
        if stage.onnx_dynamic_axes
        else None
    )
    return ExportSpec(
        module=stage.module,
        args=state.graph_inputs[stage.name],
        input_names=list(stage.inputs),
        output_names=list(stage.outputs),
        dynamic_axes=dynamic_axes,
    )


def build_export_specs_from_stages(
    stages: Sequence[Stage], batch: Any, device: torch.device
) -> dict[str, ExportSpec]:
    """Run the stage graph once and return one export spec per graph stage, in order.

    Args:
        stages: The model's stage declaration (``model.build_stages()``).
        batch: One preprocessed example batch (what ``on_after_batch_transfer`` returns).
        device: Device the tracing run executes on.

    Returns:
        Ordered mapping ``{stage.name: ExportSpec}`` — the same shape a hand-written
        ``build_export_specs`` returns.
    """
    state = run_stages_in_torch(stages, batch, device)
    return {stage.name: export_spec_for_stage(stage, state) for stage in graph_stages(stages)}
