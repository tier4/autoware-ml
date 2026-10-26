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

"""Export specs derived from a stage graph."""

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
    validate_stages,
)
from autoware_ml.utils.deploy import ExportSpec


def export_spec_for_stage(stage: GraphStage, state: RunState) -> ExportSpec:
    """Build one graph stage's spec from a completed run of the stage graph.

    The spec takes the stage's declared input and output names. Its ``dynamic_axes`` are the
    stage's ``onnx_dynamic_axes``, or ``None`` so that the module's config entry applies.
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
    """Run the stage graph once and return ``{stage name: ExportSpec}`` per graph stage.

    The graph modules are switched to eval first. An export wrapper starts in training
    mode, and ``torch.onnx.export`` restores the module's mode recursively when it is done:
    a wrapper left in training mode would switch the submodules it shares with the model,
    BatchNorm included, to training mode.

    Args:
        stages: The model's ``build_stages()``.
        batch: One preprocessed example batch.
        device: Device of the tracing run.
    """
    stages = validate_stages(stages)
    for stage in graph_stages(stages):
        stage.module.eval()
    state = run_stages_in_torch(stages, batch, device)
    return {stage.name: export_spec_for_stage(stage, state) for stage in graph_stages(stages)}
