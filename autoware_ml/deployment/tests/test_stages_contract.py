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

"""Stage-declaration contract: ``stages.py`` and ``types/backend.py`` alone, on toy stages.

Everything here is checkable with no export, pipeline, or backend module — it is the
invariant every later deployment consumer builds on, so it is tested where it is declared.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from autoware_ml.deployment.stages import (
    GraphStage,
    RunState,
    TorchStage,
    artifact_path,
    final_stage,
    graph_stages,
    run_stages_in_torch,
    validate_stages,
)
from autoware_ml.types.backend import Backend


class _Double(nn.Module):
    def forward(self, x):
        return x * 2


class _SplitHead(nn.Module):
    def forward(self, x):
        return x + 1, x - 1


def _toy_stages():
    return (
        TorchStage("prep", run=lambda ctx: {"x": ctx.batch["x"].float()}),
        GraphStage("encoder", module=_Double(), inputs=("x",), outputs=("y",)),
        TorchStage("glue", run=lambda ctx: {"z": ctx["y"].to(ctx.device) + 0.5}),
        GraphStage(
            "head",
            module=_SplitHead(),
            inputs=("z",),
            outputs=("plus", "minus"),
            output_fields=(("plus", "a"), ("minus", "b")),
        ),
    )


class TestStageDeclaration:
    def test_validate_returns_tuple_and_final_stage(self):
        stages = validate_stages(_toy_stages())
        assert [s.name for s in graph_stages(stages)] == ["encoder", "head"]
        assert final_stage(stages).name == "head"

    def test_duplicate_names_rejected(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage("encoder", run=lambda ctx: {})
        with pytest.raises(ValueError, match="Duplicate"):
            validate_stages(stages)

    def test_final_stage_needs_output_fields(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage(
            "head", module=_SplitHead(), inputs=("z",), outputs=("plus", "minus")
        )
        with pytest.raises(ValueError, match="output_fields"):
            validate_stages(stages)

    def test_only_final_stage_may_declare_output_fields(self):
        stages = list(_toy_stages())
        stages[1] = GraphStage(
            "encoder", module=_Double(), inputs=("x",), outputs=("y",), output_fields=(("y", "f"),)
        )
        with pytest.raises(ValueError, match="Only the final"):
            validate_stages(stages)

    def test_output_fields_must_name_declared_outputs(self):
        with pytest.raises(ValueError, match="not among its outputs"):
            GraphStage(
                "h", module=_Double(), inputs=("x",), outputs=("y",), output_fields=(("q", "f"),)
            )

    def test_no_graph_stage_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_stages([TorchStage("only", run=lambda ctx: {})])

    def test_opening_graph_stage_rejected(self):
        first = GraphStage(
            "first",
            module=nn.Identity(),
            inputs=("x",),
            outputs=("y",),
            output_fields=(("y", "y"),),
        )
        with pytest.raises(ValueError, match="run state starts empty"):
            validate_stages((first,))
        assert len(validate_stages((TorchStage("glue", run=lambda ctx: {}), first))) == 2

    def test_context_names_what_it_holds_on_a_missing_tensor(self):
        state = RunState(batch=None, device=torch.device("cpu"))
        state.tensors["mid"] = torch.ones(1)
        with pytest.raises(KeyError, match="available: \\['mid'\\]"):
            state["typo"]

    def test_artifact_paths_derive_from_stage_name(self, tmp_path):
        assert artifact_path(tmp_path, "encoder", Backend.ONNX) == tmp_path / "encoder.onnx"
        assert artifact_path(tmp_path, "encoder", "tensorrt") == tmp_path / "encoder.engine"
        with pytest.raises(ValueError):
            artifact_path(tmp_path, "encoder", Backend.PYTORCH)


class TestRunStagesInTorch:
    def test_runs_all_stages_in_order(self):
        state = run_stages_in_torch(
            _toy_stages(), {"x": torch.tensor([1, 2, 3])}, torch.device("cpu")
        )
        # prep: x=[1,2,3]; encoder: y=2x; glue: z=y+0.5; head: plus=z+1, minus=z-1
        assert torch.equal(state["plus"], torch.tensor([3.5, 5.5, 7.5]))
        assert torch.equal(state["minus"], torch.tensor([1.5, 3.5, 5.5]))
        assert set(state.tensors) == {"x", "y", "z", "plus", "minus"}

    def test_graph_stage_output_arity_is_checked(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage(
            "head",
            module=_Double(),
            inputs=("z",),
            outputs=("plus", "minus"),
            output_fields=(("plus", "a"),),
        )
        with pytest.raises(ValueError, match="returned 1 tensor"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))


def test_backend_and_artifact_path_agree_that_pytorch_has_no_artifact() -> None:
    with pytest.raises(ValueError, match="no exported artifact"):
        _ = Backend.PYTORCH.artifact_suffix
    with pytest.raises(ValueError, match="no exported artifact"):
        artifact_path("/tmp", "s", Backend.PYTORCH)
    assert Backend.ONNX.artifact_suffix == ".onnx"
    with pytest.raises(ValueError, match="Unknown backend"):
        Backend.parse("tflite")
