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

from pathlib import Path

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


class _DictHead(nn.Module):
    def forward(self, x):
        return {"plus": x + 1, "minus": x - 1}


class _Subtract(nn.Module):
    def forward(self, a, b):
        return a - b


def _toy_stages():
    return (
        TorchStage("prep", run=lambda state: {"x": state.batch["x"].float()}),
        GraphStage("encoder", module=_Double(), inputs=("x",), outputs=("y",)),
        TorchStage("glue", run=lambda state: {"z": state["y"].to(state.device) + 0.5}),
        GraphStage("head", module=_SplitHead(), inputs=("z",), outputs=("plus", "minus")),
    )


class TestStageDeclaration:
    def test_validate_returns_tuple_and_final_stage(self):
        stages = validate_stages(_toy_stages())
        assert [s.name for s in graph_stages(stages)] == ["encoder", "head"]
        assert final_stage(stages).name == "head"

    def test_duplicate_names_rejected(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage("encoder", run=lambda state: {})
        with pytest.raises(ValueError, match="Duplicate"):
            validate_stages(stages)

    def test_graph_stage_needs_inputs_and_outputs(self):
        with pytest.raises(ValueError, match="must declare inputs and outputs"):
            GraphStage("h", module=_Double(), inputs=(), outputs=("y",))

    def test_duplicate_graph_outputs_rejected(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_SplitHead(), inputs=("z",), outputs=("plus", "y"))
        with pytest.raises(ValueError, match="declared more than once: \\['y'\\]"):
            validate_stages(stages)

    def test_no_graph_stage_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_stages([TorchStage("only", run=lambda state: {})])

    def test_opening_graph_stage_rejected(self):
        first = GraphStage("first", module=nn.Identity(), inputs=("x",), outputs=("y",))
        with pytest.raises(ValueError, match="run state starts empty"):
            validate_stages((first,))
        assert len(validate_stages((TorchStage("glue", run=lambda state: {}), first))) == 2

    def test_state_names_what_it_holds_on_a_missing_tensor(self):
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

    def test_multi_input_graph_stage_positional_order(self):
        # The module's positional arguments follow ``inputs``, not the module's parameter
        # names or the order the glue produced them in (BEVFusion's lidar graph relies on it).
        stages = (
            TorchStage(
                "prep", run=lambda state: {"b": torch.tensor([1.0]), "a": torch.tensor([10.0])}
            ),
            GraphStage("ab", module=_Subtract(), inputs=("a", "b"), outputs=("a_minus_b",)),
            GraphStage("ba", module=_Subtract(), inputs=("b", "a"), outputs=("b_minus_a",)),
        )
        state = run_stages_in_torch(stages, None, torch.device("cpu"))
        assert torch.equal(state["a_minus_b"], torch.tensor([9.0]))
        assert torch.equal(state["b_minus_a"], torch.tensor([-9.0]))

    def test_graph_stage_mapping_return_is_rejected(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_DictHead(), inputs=("z",), outputs=("plus", "minus"))
        with pytest.raises(TypeError, match="'head' returned a mapping"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_torch_stage_non_mapping_rejected(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage("glue", run=lambda state: state["y"] + 0.5)
        with pytest.raises(TypeError, match="'glue' returned Tensor"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_glue_overwriting_graph_output_rejected(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage("glue", run=lambda state: {"y": state["y"], "z": state["y"]})
        with pytest.raises(ValueError, match="'glue' writes \\['y'\\]"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_graph_stage_output_arity_is_checked(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_Double(), inputs=("z",), outputs=("plus", "minus"))
        with pytest.raises(ValueError, match="returned 1 tensor"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))


def test_artifact_path_per_backend() -> None:
    assert artifact_path("/tmp", "s", Backend.ONNX) == Path("/tmp/s.onnx")
    assert artifact_path("/tmp", "s", Backend.TENSORRT) == Path("/tmp/s.engine")
    with pytest.raises(ValueError, match="no exported artifact"):
        artifact_path("/tmp", "s", Backend.PYTORCH)


def test_backend_name_formats_as_its_config_spelling() -> None:
    assert f"{Backend.TENSORRT}" == "tensorrt"
    assert Backend("onnx") is Backend.ONNX
    with pytest.raises(ValueError, match="Unknown backend 'tflite'"):
        Backend("tflite")
