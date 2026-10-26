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

"""Unit tests for the stage declaration in ``stages.py`` and ``types/backend.py``."""

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
    graph_stages,
    model_output_names,
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


class _NestedHead(nn.Module):
    def forward(self, x):
        return x + 1, (x - 1, x * 2)


class _ScalarHead(nn.Module):
    def forward(self, x):
        return x + 1, 3


def _toy_stages():
    return (
        TorchStage(
            "prep", run=lambda state: {"x": state.batch["x"].float()}, inputs=(), outputs=("x",)
        ),
        GraphStage("encoder", module=_Double(), inputs=("x",), outputs=("y",)),
        TorchStage(
            "glue",
            run=lambda state: {"z": state["y"].to(state.device) + 0.5},
            inputs=("y",),
            outputs=("z",),
        ),
        GraphStage("head", module=_SplitHead(), inputs=("z",), outputs=("plus", "minus")),
    )


class TestStageDeclaration:
    def test_validate_returns_tuple_and_a_chain_outputs_its_last_graph(self):
        stages = validate_stages(_toy_stages())
        assert [s.name for s in graph_stages(stages)] == ["encoder", "head"]
        assert model_output_names(stages) == ("plus", "minus")

    def test_two_heads_over_one_encoder_both_output(self):
        stages = (
            TorchStage(
                "prep", run=lambda state: {"x": state.batch["x"].float()}, inputs=(), outputs=("x",)
            ),
            GraphStage("encoder", module=_SplitHead(), inputs=("x",), outputs=("y", "aux")),
            GraphStage("seg", module=_SplitHead(), inputs=("y",), outputs=("plus", "minus")),
            GraphStage("det", module=_Double(), inputs=("y",), outputs=("twice",)),
        )
        # ``y`` is consumed by the heads; ``aux`` is not, so it stays a model output.
        assert model_output_names(validate_stages(stages)) == ("aux", "plus", "minus", "twice")

    def test_glue_consuming_every_graph_output_leaves_no_model_outputs(self):
        post = TorchStage(
            "post",
            run=lambda state: {"final": state["plus"] + state["minus"]},
            inputs=("plus", "minus"),
            outputs=("final",),
        )
        with pytest.raises(ValueError, match="No graph output is left"):
            validate_stages((*_toy_stages(), post))

    def test_a_glue_output_the_model_hands_over_is_not_a_model_output(self):
        post = TorchStage(
            "post", run=lambda state: {"aux": state["plus"]}, inputs=("plus",), outputs=("aux",)
        )
        assert model_output_names((*_toy_stages(), post)) == ("minus",)

    def test_duplicate_names_rejected(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage("encoder", run=lambda state: {}, inputs=(), outputs=("z",))
        with pytest.raises(ValueError, match="Duplicate"):
            validate_stages(stages)

    def test_graph_stage_needs_inputs_and_outputs(self):
        with pytest.raises(ValueError, match="must declare inputs and outputs"):
            GraphStage("h", module=_Double(), inputs=(), outputs=("y",))

    def test_graph_stage_names_each_onnx_tensor_once(self):
        with pytest.raises(ValueError, match=r"\['x'\] more than once in its inputs"):
            GraphStage("sub", module=_Subtract(), inputs=("x", "x"), outputs=("y",))
        with pytest.raises(ValueError, match=r"\['y'\] more than once in its outputs"):
            GraphStage("head", module=_SplitHead(), inputs=("x",), outputs=("y", "y"))

    def test_entries_other_than_stages_rejected(self):
        with pytest.raises(TypeError, match=r"got \['dict'\]"):
            validate_stages((*_toy_stages(), {"name": "post"}))

    def test_names_are_declared_as_tuples_not_strings(self):
        with pytest.raises(TypeError, match=r"declares its outputs as the string 'x'"):
            TorchStage("seed", run=lambda state: {"x": 1}, inputs=(), outputs="x")
        with pytest.raises(TypeError, match=r"declares its inputs as the string 'xy'"):
            GraphStage("g", module=_Double(), inputs="xy", outputs=("y",))

    def test_duplicate_graph_outputs_rejected(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_SplitHead(), inputs=("z",), outputs=("plus", "y"))
        with pytest.raises(ValueError, match="'head' writes 'y', which stage 'encoder' already"):
            validate_stages(stages)

    def test_no_graph_stage_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_stages([TorchStage("only", run=lambda state: {}, inputs=(), outputs=("x",))])

    def test_opening_graph_stage_rejected(self):
        first = GraphStage("first", module=nn.Identity(), inputs=("x",), outputs=("y",))
        with pytest.raises(ValueError, match="'first' reads \\['x'\\], which no earlier stage"):
            validate_stages((first,))
        seed = TorchStage("seed", run=lambda state: {"x": 1}, inputs=(), outputs=("x",))
        assert len(validate_stages((seed, first))) == 2

    def test_glue_reading_an_unwritten_name_is_rejected_statically(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"z": state["typo"]}, inputs=("typo",), outputs=("z",)
        )
        with pytest.raises(ValueError, match="'glue' reads \\['typo'\\]"):
            validate_stages(stages)

    def test_state_names_what_it_holds_on_a_missing_tensor(self):
        state = RunState(batch=None, device=torch.device("cpu"))
        state.tensors["mid"] = torch.ones(1)
        with pytest.raises(KeyError, match="available: \\['mid'\\]"):
            state["typo"]


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
        # Arguments follow ``inputs``, not the parameter names or the glue's output order.
        stages = (
            TorchStage(
                "prep",
                run=lambda state: {"b": torch.tensor([1.0]), "a": torch.tensor([10.0])},
                inputs=(),
                outputs=("b", "a"),
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
        stages[2] = TorchStage(
            "glue", run=lambda state: state["y"] + 0.5, inputs=("y",), outputs=("z",)
        )
        with pytest.raises(TypeError, match="'glue' returned Tensor"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_glue_overwriting_graph_output_rejected(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue",
            run=lambda state: {"y": state["y"], "z": state["y"]},
            inputs=("y",),
            outputs=("y", "z"),
        )
        with pytest.raises(ValueError, match="'glue' writes 'y', which stage 'encoder'"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_glue_returning_other_names_than_declared_is_rejected(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"zz": state["y"]}, inputs=("y",), outputs=("z",)
        )
        with pytest.raises(ValueError, match="'glue' returned \\['zz'\\] but declares"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_glue_runs_on_its_declared_inputs_only(self):
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"z": state["x"]}, inputs=("y",), outputs=("z",)
        )
        with pytest.raises(KeyError, match="did not declare as an input"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_graph_stage_output_arity_is_checked(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_Double(), inputs=("z",), outputs=("plus", "minus"))
        with pytest.raises(ValueError, match="returned 1 tensor"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_graph_stage_outputs_are_tensors(self):
        for module, other in ((_NestedHead(), "tuple"), (_ScalarHead(), "int")):
            stages = list(_toy_stages())
            stages[3] = GraphStage("head", module=module, inputs=("z",), outputs=("plus", "minus"))
            with pytest.raises(TypeError, match=rf"'head' returned \['{other}'\] among"):
                run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))


def test_artifact_path_per_backend() -> None:
    assert artifact_path("/tmp", "s", Backend.ONNX) == Path("/tmp/s.onnx")
    assert artifact_path("/tmp", "s", Backend.TENSORRT) == Path("/tmp/s.engine")
    with pytest.raises(ValueError, match="no exported artifact"):
        artifact_path("/tmp", "s", Backend.PYTORCH)


def test_backend_name_formats_as_its_config_spelling() -> None:
    assert f"{Backend.TENSORRT}" == "tensorrt"
    assert Backend("onnx") is Backend.ONNX
    with pytest.raises(ValueError, match="Unknown backend 'tensorflow'"):
        Backend("tensorflow")


def test_glue_stage_must_declare_outputs() -> None:
    with pytest.raises(ValueError, match="must declare outputs"):
        TorchStage("glue", run=lambda state: {}, inputs=(), outputs=())
