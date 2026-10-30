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
import unittest

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


class _DoubleInPlace(nn.Module):
    def forward(self, x):
        return x.mul_(2)


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


class TestStageDeclaration(unittest.TestCase):
    """Unit tests for the stage declaration and its static dataflow check."""

    def test_validate_returns_tuple_and_a_chain_outputs_its_last_graph(self) -> None:
        """Test that a chain's model outputs are its last graph's outputs."""
        stages = validate_stages(_toy_stages())
        self.assertEqual([s.name for s in graph_stages(stages)], ["encoder", "head"])
        self.assertEqual(model_output_names(stages), ("plus", "minus"))

    def test_two_heads_over_one_encoder_both_output(self) -> None:
        """Test that every head over one encoder outputs, with the encoder's unread output."""
        stages = (
            TorchStage(
                "prep", run=lambda state: {"x": state.batch["x"].float()}, inputs=(), outputs=("x",)
            ),
            GraphStage("encoder", module=_SplitHead(), inputs=("x",), outputs=("y", "aux")),
            GraphStage("seg", module=_SplitHead(), inputs=("y",), outputs=("plus", "minus")),
            GraphStage("det", module=_Double(), inputs=("y",), outputs=("twice",)),
        )
        # ``y`` is consumed by the heads; ``aux`` is not, so it stays a model output.
        self.assertEqual(
            model_output_names(validate_stages(stages)), ("aux", "plus", "minus", "twice")
        )

    def test_glue_consuming_every_graph_output_leaves_no_model_outputs(self) -> None:
        """Test that a declaration whose glue consumes every graph output is rejected."""
        post = TorchStage(
            "post",
            run=lambda state: {"final": state["plus"] + state["minus"]},
            inputs=("plus", "minus"),
            outputs=("final",),
        )
        with self.assertRaisesRegex(ValueError, "No graph output is left"):
            validate_stages((*_toy_stages(), post))

    def test_a_glue_output_the_model_hands_over_is_not_a_model_output(self) -> None:
        """Test that a graph output read by later glue is not a model output."""
        post = TorchStage(
            "post", run=lambda state: {"aux": state["plus"]}, inputs=("plus",), outputs=("aux",)
        )
        self.assertEqual(model_output_names((*_toy_stages(), post)), ("minus",))

    def test_duplicate_names_rejected(self) -> None:
        """Test that two stages with the same name are rejected."""
        stages = list(_toy_stages())
        stages[2] = TorchStage("encoder", run=lambda state: {}, inputs=(), outputs=("z",))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_stages(stages)

    def test_graph_stage_needs_inputs_and_outputs(self) -> None:
        """Test that a graph stage without inputs is rejected."""
        with self.assertRaisesRegex(ValueError, "must declare inputs and outputs"):
            GraphStage("h", module=_Double(), inputs=(), outputs=("y",))

    def test_glue_stage_must_declare_outputs(self) -> None:
        """Test that a glue stage without outputs is rejected."""
        with self.assertRaisesRegex(ValueError, "must declare outputs"):
            TorchStage("glue", run=lambda state: {}, inputs=(), outputs=())

    def test_graph_stage_names_each_onnx_tensor_once(self) -> None:
        """Test that a graph stage listing a tensor name twice is rejected."""
        with self.assertRaisesRegex(ValueError, r"\['x'\] more than once in its inputs"):
            GraphStage("sub", module=_Subtract(), inputs=("x", "x"), outputs=("y",))
        with self.assertRaisesRegex(ValueError, r"\['y'\] more than once in its outputs"):
            GraphStage("head", module=_SplitHead(), inputs=("x",), outputs=("y", "y"))

    def test_dynamic_axes_name_the_stage_tensors_only(self) -> None:
        """Test that dynamic axes of a tensor the stage does not have are rejected."""
        with self.assertRaisesRegex(ValueError, r"onnx_dynamic_axes for \['xs'\], which are not"):
            GraphStage(
                "g",
                module=_Double(),
                inputs=("x",),
                outputs=("y",),
                onnx_dynamic_axes={"xs": {0: "n"}, "y": {0: "n"}},
            )

    def test_entries_other_than_stages_rejected(self) -> None:
        """Test that an entry that is not a stage is rejected."""
        with self.assertRaisesRegex(TypeError, r"got \['dict'\]"):
            validate_stages((*_toy_stages(), {"name": "post"}))

    def test_names_are_declared_as_tuples_not_strings(self) -> None:
        """Test that names declared as a bare string are rejected."""
        with self.assertRaisesRegex(TypeError, r"declares its outputs as the string 'x'"):
            TorchStage("seed", run=lambda state: {"x": 1}, inputs=(), outputs="x")
        with self.assertRaisesRegex(TypeError, r"declares its inputs as the string 'xy'"):
            GraphStage("g", module=_Double(), inputs="xy", outputs=("y",))

    def test_duplicate_graph_outputs_rejected(self) -> None:
        """Test that two stages writing the same name are rejected."""
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_SplitHead(), inputs=("z",), outputs=("plus", "y"))
        with self.assertRaisesRegex(ValueError, "'head' writes 'y', which stage 'encoder' already"):
            validate_stages(stages)

    def test_no_graph_stage_rejected(self) -> None:
        """Test that a declaration without a graph stage is rejected."""
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_stages([TorchStage("only", run=lambda state: {}, inputs=(), outputs=("x",))])

    def test_opening_graph_stage_rejected(self) -> None:
        """Test that a graph stage needs glue before it to write its inputs."""
        first = GraphStage("first", module=nn.Identity(), inputs=("x",), outputs=("y",))
        with self.assertRaisesRegex(ValueError, r"'first' reads \['x'\], which no earlier stage"):
            validate_stages((first,))
        seed = TorchStage("seed", run=lambda state: {"x": 1}, inputs=(), outputs=("x",))
        self.assertEqual(len(validate_stages((seed, first))), 2)

    def test_glue_reading_an_unwritten_name_is_rejected_statically(self) -> None:
        """Test that glue reading a name no earlier stage writes is rejected before running."""
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"z": state["typo"]}, inputs=("typo",), outputs=("z",)
        )
        with self.assertRaisesRegex(ValueError, r"'glue' reads \['typo'\]"):
            validate_stages(stages)

    def test_state_names_what_it_holds_on_a_missing_tensor(self) -> None:
        """Test that a missing tensor's error lists the tensors the run state holds."""
        state = RunState(batch=None, device=torch.device("cpu"))
        state.tensors["mid"] = torch.ones(1)
        with self.assertRaisesRegex(KeyError, r"available: \['mid'\]"):
            state["typo"]
        with self.assertRaisesRegex(KeyError, r"available: \['mid'\]"):
            state.narrowed(("typo",))

    def test_state_refuses_to_overwrite_a_name(self) -> None:
        """Test that the run state refuses a second write of a name and keeps the first."""
        state = RunState(batch=None, device=torch.device("cpu"))
        state.add({"x": torch.ones(1)})
        with self.assertRaisesRegex(ValueError, r"already holds \['x'\]"):
            state.add({"x": torch.zeros(1), "y": torch.zeros(1)})
        self.assertEqual(state["x"].item(), 1.0)


class TestRunStagesInTorch(unittest.TestCase):
    """Unit tests for running a stage graph in PyTorch and its runtime checks."""

    def test_runs_all_stages_in_order(self) -> None:
        """Test that the stages run in order and fill the run state."""
        state = run_stages_in_torch(
            _toy_stages(), {"x": torch.tensor([1, 2, 3])}, torch.device("cpu")
        )
        # prep: x=[1,2,3]; encoder: y=2x; glue: z=y+0.5; head: plus=z+1, minus=z-1
        self.assertTrue(torch.equal(state["plus"], torch.tensor([3.5, 5.5, 7.5])))
        self.assertTrue(torch.equal(state["minus"], torch.tensor([1.5, 3.5, 5.5])))
        self.assertEqual(set(state.tensors), {"x", "y", "z", "plus", "minus"})

    def test_multi_input_graph_stage_positional_order(self) -> None:
        """Test that a graph stage's arguments follow its declared ``inputs`` order."""
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
        self.assertTrue(torch.equal(state["a_minus_b"], torch.tensor([9.0])))
        self.assertTrue(torch.equal(state["b_minus_a"], torch.tensor([-9.0])))

    def test_graph_stage_mapping_return_is_rejected(self) -> None:
        """Test that a graph stage returning a mapping is rejected."""
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_DictHead(), inputs=("z",), outputs=("plus", "minus"))
        with self.assertRaisesRegex(TypeError, "'head' returned a mapping"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_torch_stage_non_mapping_rejected(self) -> None:
        """Test that a glue stage returning something other than a mapping is rejected."""
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: state["y"] + 0.5, inputs=("y",), outputs=("z",)
        )
        with self.assertRaisesRegex(TypeError, "'glue' returned Tensor"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_glue_overwriting_graph_output_rejected(self) -> None:
        """Test that glue writing a graph output's name is rejected."""
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue",
            run=lambda state: {"y": state["y"], "z": state["y"]},
            inputs=("y",),
            outputs=("y", "z"),
        )
        with self.assertRaisesRegex(ValueError, "'glue' writes 'y', which stage 'encoder'"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_glue_returning_other_names_than_declared_is_rejected(self) -> None:
        """Test that glue returning names other than its declared outputs is rejected."""
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"zz": state["y"]}, inputs=("y",), outputs=("z",)
        )
        with self.assertRaisesRegex(ValueError, r"'glue' returned \['zz'\] but declares"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_glue_runs_on_its_declared_inputs_only(self) -> None:
        """Test that glue reading a name it did not declare fails."""
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"z": state["x"]}, inputs=("y",), outputs=("z",)
        )
        with self.assertRaisesRegex(KeyError, "did not declare as an input"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_graph_stage_output_arity_is_checked(self) -> None:
        """Test that a graph stage returning fewer tensors than it declares is rejected."""
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_Double(), inputs=("z",), outputs=("plus", "minus"))
        with self.assertRaisesRegex(ValueError, "returned 1 tensor"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_graph_stage_inputs_are_tensors(self) -> None:
        """Test that a graph stage input that is not a tensor is rejected."""
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"z": [state["y"]]}, inputs=("y",), outputs=("z",)
        )
        with self.assertRaisesRegex(TypeError, "'head' input 'z' is a list"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_glue_modifying_an_input_in_place_is_rejected(self) -> None:
        """Test that glue modifying an input in place is rejected."""
        stages = list(_toy_stages())
        stages[2] = TorchStage(
            "glue", run=lambda state: {"z": state["y"].add_(100.0)}, inputs=("y",), outputs=("z",)
        )
        with self.assertRaisesRegex(ValueError, r"TorchStage 'glue' modified its inputs \['y'\]"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_graph_module_modifying_its_input_in_place_is_rejected(self) -> None:
        """Test that a graph module modifying its input in place is rejected."""
        stages = list(_toy_stages())
        stages[1] = GraphStage("encoder", module=_DoubleInPlace(), inputs=("x",), outputs=("y",))
        with self.assertRaisesRegex(ValueError, "GraphStage 'encoder' modified its inputs"):
            run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))

    def test_graph_inputs_are_the_tensors_the_graph_ran_on(self) -> None:
        """Test that the recorded graph inputs are the run-state tensors themselves."""
        state = run_stages_in_torch(_toy_stages(), {"x": torch.tensor([1.0])}, torch.device("cpu"))
        (head_input,) = state.graph_inputs["head"]
        self.assertIs(head_input, state["z"])

    def test_graph_stage_outputs_are_tensors(self) -> None:
        """Test that a nested or non-tensor graph output is rejected."""
        for module, other in ((_NestedHead(), "tuple"), (_ScalarHead(), "int")):
            with self.subTest(other=other):
                stages = list(_toy_stages())
                stages[3] = GraphStage(
                    "head", module=module, inputs=("z",), outputs=("plus", "minus")
                )
                with self.assertRaisesRegex(TypeError, rf"'head' returned \['{other}'\] among"):
                    run_stages_in_torch(stages, {"x": torch.tensor([1.0])}, torch.device("cpu"))


class TestBackend(unittest.TestCase):
    """Unit tests for the backend names and their artifact paths."""

    def test_artifact_path_per_backend(self) -> None:
        """Test that each exporting backend has its artifact path and pytorch has none."""
        self.assertEqual(artifact_path("/tmp", "s", Backend.ONNX), Path("/tmp/s.onnx"))
        self.assertEqual(artifact_path("/tmp", "s", Backend.TENSORRT), Path("/tmp/s.engine"))
        with self.assertRaisesRegex(ValueError, "no exported artifact"):
            artifact_path("/tmp", "s", Backend.PYTORCH)

    def test_backend_name_formats_as_its_config_spelling(self) -> None:
        """Test that a backend formats as its config spelling and an unknown one is rejected."""
        self.assertEqual(f"{Backend.TENSORRT}", "tensorrt")
        self.assertIs(Backend("onnx"), Backend.ONNX)
        with self.assertRaisesRegex(ValueError, "Unknown backend 'tensorflow'"):
            Backend("tensorflow")


if __name__ == "__main__":
    unittest.main()
