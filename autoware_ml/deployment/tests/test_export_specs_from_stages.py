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

"""Unit tests for export specs derived from a stage graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
from typing import Any
import unittest

from omegaconf import OmegaConf
import torch
from torch import nn

from autoware_ml.deployment.export_specs import build_export_specs_from_stages
from autoware_ml.deployment.stages import GraphStage, Stage, TorchStage
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec, export_to_onnx


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2) * 2)

    def forward(self, features):
        return self.linear(features)


class _Head(nn.Module):
    def forward(self, bev):
        return bev + 1, bev - 1


class _ToyModel(ModuleBaseModel):
    def __init__(self) -> None:
        super().__init__(
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        )

    def decode_outputs(self, multi_task_batch_inputs: Mapping[str, Any], outputs: Any) -> Any:
        del multi_task_batch_inputs
        return outputs


class _StagedModel(_ToyModel):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.head = _Head()

    def forward(self, multi_task_batch_inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        bev = self.encoder(multi_task_batch_inputs["points"].float()).sum(dim=0, keepdim=True)
        plus, minus = self.head(bev)
        return {"plus": plus, "minus": minus}

    def compute_metrics(self, multi_task_batch_inputs: Mapping[str, Any], outputs: Any):
        del multi_task_batch_inputs
        return {"loss": outputs["plus"].sum()}

    def build_stages(self) -> Sequence[Stage]:
        def prep(state):
            return {"features": state.batch["points"].float()}

        def scatter(state):
            return {"bev": state["encoded"].sum(dim=0, keepdim=True)}

        return (
            TorchStage("prep", run=prep, inputs=(), outputs=("features",)),
            GraphStage("encoder", module=self.encoder, inputs=("features",), outputs=("encoded",)),
            TorchStage("scatter", run=scatter, inputs=("encoded",), outputs=("bev",)),
            GraphStage("head", module=self.head, inputs=("bev",), outputs=("plus", "minus")),
        )


class _PlainModel(_ToyModel):
    def forward(self, multi_task_batch_inputs: Mapping[str, Any]) -> torch.Tensor:
        return multi_task_batch_inputs["x"] + 1.0

    def compute_metrics(self, multi_task_batch_inputs: Mapping[str, Any], outputs: Any):
        del multi_task_batch_inputs
        return {"loss": outputs.sum()}

    def build_export_spec(self, batch_inputs: Mapping[str, Any]) -> ExportSpec:
        return ExportSpec(
            module=self, args=(batch_inputs["x"],), input_names=["x"], output_names=["y"]
        )


class _Wrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def _batch() -> dict[str, torch.Tensor]:
    return {"points": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "unused": torch.zeros(1)}


class TestExportSpecsFromStages(unittest.TestCase):
    """Unit tests for the export specs derived from a declared stage graph."""

    def test_specs_are_derived_one_per_graph_stage_in_order(self) -> None:
        """Test that each graph stage gets one spec, in order, with its declared names."""
        model = _StagedModel().eval()
        specs = model.build_export_specs(_batch())

        self.assertEqual(list(specs), ["encoder", "head"])
        encoder, head = specs["encoder"], specs["head"]
        self.assertEqual(encoder.input_names, ["features"])
        self.assertEqual(encoder.output_names, ["encoded"])
        self.assertEqual(head.input_names, ["bev"])
        self.assertEqual(head.output_names, ["plus", "minus"])
        self.assertIs(encoder.module, model.encoder)
        self.assertIs(head.module, model.head)
        self.assertIsNone(encoder.dynamic_axes)
        self.assertEqual(encoder.supported_stages, frozenset({"onnx", "tensorrt"}))

    def test_trace_inputs_are_the_run_state_tensors_the_glue_produced(self) -> None:
        """Test that each spec traces on the tensors the glue produced, matching forward()."""
        model = _StagedModel().eval()
        specs = model.build_export_specs(_batch())

        (features,) = specs["encoder"].args
        self.assertTrue(torch.equal(features, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        (bev,) = specs["head"].args
        # encoder doubles, scatter sums the rows: [[2,4],[6,8]] -> [[8,12]]
        self.assertTrue(torch.equal(bev, torch.tensor([[8.0, 12.0]])))
        plus, minus = specs["head"].module(*specs["head"].args)
        forward = model(_batch())
        self.assertTrue(torch.equal(plus, forward["plus"]))
        self.assertTrue(torch.equal(minus, forward["minus"]))

    def test_stage_declared_axes_land_in_the_spec(self) -> None:
        """Test that a stage's declared dynamic axes become the spec's dynamic axes."""
        stages = (
            TorchStage(
                "seed", run=lambda state: {"x": state.batch["x"]}, inputs=(), outputs=("x",)
            ),
            GraphStage(
                "points",
                module=nn.Identity(),
                inputs=("x",),
                outputs=("y",),
                onnx_dynamic_axes={"x": {0: "num_points"}, "y": {0: "num_points"}},
            ),
        )
        specs = build_export_specs_from_stages(stages, {"x": torch.ones(3, 2)}, torch.device("cpu"))
        self.assertEqual(
            specs["points"].dynamic_axes, {"x": {0: "num_points"}, "y": {0: "num_points"}}
        )

    def test_multi_input_spec_args_follow_the_declared_input_order(self) -> None:
        """Test that a spec's args and input names follow the stage's declared input order."""

        # ONNX binds inputs by position: args and names follow ``inputs``, not the order the
        # glue produced the tensors in.
        class _Subtract(nn.Module):
            def forward(self, a, b):
                return a - b

        stages = (
            TorchStage(
                "seed",
                run=lambda state: {"b": torch.tensor([1.0]), "a": torch.tensor([10.0])},
                inputs=(),
                outputs=("b", "a"),
            ),
            GraphStage("sub", module=_Subtract(), inputs=("a", "b"), outputs=("diff",)),
        )
        spec = build_export_specs_from_stages(stages, None, torch.device("cpu"))["sub"]
        self.assertEqual(spec.input_names, ["a", "b"])
        self.assertEqual([arg.item() for arg in spec.args], [10.0, 1.0])
        self.assertTrue(torch.equal(spec.module(*spec.args), torch.tensor([9.0])))

    def test_stages_passed_as_a_generator_export_every_graph(self) -> None:
        """Test that stages given as a generator are read once and every graph is exported."""

        # The stages must be read once: a generator is empty on a second pass.
        def stages():
            yield TorchStage(
                "seed", run=lambda state: {"x": torch.ones(2)}, inputs=(), outputs=("x",)
            )
            yield GraphStage("g", module=nn.Identity(), inputs=("x",), outputs=("y",))

        self.assertEqual(
            list(build_export_specs_from_stages(stages(), None, torch.device("cpu"))), ["g"]
        )

    def test_a_stage_reading_an_unwritten_name_fails_before_anything_runs(self) -> None:
        """Test that a misspelled input name fails at export, naming the stage and the name."""
        stages = (
            TorchStage(
                "seed", run=lambda state: {"x": state.batch["x"]}, inputs=(), outputs=("x",)
            ),
            GraphStage("g", module=nn.Identity(), inputs=("typo",), outputs=("y",)),
        )
        with self.assertRaisesRegex(ValueError, r"'g' reads \['typo'\].*written so far: \['x'\]"):
            build_export_specs_from_stages(stages, {"x": torch.ones(1)}, torch.device("cpu"))

    def test_glue_reusing_a_graph_output_in_place_is_rejected(self) -> None:
        """Test that glue rewriting a graph's output in place is rejected at export."""
        # nn.Identity returns the tensor it got: in place, the glue would rewrite g's trace
        # input.
        stages = (
            TorchStage("seed", run=lambda state: {"x": torch.ones(2)}, inputs=(), outputs=("x",)),
            GraphStage("g", module=nn.Identity(), inputs=("x",), outputs=("y",)),
            TorchStage(
                "reuse",
                run=lambda state: {"z": state["y"].mul_(3.0)},
                inputs=("y",),
                outputs=("z",),
            ),
            GraphStage("h", module=nn.Identity(), inputs=("z",), outputs=("w",)),
        )
        with self.assertRaisesRegex(ValueError, r"'reuse' modified its inputs \['y'\] in place"):
            build_export_specs_from_stages(stages, None, torch.device("cpu"))


class TestExportEvalMode(unittest.TestCase):
    """Unit tests for the eval mode a stage graph is traced and left in."""

    def test_a_stage_graph_refuses_to_trace_in_training_mode(self) -> None:
        """Test that a stage-graph model in training mode refuses to build its specs."""
        model = _StagedModel().train()
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            model.build_export_specs(_batch())

    def test_a_stage_graph_refuses_to_trace_with_a_submodule_in_training_mode(self) -> None:
        """Test that a submodule in training mode under an eval model is named and refused."""
        model = _StagedModel().eval()
        model.encoder.train()
        with self.assertRaisesRegex(RuntimeError, r"eval mode.*\['encoder', 'encoder.linear'\]"):
            model.build_export_specs(_batch())

    def test_export_leaves_the_submodules_a_wrapper_shares_in_eval_mode(self) -> None:
        """Test that exporting a wrapper leaves the submodules it shares in eval mode."""
        shared = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).eval()
        stages = (
            TorchStage(
                "seed", run=lambda state: {"x": state.batch["x"]}, inputs=(), outputs=("x",)
            ),
            GraphStage("wrapped", module=_Wrapper(shared), inputs=("x",), outputs=("y",)),
        )
        spec = build_export_specs_from_stages(stages, {"x": torch.rand(4, 2)}, torch.device("cpu"))[
            "wrapped"
        ]
        self.assertFalse(spec.module.training)

        with tempfile.TemporaryDirectory() as output_dir:
            export_to_onnx(
                spec,
                OmegaConf.create({"dynamo": False, "opset_version": 17}),
                Path(output_dir) / "wrapped.onnx",
            )
        # torch.onnx.export restores the wrapper's mode recursively; a wrapper left in training
        # mode would have switched the shared BatchNorm to training mode.
        self.assertFalse(any(module.training for module in shared.modules()))


class TestModelsWithoutStageGraph(unittest.TestCase):
    """Unit tests for the export path of a model that declares no stage graph."""

    def test_models_without_a_stage_graph_keep_the_end_to_end_default(self) -> None:
        """Test that a model without build_stages() keeps the single end_to_end spec."""
        model = _PlainModel()
        self.assertIsNone(model.build_stages())
        specs = model.build_export_specs({"x": torch.tensor([1.0])})
        self.assertEqual(list(specs), ["end_to_end"])
        self.assertEqual(specs["end_to_end"].input_names, ["x"])

    def test_scoring_deployed_outputs_needs_the_model_mapping(self) -> None:
        """Test that scoring deployed outputs fails for a model that does not map them."""
        with self.assertRaisesRegex(NotImplementedError, "_PlainModel does not implement"):
            _PlainModel().build_deployed_eval_output({}, {})


if __name__ == "__main__":
    unittest.main()
