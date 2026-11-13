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

"""Unit tests for export specs derived from a deployment declaration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
from typing import Any
import unittest

from omegaconf import OmegaConf
import torch
from torch import nn

from autoware_ml.deployment.stages import (
    DeploymentDeclaration,
    DeployGraph,
    build_export_specs_from_deployment,
)
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


class _GraphModel(_ToyModel):
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

    def deploy_graphs(self) -> Sequence[DeployGraph]:
        return (
            DeployGraph(
                "encoder",
                module=self.encoder,
                inputs=("features",),
                outputs=("encoded",),
            ),
            DeployGraph("head", module=self.head, inputs=("bev",), outputs=("plus", "minus")),
        )

    def deployed_forward(self, multi_task_batch_inputs, graphs):
        features = multi_task_batch_inputs["points"].float()
        encoded = graphs["encoder"](features=features)["encoded"]
        return graphs["head"](bev=encoded.sum(dim=0, keepdim=True))


class _PlainModel(_ToyModel):
    def forward(self, multi_task_batch_inputs: Mapping[str, Any]) -> torch.Tensor:
        return multi_task_batch_inputs["x"] + 1.0

    def compute_metrics(self, multi_task_batch_inputs: Mapping[str, Any], outputs: Any):
        del multi_task_batch_inputs
        return {"loss": outputs.sum()}

    def build_export_spec(self, batch_inputs: Mapping[str, Any]) -> ExportSpec:
        return ExportSpec(
            module=self,
            args=(batch_inputs["x"],),
            input_names=["x"],
            output_names=["y"],
        )


class _Wrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def _batch() -> dict[str, torch.Tensor]:
    return {"points": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "unused": torch.zeros(1)}


def _identity_deployment(**axes) -> DeploymentDeclaration:
    """One identity graph ``g`` fed the batch's ``x``."""
    return DeploymentDeclaration(
        (DeployGraph("g", module=nn.Identity(), inputs=("x",), outputs=("y",), **axes),),
        lambda multi_task_batch_inputs, graphs: graphs["g"](x=multi_task_batch_inputs["x"]),
    )


class TestExportSpecsFromDeployment(unittest.TestCase):
    """Unit tests for the export specs derived from a declared deployment."""

    def test_specs_are_derived_one_per_graph_in_declaration_order(self) -> None:
        """Test that each graph gets one spec, in order, with its declared names."""
        model = _GraphModel().eval()
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

    def test_trace_inputs_are_the_tensors_the_function_passed(self) -> None:
        """Test that each spec traces on the tensors the function passed, matching forward()."""
        model = _GraphModel().eval()
        specs = model.build_export_specs(_batch())

        (features,) = specs["encoder"].args
        self.assertTrue(torch.equal(features, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        (bev,) = specs["head"].args
        # encoder doubles, the glue sums the rows: [[2,4],[6,8]] -> [[8,12]]
        self.assertTrue(torch.equal(bev, torch.tensor([[8.0, 12.0]])))
        plus, minus = specs["head"].module(*specs["head"].args)
        forward = model(_batch())
        self.assertTrue(torch.equal(plus, forward["plus"]))
        self.assertTrue(torch.equal(minus, forward["minus"]))

    def test_graph_declared_axes_land_in_the_spec(self) -> None:
        """Test that a graph's declared dynamic axes become the spec's dynamic axes."""
        deployment = _identity_deployment(
            onnx_dynamic_axes={"x": {0: "num_points"}, "y": {0: "num_points"}}
        )
        specs = build_export_specs_from_deployment(
            deployment, {"x": torch.ones(3, 2)}, torch.device("cpu")
        )
        self.assertEqual(specs["g"].dynamic_axes, {"x": {0: "num_points"}, "y": {0: "num_points"}})

    def test_multi_input_spec_args_follow_the_declared_input_order(self) -> None:
        """Test that a spec's args and input names follow the graph's declared input order."""

        # ONNX binds inputs by position: args and names follow ``inputs``, not the order the
        # function passed the keywords in.
        class _Subtract(nn.Module):
            def forward(self, a, b):
                return a - b

        deployment = DeploymentDeclaration(
            (DeployGraph("sub", module=_Subtract(), inputs=("a", "b"), outputs=("diff",)),),
            lambda multi_task_batch_inputs, graphs: graphs["sub"](
                b=torch.tensor([1.0]), a=torch.tensor([10.0])
            ),
        )
        spec = build_export_specs_from_deployment(deployment, None, torch.device("cpu"))["sub"]
        self.assertEqual(spec.input_names, ["a", "b"])
        self.assertEqual([arg.item() for arg in spec.args], [10.0, 1.0])
        self.assertTrue(torch.equal(spec.module(*spec.args), torch.tensor([9.0])))

    def test_graphs_passed_as_a_generator_export_every_graph(self) -> None:
        """Test that graphs given as a generator are read once and every graph is exported."""

        # The graphs must be read once: a generator is empty on a second pass.
        def graphs():
            yield DeployGraph("g", module=nn.Identity(), inputs=("x",), outputs=("y",))

        deployment = DeploymentDeclaration(
            graphs(),
            lambda multi_task_batch_inputs, graphs: graphs["g"](x=torch.ones(2)),
        )
        self.assertEqual(
            list(build_export_specs_from_deployment(deployment, None, torch.device("cpu"))),
            ["g"],
        )

    def test_a_misspelled_input_name_fails_at_export_naming_both_sides(self) -> None:
        """Test that a misspelled input name fails at export, naming the graph and the names."""
        deployment = DeploymentDeclaration(
            (DeployGraph("g", module=nn.Identity(), inputs=("typo",), outputs=("y",)),),
            lambda multi_task_batch_inputs, graphs: graphs["g"](x=multi_task_batch_inputs["x"]),
        )
        with self.assertRaisesRegex(TypeError, r"'g' takes \['typo'\], got \['x'\]"):
            build_export_specs_from_deployment(
                deployment, {"x": torch.ones(1)}, torch.device("cpu")
            )

    def test_a_graph_input_reused_in_place_is_rejected(self) -> None:
        """Test that the function rewriting a graph's input in place is rejected at export."""

        # nn.Identity returns the tensor it got: in place, the function would rewrite g's
        # trace input.
        def forward(multi_task_batch_inputs, graphs):
            y = graphs["g"](x=torch.ones(2))["y"]
            return graphs["h"](z=y.mul_(3.0))

        deployment = DeploymentDeclaration(
            (
                DeployGraph("g", module=nn.Identity(), inputs=("x",), outputs=("y",)),
                DeployGraph("h", module=nn.Identity(), inputs=("z",), outputs=("w",)),
            ),
            forward,
        )
        with self.assertRaisesRegex(ValueError, r"\{'g': \['x'\]\}"):
            build_export_specs_from_deployment(deployment, None, torch.device("cpu"))


class TestExportEvalMode(unittest.TestCase):
    """Unit tests for the eval mode a deployment is traced and left in."""

    def test_a_deployment_refuses_to_trace_in_training_mode(self) -> None:
        """Test that a model with deployment graphs in training mode refuses to build specs."""
        model = _GraphModel().train()
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            model.build_export_specs(_batch())

    def test_a_deployment_refuses_to_trace_with_a_submodule_in_training_mode(
        self,
    ) -> None:
        """Test that a submodule in training mode under an eval model is named and refused."""
        model = _GraphModel().eval()
        model.encoder.train()
        with self.assertRaisesRegex(RuntimeError, r"eval mode.*\['encoder', 'encoder.linear'\]"):
            model.build_export_specs(_batch())

    def test_export_leaves_the_submodules_a_wrapper_shares_in_eval_mode(self) -> None:
        """Test that exporting a wrapper leaves the submodules it shares in eval mode."""
        shared = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).eval()
        deployment = DeploymentDeclaration(
            (DeployGraph("wrapped", module=_Wrapper(shared), inputs=("x",), outputs=("y",)),),
            lambda multi_task_batch_inputs, graphs: graphs["wrapped"](
                x=multi_task_batch_inputs["x"]
            ),
        )
        spec = build_export_specs_from_deployment(
            deployment, {"x": torch.rand(4, 2)}, torch.device("cpu")
        )["wrapped"]
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


class TestModelsWithoutDeploymentGraphs(unittest.TestCase):
    """Unit tests for the export path of a model that declares no deployment graphs."""

    def test_models_without_graphs_keep_the_end_to_end_default(self) -> None:
        """Test that a model without deploy_graphs() keeps the single end_to_end spec."""
        model = _PlainModel()
        self.assertIsNone(model.deploy_graphs())
        self.assertIsNone(model.deployment_declaration())
        specs = model.build_export_specs({"x": torch.tensor([1.0])})
        self.assertEqual(list(specs), ["end_to_end"])
        self.assertEqual(specs["end_to_end"].input_names, ["x"])


if __name__ == "__main__":
    unittest.main()
