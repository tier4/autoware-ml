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
import unittest

from jaxtyping import Float32
from omegaconf import OmegaConf
import torch
from torch import nn

from autoware_ml.deployment.graphs import (
    DeploymentDeclaration,
    DeploymentGraph,
    build_export_specs_from_deployment,
)
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec, export_to_onnx

#: The toy batch: named float tensors of any shape.
_Batch = Mapping[str, Float32[torch.Tensor, "*shape"]]
#: What a toy model's forward() returns: one tensor, or tensors by name.
_Outputs = Float32[torch.Tensor, "*shape"] | Mapping[str, Float32[torch.Tensor, "*shape"]]


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

    def decode_outputs(self, multi_task_batch_inputs: _Batch, outputs: _Outputs) -> _Outputs:
        del multi_task_batch_inputs
        return outputs


class _GraphModel(_ToyModel):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.head = _Head()

    def forward(self, multi_task_batch_inputs: _Batch) -> dict[str, Float32[torch.Tensor, "1 2"]]:
        bev = self.encoder(multi_task_batch_inputs["points"].float()).sum(dim=0, keepdim=True)
        plus, minus = self.head(bev)
        return {"plus": plus, "minus": minus}

    def compute_metrics(
        self, multi_task_batch_inputs: _Batch, outputs: Mapping[str, Float32[torch.Tensor, "1 2"]]
    ) -> dict[str, Float32[torch.Tensor, ""]]:
        del multi_task_batch_inputs
        return {"loss": outputs["plus"].sum()}

    def deployment_graphs(self) -> Sequence[DeploymentGraph]:
        return (
            DeploymentGraph(
                "encoder",
                module=self.encoder,
                input_names=("features",),
                output_names=("encoded",),
            ),
            DeploymentGraph(
                "head", module=self.head, input_names=("bev",), output_names=("plus", "minus")
            ),
        )

    def deployment_forward(self, multi_task_batch_inputs, executors, device):
        features = multi_task_batch_inputs["points"].float()
        encoded = executors["encoder"].run(features=features)["encoded"]
        return executors["head"].run(bev=encoded.sum(dim=0, keepdim=True))


class _PlainModel(_ToyModel):
    def forward(self, multi_task_batch_inputs: _Batch) -> Float32[torch.Tensor, " num"]:
        return multi_task_batch_inputs["x"] + 1.0

    def compute_metrics(
        self, multi_task_batch_inputs: _Batch, outputs: Float32[torch.Tensor, " num"]
    ) -> dict[str, Float32[torch.Tensor, ""]]:
        del multi_task_batch_inputs
        return {"loss": outputs.sum()}

    def build_export_spec(self, batch_inputs: _Batch) -> ExportSpec:
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


def _batch() -> dict[str, Float32[torch.Tensor, "*shape"]]:
    """The toy batch: ``points`` (2, 2) and an ``unused`` tensor the deployment ignores."""
    return {"points": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "unused": torch.zeros(1)}


def _identity_deployment(
    onnx_dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
) -> DeploymentDeclaration:
    """One identity graph ``g`` fed the batch's ``x``, with the given dynamic axes."""
    return DeploymentDeclaration(
        (
            DeploymentGraph(
                "g",
                module=nn.Identity(),
                input_names=("x",),
                output_names=("y",),
                onnx_dynamic_axes=onnx_dynamic_axes or {},
            ),
        ),
        lambda multi_task_batch_inputs, executors, device: executors["g"].run(
            x=multi_task_batch_inputs["x"]
        ),
    )


class TestExportSpecsFromDeployment(unittest.TestCase):
    """Unit tests for the export specs derived from a declared deployment."""

    def test_specs_are_derived_one_per_graph_in_declaration_order(self) -> None:
        """Test that each graph gets one spec, in order, with its declared names.

        Input: the two-graph toy model in eval mode and the toy batch.
        Expected: specs ``encoder`` then ``head`` with the declared input and output names,
            the live modules and no dynamic axes.
        Check: the spec keys, names, module identity and ``dynamic_axes``.
        """
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

    def test_trace_inputs_are_the_tensors_the_function_passed(self) -> None:
        """Test that each spec traces on the tensors the function passed, matching forward().

        Input: the two-graph toy model on points ``[[1, 2], [3, 4]]``.
        Expected: the encoder spec's arg equals the points; the head spec's arg is the doubled
            rows summed, ``[[8, 12]]``; the head run on it equals ``forward()``.
        Check: both spec args and the head outputs against ``forward()``.
        """
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
        """Test that a graph's declared dynamic axes become the spec's dynamic axes.

        Input: an identity graph declaring axis 0 of ``x`` and ``y`` as ``num_points``.
        Expected: the spec's ``dynamic_axes`` is that same mapping.
        Check: equality of ``dynamic_axes``.
        """
        deployment = _identity_deployment(
            onnx_dynamic_axes={"x": {0: "num_points"}, "y": {0: "num_points"}}
        )
        specs = build_export_specs_from_deployment(
            deployment, {"x": torch.ones(3, 2)}, torch.device("cpu")
        )
        self.assertEqual(specs["g"].dynamic_axes, {"x": {0: "num_points"}, "y": {0: "num_points"}})

    def test_multi_input_spec_args_follow_the_declared_input_order(self) -> None:
        """Test that a spec's args and input names follow the graph's declared input order.

        Input: a subtraction graph declaring ``(a, b)``, run with ``b=1`` before ``a=10``.
        Expected: ``input_names == ['a', 'b']``, args ``(10, 1)``, and the module gives 9.
        Check: the spec's names, arg values and the module's output.
        """

        # ONNX binds inputs by position: args and names follow ``inputs``, not the order the
        # function passed the keywords in.
        class _Subtract(nn.Module):
            def forward(self, a, b):
                return a - b

        deployment = DeploymentDeclaration(
            (
                DeploymentGraph(
                    "sub", module=_Subtract(), input_names=("a", "b"), output_names=("diff",)
                ),
            ),
            lambda multi_task_batch_inputs, executors, device: executors["sub"].run(
                b=torch.tensor([1.0]), a=torch.tensor([10.0])
            ),
        )
        spec = build_export_specs_from_deployment(deployment, None, torch.device("cpu"))["sub"]
        self.assertEqual(spec.input_names, ["a", "b"])
        self.assertEqual([arg.item() for arg in spec.args], [10.0, 1.0])
        self.assertTrue(torch.equal(spec.module(*spec.args), torch.tensor([9.0])))

    def test_trace_inputs_are_copies_taken_when_the_graph_ran(self) -> None:
        """Test that a spec traces on a copy of the graph's inputs, taken when the graph ran.

        Input: two identity graphs; the function multiplies ``g``'s output, which is its
            input tensor, in place before passing it to ``h``.
        Expected: ``g``'s trace input is still ones; ``h``'s is threes.
        Check: both spec args.
        """

        # nn.Identity returns the tensor it got: without the copy, the in-place multiply
        # would rewrite g's trace input.
        def forward(multi_task_batch_inputs, executors, device):
            y = executors["g"].run(x=torch.ones(2))["y"]
            return executors["h"].run(z=y.mul_(3.0))

        deployment = DeploymentDeclaration(
            (
                DeploymentGraph("g", module=nn.Identity(), input_names=("x",), output_names=("y",)),
                DeploymentGraph("h", module=nn.Identity(), input_names=("z",), output_names=("w",)),
            ),
            forward,
        )
        specs = build_export_specs_from_deployment(deployment, None, torch.device("cpu"))
        (x,) = specs["g"].args
        (z,) = specs["h"].args
        self.assertTrue(torch.equal(x, torch.ones(2)))
        self.assertTrue(torch.equal(z, torch.full((2,), 3.0)))


class TestExportEvalMode(unittest.TestCase):
    """Unit tests for the eval mode a deployment is traced and left in."""

    def test_a_deployment_refuses_to_trace_in_training_mode(self) -> None:
        """Test that a model with deployment graphs in training mode refuses to build specs.

        Input: the two-graph toy model in training mode.
        Expected: RuntimeError asking for eval mode.
        Check: the error message.
        """
        model = _GraphModel().train()
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            model.build_export_specs(_batch())

    def test_a_deployment_refuses_to_trace_with_a_submodule_in_training_mode(
        self,
    ) -> None:
        """Test that a submodule in training mode under an eval model is named and refused.

        Input: the toy model in eval mode with its ``encoder`` switched to training mode.
        Expected: RuntimeError listing ``['encoder', 'encoder.linear']``.
        Check: the error message.
        """
        model = _GraphModel().eval()
        model.encoder.train()
        with self.assertRaisesRegex(RuntimeError, r"eval mode.*\['encoder', 'encoder.linear'\]"):
            model.build_export_specs(_batch())

    def test_export_leaves_the_submodules_a_wrapper_shares_in_eval_mode(self) -> None:
        """Test that exporting a wrapper leaves the submodules it shares in eval mode.

        Input: a wrapper around an eval ``Linear + BatchNorm1d`` the model shares, traced and
            exported to ONNX with the TorchScript exporter.
        Expected: the spec's module is in eval mode and so is every shared submodule after
            the export.
        Check: ``training`` of the spec module and of each shared module.
        """
        shared = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).eval()
        deployment = DeploymentDeclaration(
            (
                DeploymentGraph(
                    "wrapped", module=_Wrapper(shared), input_names=("x",), output_names=("y",)
                ),
            ),
            lambda multi_task_batch_inputs, executors, device: executors["wrapped"].run(
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
        """Test that a model without deployment_graphs() keeps the single end_to_end spec.

        Input: a model with ``build_export_spec()`` only.
        Expected: no graphs, no declaration, and one ``end_to_end`` spec with input ``x``.
        Check: the two ``None`` returns, the spec keys and the input names.
        """
        model = _PlainModel()
        self.assertIsNone(model.deployment_graphs())
        self.assertIsNone(model.deployment_declaration())
        specs = model.build_export_specs({"x": torch.tensor([1.0])})
        self.assertEqual(list(specs), ["end_to_end"])
        self.assertEqual(specs["end_to_end"].input_names, ["x"])


if __name__ == "__main__":
    unittest.main()
