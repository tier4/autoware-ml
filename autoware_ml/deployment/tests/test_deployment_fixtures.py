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

"""Tests of the forward-parity check shared by the deployed models' tests."""

from collections.abc import Mapping, Sequence
import unittest

from jaxtyping import Float32
import torch
from torch import nn

from autoware_ml.deployment.graphs import DeploymentDeclaration, DeploymentGraph, Graphs
from autoware_ml.deployment.tests.deployment_fixtures import assert_deployment_reproduces_forward


class _TwoStepModel(nn.Module):
    """``forward()`` = scale then linear; the deployment does the same two steps.

    Pairs its graphs and function as ``ModuleBaseModel.deployment_declaration()`` does, which
    is what the check asks a model for.
    """

    def __init__(self, graph_scale: float = 2.0) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.graph_scale = graph_scale

    def forward(
        self, multi_task_batch_inputs: Float32[torch.Tensor, "batch_size 3"]
    ) -> Float32[torch.Tensor, "batch_size channels"]:
        return self.linear(multi_task_batch_inputs * 2.0)

    def deployment_graphs(self) -> Sequence[DeploymentGraph]:
        return (
            DeploymentGraph("linear", module=self.linear, inputs=("scaled",), outputs=("logits",)),
        )

    def deployment_forward(
        self, multi_task_batch_inputs: Float32[torch.Tensor, "batch_size 3"], graphs: Graphs
    ) -> Mapping[str, Float32[torch.Tensor, "batch_size channels"]]:
        return graphs["linear"](scaled=multi_task_batch_inputs * self.graph_scale)

    def deployment_declaration(self) -> DeploymentDeclaration:
        return DeploymentDeclaration(tuple(self.deployment_graphs()), self.deployment_forward)


class TestAssertDeploymentReproducesForward(unittest.TestCase):
    """Unit tests of ``assert_deployment_reproduces_forward``."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.batch = torch.randn(4, 3)

    def test_returns_the_run_when_the_deployment_reproduces_forward(self) -> None:
        """Test that equal compositions pass and the run, with its graph inputs, comes back."""
        model = _TwoStepModel().eval()
        check = assert_deployment_reproduces_forward(
            model,
            self.batch,
            self.device,
            lambda outputs: {"logits": outputs},
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual([graph.name for graph in check.graphs], ["linear"])
        self.assertEqual(tuple(check.forward_outputs.shape), (4, 2))
        (scaled,) = check.run.graph_inputs["linear"]
        self.assertEqual(tuple(scaled.shape), (4, 3))

    def test_fails_when_the_glue_differs_from_forward(self) -> None:
        """Test that a deployment whose glue drifts from forward() fails the check."""
        model = _TwoStepModel(graph_scale=3.0).eval()
        with self.assertRaisesRegex(AssertionError, "'logits' differs from forward"):
            assert_deployment_reproduces_forward(
                model,
                self.batch,
                self.device,
                lambda outputs: {"logits": outputs},
                rtol=1e-6,
                atol=1e-6,
            )

    def test_fails_when_the_reference_does_not_name_the_model_outputs(self) -> None:
        """Test that a reference mapping with other names than the model outputs is refused."""
        model = _TwoStepModel().eval()
        with self.assertRaisesRegex(AssertionError, "exactly the model outputs"):
            assert_deployment_reproduces_forward(
                model,
                self.batch,
                self.device,
                lambda outputs: {"scores": outputs},
                rtol=1e-6,
                atol=1e-6,
            )

    def test_fails_for_a_model_without_graphs(self) -> None:
        """Test that a model declaring no graphs fails instead of passing vacuously."""

        class _NoGraphs(nn.Module):
            def forward(
                self, multi_task_batch_inputs: Float32[torch.Tensor, "batch_size 3"]
            ) -> Float32[torch.Tensor, "batch_size channels"]:
                return multi_task_batch_inputs

            def deployment_declaration(self) -> None:
                return None

        with self.assertRaisesRegex(AssertionError, "declares no deployment graphs"):
            assert_deployment_reproduces_forward(
                _NoGraphs().eval(),
                self.batch,
                self.device,
                lambda outputs: {"x": outputs},
                rtol=1e-6,
                atol=1e-6,
            )
