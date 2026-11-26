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

"""Unit tests for the TensorRT deployment runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import tempfile
import unittest

from jaxtyping import Float32
import torch
from torch import nn

from autoware_ml.deployment.backends.tensorrt_deployment_runner import (
    TensorRTDeploymentRunner,
)
from autoware_ml.deployment.graphs import (
    DeploymentDeclaration,
    DeploymentGraph,
    GraphExecutor,
    GraphTensor,
    run_deployment_in_torch,
)
from autoware_ml.deployment.tests.tensorrt_fixtures import build_engine

REQUIRES_CUDA = unittest.skipUnless(torch.cuda.is_available(), "TensorRT needs CUDA")

#: The toy deployment's batch: one tensor named ``x``.
_Batch = Mapping[str, Float32[torch.Tensor, "*shape"]]


class _Scale(nn.Module):
    """The toy encoder: multiplies its input by a learned scale, 2 at construction.

    The scale is a one-element tensor, not a scalar: PyTorch lets a CPU scalar into a CUDA op,
    which would hide a module left on the CPU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x: Float32[torch.Tensor, "*shape"]) -> Float32[torch.Tensor, "*shape"]:
        """Scale the input.

        Args:
            x: The input.

        Returns:
            ``x`` times the scale.
        """
        return x * self.scale


class _SplitHead(nn.Module):
    """The toy head: its input plus one and minus one."""

    def forward(
        self, x: Float32[torch.Tensor, "*shape"]
    ) -> tuple[Float32[torch.Tensor, "*shape"], Float32[torch.Tensor, "*shape"]]:
        """Split the input into two outputs.

        Args:
            x: The input.

        Returns:
            ``x + 1`` and ``x - 1``.
        """
        return x + 1, x - 1


def _toy_forward(
    batch: _Batch, executors: Mapping[str, GraphExecutor], device: torch.device
) -> Mapping[str, GraphTensor]:
    """The toy deployed function: the encoder, half a unit of glue, then the head.

    Args:
        batch: The batch, with ``x``.
        executors: The ``encoder`` and ``head`` executors.
        device: The run device the glue moves the encoder output to.

    Returns:
        The head's ``plus`` and ``minus``.
    """
    y = executors["encoder"].run(x=batch["x"].float())["y"]
    return executors["head"].run(z=y.to(device) + 0.5)


def _toy_deployment(encoder: nn.Module, head: nn.Module) -> DeploymentDeclaration:
    """The toy deployment: ``encoder`` on ``x -> y`` and ``head`` on ``z -> (plus, minus)``.

    Args:
        encoder: The ``encoder`` graph's module.
        head: The ``head`` graph's module.

    Returns:
        The declaration with :func:`_toy_forward`.
    """
    return DeploymentDeclaration(
        (
            DeploymentGraph("encoder", module=encoder, input_names=("x",), output_names=("y",)),
            DeploymentGraph(
                "head", module=head, input_names=("z",), output_names=("plus", "minus")
            ),
        ),
        _toy_forward,
    )


class TestTensorRTDeploymentRunner(unittest.TestCase):
    """The TensorRT deployment runner runs the deployed function on the exported engines."""

    def setUp(self) -> None:
        """Build the toy deployment and a directory for its artifacts."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)
        self.deployment = _toy_deployment(_Scale(), _SplitHead())

    @REQUIRES_CUDA
    def test_runner_matches_pytorch(self) -> None:
        """Test that the runner runs the exported engines and matches the run in PyTorch.

        Input: the toy graphs exported to ONNX and built into engines under their names, the
            modules then moved to CUDA, the deployed function run on ``x = [1, 2, 3]`` in
            PyTorch and by a runner built on ``"cuda"``.
        Expected: the runner's outputs equal the PyTorch outputs, and the runner is on the
            current CUDA device by its index, as its engines are.
        Check: both outputs and the runner's ``device``.
        """
        axis = {"min_shape": [1], "opt_shape": [3], "max_shape": [8]}
        for graph in self.deployment.graphs:
            onnx_path = self.tmp_path / f"{graph.name}.onnx"
            torch.onnx.export(
                graph.module,
                (torch.ones(3),),
                str(onnx_path),
                input_names=list(graph.input_names),
                output_names=list(graph.output_names),
                dynamic_axes={name: {0: "n"} for name in (*graph.input_names, *graph.output_names)},
                opset_version=17,
                dynamo=False,
            )
            build_engine(onnx_path, {name: axis for name in graph.input_names})
            graph.module.to("cuda")

        cuda = torch.device("cuda")
        runner = TensorRTDeploymentRunner(self.deployment, self.tmp_path, cuda)
        self.assertEqual(runner.device, torch.device("cuda", torch.cuda.current_device()))
        batch = {"x": torch.tensor([1, 2, 3], device="cuda")}
        expected = run_deployment_in_torch(self.deployment, batch, cuda)
        outputs = runner.run(batch)
        for name in ("plus", "minus"):
            torch.testing.assert_close(outputs[name], expected[name])


if __name__ == "__main__":
    unittest.main()
