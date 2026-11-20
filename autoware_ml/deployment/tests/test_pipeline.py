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

"""Unit tests for the deployed pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

from jaxtyping import Float32
import torch
from torch import nn

from autoware_ml.deployment.graphs import (
    DeploymentDeclaration,
    DeploymentGraph,
    GraphExecutor,
    GraphTensor,
)
from autoware_ml.deployment.pipeline import DeployedPipeline, artifact_path
from autoware_ml.types.backend import Backend

REQUIRES_TENSORRT = unittest.skipUnless(
    find_spec("tensorrt") is not None and torch.cuda.is_available(),
    "the tensorrt backend needs the tensorrt package and CUDA",
)

#: The toy deployment's batch: one tensor named ``x``.
_Batch = Mapping[str, Float32[torch.Tensor, "*shape"]]


class _Scale(nn.Module):
    """The toy encoder: multiplies its input by a learned scale, 2 at construction."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: Float32[torch.Tensor, "*shape"]) -> Float32[torch.Tensor, "*shape"]:
        return x * self.scale


class _SplitHead(nn.Module):
    """The toy head: its input plus one and minus one."""

    def forward(
        self, x: Float32[torch.Tensor, "*shape"]
    ) -> tuple[Float32[torch.Tensor, "*shape"], Float32[torch.Tensor, "*shape"]]:
        return x + 1, x - 1


def _toy_forward(
    batch: _Batch, executors: Mapping[str, GraphExecutor], device: torch.device
) -> Mapping[str, GraphTensor]:
    """The toy deployed function: the encoder, half a unit of glue, then the head."""
    y = executors["encoder"].run(x=batch["x"].float())["y"]
    return executors["head"].run(z=y.to(device) + 0.5)


def _toy_deployment(encoder: nn.Module, head: nn.Module) -> DeploymentDeclaration:
    """The toy deployment: ``encoder`` on ``x -> y`` and ``head`` on ``z -> (plus, minus)``."""
    return DeploymentDeclaration(
        (
            DeploymentGraph("encoder", module=encoder, input_names=("x",), output_names=("y",)),
            DeploymentGraph(
                "head", module=head, input_names=("z",), output_names=("plus", "minus")
            ),
        ),
        _toy_forward,
    )


class TestDeployedPipeline(unittest.TestCase):
    """The deployed pipeline runs the deployed function on PyTorch and TensorRT."""

    def setUp(self) -> None:
        """Build the toy deployment and a directory for its artifacts."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)
        self.encoder = _Scale()
        self.deployment = _toy_deployment(self.encoder, _SplitHead())

    def test_pytorch_backend_runs_the_function_on_the_live_modules(self) -> None:
        """Test that the pytorch backend runs the deployed function on the modules as they are.

        Input: the toy deployment on CPU with ``x = [1, 2, 3]``, the encoder in training mode;
            then the encoder's scale set to 3 and the pipeline run again.
        Expected: ``plus = 2x + 1.5`` and ``minus = 2x - 0.5``; the encoder still in training
            mode; ``plus = 3x + 1.5`` after the change, so the pipeline holds no copy.
        Check: the outputs of both runs and the encoder's ``training`` flag.
        """
        self.encoder.train()
        pipeline = DeployedPipeline(self.deployment, Backend.PYTORCH, torch.device("cpu"))
        outputs = pipeline.run({"x": torch.tensor([1, 2, 3])})

        torch.testing.assert_close(outputs["plus"], torch.tensor([3.5, 5.5, 7.5]))
        torch.testing.assert_close(outputs["minus"], torch.tensor([1.5, 3.5, 5.5]))
        self.assertTrue(self.encoder.training)

        with torch.no_grad():
            self.encoder.scale.fill_(3.0)
        torch.testing.assert_close(
            pipeline.run({"x": torch.tensor([1, 2, 3])})["plus"], torch.tensor([4.5, 7.5, 10.5])
        )

    def test_tensorrt_backend_requires_artifacts_dir(self) -> None:
        """Test that the tensorrt backend without an artifacts directory is refused.

        Input: the toy deployment on the tensorrt backend with no ``artifacts_dir``.
        Expected: a ValueError naming ``artifacts_dir``.
        Check: the error message.
        """
        with self.assertRaisesRegex(ValueError, "artifacts_dir"):
            DeployedPipeline(self.deployment, Backend.TENSORRT, torch.device("cuda"))

    def test_graph_output_arity_is_checked(self) -> None:
        """Test that a graph returning fewer tensors than it declares is refused.

        Input: the toy deployment with a one-output module as the two-output head.
        Expected: a ValueError naming the graph, the tensor count and the declared count.
        Check: the error message.
        """
        pipeline = DeployedPipeline(
            _toy_deployment(self.encoder, _Scale()), Backend.PYTORCH, torch.device("cpu")
        )
        with self.assertRaisesRegex(
            ValueError, r"'head' returned 1 tensors; it declares 2 outputs"
        ):
            pipeline.run({"x": torch.tensor([1.0])})

    @REQUIRES_TENSORRT
    def test_tensorrt_backend_matches_pytorch(self) -> None:
        """Test that the tensorrt backend runs the exported engines and matches pytorch.

        Input: the toy graphs exported to ONNX and built into engines under their names, the
            pipelines of both backends run on ``x = [1, 2, 3]``.
        Expected: the tensorrt outputs equal the pytorch outputs.
        Check: both outputs.
        """
        from autoware_ml.deployment.tests.test_tensorrt_runner import build_engine  # noqa: PLC0415

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

        reference = DeployedPipeline(self.deployment, Backend.PYTORCH, torch.device("cuda"))
        tensorrt = DeployedPipeline(
            self.deployment, Backend.TENSORRT, torch.device("cuda"), artifacts_dir=self.tmp_path
        )
        batch = {"x": torch.tensor([1, 2, 3], device="cuda")}
        expected, outputs = reference.run(batch), tensorrt.run(batch)
        for name in ("plus", "minus"):
            torch.testing.assert_close(outputs[name], expected[name])


class TestArtifacts(unittest.TestCase):
    """The artifact each backend runs, and the backend names."""

    def test_artifact_path_per_backend(self) -> None:
        """Test that the tensorrt backend has its artifact path and pytorch has none.

        Input: the graph name ``s`` under ``/tmp`` for each backend.
        Expected: ``s.engine``, and a ValueError for pytorch.
        Check: the path and the error message.
        """
        self.assertEqual(artifact_path("/tmp", "s", Backend.TENSORRT), Path("/tmp/s.engine"))
        with self.assertRaisesRegex(ValueError, "no exported artifact"):
            artifact_path("/tmp", "s", Backend.PYTORCH)

    def test_backend_name_formats_as_its_config_spelling(self) -> None:
        """Test that a backend formats as its config spelling and an unknown one is rejected.

        Input: ``Backend.TENSORRT`` formatted, ``Backend("pytorch")`` and ``Backend("onnx")``.
        Expected: ``tensorrt``, ``Backend.PYTORCH``, and a ValueError naming the unknown backend.
        Check: the string, the identity and the error message.
        """
        self.assertEqual(f"{Backend.TENSORRT}", "tensorrt")
        self.assertIs(Backend("pytorch"), Backend.PYTORCH)
        with self.assertRaisesRegex(ValueError, "Unknown backend 'onnx'"):
            Backend("onnx")


if __name__ == "__main__":
    unittest.main()
