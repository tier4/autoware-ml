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

"""Unit tests for the TensorRT runner (GPU)."""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from omegaconf import OmegaConf
from onnx import TensorProto, helper
import torch

from autoware_ml.utils.deploy import build_tensorrt_engine

IS_TENSORRT_AVAILABLE = find_spec("tensorrt") is not None

if IS_TENSORRT_AVAILABLE:
    from autoware_ml.deployment.backends.tensorrt_runner import TensorRTModuleRunner, trt_logger

REQUIRES_CUDA = unittest.skipUnless(torch.cuda.is_available(), "TensorRT needs CUDA")

#: One dynamic input's profile, ``{"min_shape": ..., "opt_shape": ..., "max_shape": ...}``.
_Profile = Mapping[str, Sequence[int]]


def _save_graph(
    path: Path,
    nodes: Sequence[onnx.NodeProto],
    inputs: Sequence[onnx.ValueInfoProto],
    outputs: Sequence[onnx.ValueInfoProto],
    initializers: Sequence[onnx.TensorProto] = (),
) -> Path:
    """Save a one-graph ONNX model at opset 17 and return its path."""
    graph = helper.make_graph(list(nodes), "g", list(inputs), list(outputs), list(initializers))
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    return path


def identity_gemm_graph(directory: Path) -> Path:
    """Save ``y = x @ I`` over a static ``[2, 4]`` input."""
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten()
    )
    return _save_graph(
        directory / "gemm.onnx",
        [helper.make_node("Gemm", ["x", "w"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        [weight],
    )


def add_graph(directory: Path) -> Path:
    """Save ``y = a + b`` over two ``[n, 2]`` inputs sharing the dynamic axis ``n``."""
    return _save_graph(
        directory / "add.onnx",
        [helper.make_node("Add", ["a", "b"], ["y"])],
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, ["n", 2]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, ["n", 2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 2])],
    )


def nonzero_graph(directory: Path) -> Path:
    """Save ``y = NonZero(x)``, whose output shape depends on the data."""
    return _save_graph(
        directory / "nonzero.onnx",
        [helper.make_node("NonZero", ["x"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n"])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [1, "k"])],
    )


def build_engine(onnx_path: Path, input_shapes: Mapping[str, _Profile] = {}) -> Path:
    """Build ``<stem>.engine`` next to ``onnx_path`` with the export's builder and return it."""
    engine_path = onnx_path.with_suffix(".engine")
    deploy_cfg = OmegaConf.create(
        {
            "tensorrt": {
                "workspace_size": 1 << 28,
                "modules": {onnx_path.stem: {"input_shapes": dict(input_shapes)}},
            }
        }
    )
    build_tensorrt_engine(onnx_path, deploy_cfg, engine_path, onnx_path.stem)
    return engine_path


#: The ``[n, 2]`` profile of the add graph: n from 1 to 8.
_N_PROFILE: _Profile = {"min_shape": [1, 2], "opt_shape": [4, 2], "max_shape": [8, 2]}


@unittest.skipUnless(IS_TENSORRT_AVAILABLE, "the TensorRT runner needs the tensorrt package")
class TestTensorRTModuleRunner(unittest.TestCase):
    """The TensorRT runner matches PyTorch, hands out its outputs and checks its inputs."""

    def setUp(self) -> None:
        """Give each test its own directory for the ONNX files and their engines."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    @REQUIRES_CUDA
    def test_engine_matches_pytorch_and_the_caller_keeps_its_outputs(self) -> None:
        """Test that the engine matches PyTorch and a later call leaves earlier outputs alone.

        Input: the identity Gemm engine run on ``arange(8)`` and then on ``arange(8) + 1``.
        Expected: the I/O names of the graph, each output equal to its input, and the first
            output unchanged by the second call.
        Check: the names and both outputs after the second call.
        """
        runner = TensorRTModuleRunner(
            build_engine(identity_gemm_graph(self.tmp_path)), torch.device("cuda")
        )
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4).cuda()

        first = runner.run({"x": x})
        second = runner.run({"x": x + 1})
        self.assertEqual(runner.input_names, ["x"])
        self.assertEqual(runner.output_names, ["y"])
        torch.testing.assert_close(first["y"], x)
        torch.testing.assert_close(second["y"], x + 1)

    @REQUIRES_CUDA
    def test_dynamic_shapes_rebind_per_call(self) -> None:
        """Test that a dynamic engine follows the input shape of every call.

        Input: the add engine run on ``n = 4``, then ``n = 2``, then ``n = 4`` again.
        Expected: an ``[n, 2]`` output of threes each time.
        Check: the shape and the values of the three outputs.
        """
        runner = TensorRTModuleRunner(
            build_engine(add_graph(self.tmp_path), {"a": _N_PROFILE, "b": _N_PROFILE}),
            torch.device("cuda"),
        )
        for n in (4, 2, 4):
            inputs = {
                "a": torch.ones(n, 2, device="cuda"),
                "b": torch.full((n, 2), 2.0, device="cuda"),
            }
            torch.testing.assert_close(
                runner.run(inputs)["y"], torch.full((n, 2), 3.0, device="cuda")
            )

    @REQUIRES_CUDA
    def test_engines_share_one_logger_and_survive_a_reload(self) -> None:
        """Test that engines share the process logger and a second load runs after the first.

        Input: the identity Gemm engine loaded, run, deleted and collected, then loaded again.
        Expected: the runtime logs through ``trt_logger()`` and the second engine produces the
            first one's outputs.
        Check: the runtime's logger identity and the outputs.
        """
        engine_path = build_engine(identity_gemm_graph(self.tmp_path))
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4).cuda()

        first = TensorRTModuleRunner(engine_path, torch.device("cuda"))
        self.assertIs(first.runtime.logger, trt_logger())
        expected = first.run({"x": x})["y"]
        del first
        gc.collect()

        second = TensorRTModuleRunner(engine_path, torch.device("cuda"))
        torch.testing.assert_close(second.run({"x": x})["y"], expected)

    @REQUIRES_CUDA
    def test_runner_rejects_wrong_input_names_and_shapes(self) -> None:
        """Test that wrong input names and a shape outside the engine are refused, not run.

        Input: the static ``[2, 4]`` engine fed a wrong name, an extra name, a ``[3, 4]``
            tensor, then a valid tensor.
        Expected: a ValueError naming the missing and unexpected inputs, a RuntimeError naming
            the rejected shape, and the valid call still running.
        Check: the error messages and the last call's output.
        """
        runner = TensorRTModuleRunner(
            build_engine(identity_gemm_graph(self.tmp_path)), torch.device("cuda")
        )
        x = torch.ones(2, 4, device="cuda")

        with self.assertRaisesRegex(ValueError, r"missing \['x'\], unexpected \['z'\]"):
            runner.run({"z": x})
        with self.assertRaisesRegex(ValueError, r"unexpected \['z'\]"):
            runner.run({"x": x, "z": x})
        with self.assertRaisesRegex(RuntimeError, r"rejected shape \(3, 4\) for input 'x'"):
            runner.run({"x": torch.ones(3, 4, device="cuda")})
        torch.testing.assert_close(runner.run({"x": x})["y"], x)

    @REQUIRES_CUDA
    def test_runner_rejects_inputs_that_disagree_on_a_shared_axis(self) -> None:
        """Test that inputs that are each inside the profile but disagree are refused.

        Input: the add engine fed ``a`` of ``[3, 2]`` and ``b`` of ``[5, 2]``, then ``[3, 2]`` both.
        Expected: a RuntimeError naming the output and both input shapes, and the consistent
            call still running.
        Check: the error message and the last call's output.
        """
        runner = TensorRTModuleRunner(
            build_engine(add_graph(self.tmp_path), {"a": _N_PROFILE, "b": _N_PROFILE}),
            torch.device("cuda"),
        )
        with self.assertRaisesRegex(RuntimeError, r"output 'y'.*'a': \(3, 2\), 'b': \(5, 2\)"):
            runner.run({"a": torch.ones(3, 2, device="cuda"), "b": torch.ones(5, 2, device="cuda")})
        outputs = runner.run(
            {"a": torch.ones(3, 2, device="cuda"), "b": torch.ones(3, 2, device="cuda")}
        )
        torch.testing.assert_close(outputs["y"], torch.full((3, 2), 2.0, device="cuda"))

    @REQUIRES_CUDA
    def test_runner_rejects_a_data_dependent_output_shape(self) -> None:
        """Test that an engine whose output shape depends on the data is refused, not run.

        Input: the NonZero engine fed five values.
        Expected: a RuntimeError naming the output and its unresolved shape.
        Check: the error message.
        """
        runner = TensorRTModuleRunner(
            build_engine(
                nonzero_graph(self.tmp_path),
                {"x": {"min_shape": [1], "opt_shape": [8], "max_shape": [64]}},
            ),
            torch.device("cuda"),
        )
        with self.assertRaisesRegex(
            RuntimeError, r"Output 'y' has the data-dependent shape \(1, -1\)"
        ):
            runner.run({"x": torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0], device="cuda")})

    def test_runner_refuses_a_non_cuda_device(self) -> None:
        """Test that the runner refuses a CPU device before touching the engine file.

        Input: a missing engine path and ``torch.device("cpu")``.
        Expected: a ValueError naming CUDA.
        Check: the error message.
        """
        with self.assertRaisesRegex(ValueError, "CUDA"):
            TensorRTModuleRunner(self.tmp_path / "missing.engine", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
