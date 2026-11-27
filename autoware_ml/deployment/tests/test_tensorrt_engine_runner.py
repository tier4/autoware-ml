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

"""Unit tests for the TensorRT engine runner (GPU)."""

from __future__ import annotations

import gc
from pathlib import Path
import tempfile
import unittest

import torch

from autoware_ml.deployment.backends.tensorrt_engine_runner import (
    TensorRTEngineRunner,
    pinned_cuda_device,
)
from autoware_ml.deployment.tests.tensorrt_fixtures import (
    N_PROFILE,
    add_graph,
    build_engine,
    doubling_gemm_graph,
    nonzero_graph,
)

REQUIRES_CUDA = unittest.skipUnless(torch.cuda.is_available(), "TensorRT needs CUDA")


class TestTensorRTEngineRunner(unittest.TestCase):
    """The TensorRT engine runner matches PyTorch, hands out its outputs and checks its inputs."""

    def setUp(self) -> None:
        """Give each test its own directory for the ONNX files and their engines."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    @REQUIRES_CUDA
    def test_engine_matches_pytorch_and_the_caller_keeps_its_outputs(self) -> None:
        """Test that the engine matches PyTorch and a later call leaves earlier outputs alone.

        Input: the doubling Gemm engine loaded on ``"cuda"`` and run on ``arange(8)``, then on
            ``arange(8) + 1``.
        Expected: the runner on the current CUDA device, the I/O names of the graph, each
            output twice its input, and the first output unchanged by the second call.
        Check: the device, the names and both outputs after the second call.
        """
        runner = TensorRTEngineRunner(
            build_engine(doubling_gemm_graph(self.tmp_path), {}), torch.device("cuda")
        )
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4).cuda()

        first = runner.run({"x": x})
        second = runner.run({"x": x + 1})
        self.assertEqual(runner.device, torch.device("cuda", torch.cuda.current_device()))
        self.assertEqual(runner.input_names, ["x"])
        self.assertEqual(runner.output_names, ["y"])
        torch.testing.assert_close(first["y"], 2 * x)
        torch.testing.assert_close(second["y"], 2 * (x + 1))

    @REQUIRES_CUDA
    def test_dynamic_shapes_rebind_per_call(self) -> None:
        """Test that a dynamic engine follows the input shape of every call.

        Input: the add engine run on ``n = 4``, then ``n = 2``, then ``n = 4`` again.
        Expected: an ``[n, 2]`` output of threes each time.
        Check: the shape and the values of the three outputs.
        """
        runner = TensorRTEngineRunner(
            build_engine(add_graph(self.tmp_path), {"a": N_PROFILE, "b": N_PROFILE}),
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
    def test_a_view_that_is_not_contiguous_runs_as_its_values(self) -> None:
        """Test that a view that is not contiguous runs as the values it shows.

        Input: the add engine with ``a = base[:, 1:]`` of a ``[4, 3]`` tensor, a view with a
            storage offset and a row stride of 3, and ``b`` of ones.
        Expected: ``a + 1``.
        Check: the output against the sum computed in PyTorch.
        """
        runner = TensorRTEngineRunner(
            build_engine(add_graph(self.tmp_path), {"a": N_PROFILE, "b": N_PROFILE}),
            torch.device("cuda"),
        )
        a = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3)[:, 1:]
        b = torch.ones(4, 2, device="cuda")

        self.assertFalse(a.is_contiguous())
        torch.testing.assert_close(runner.run({"a": a, "b": b})["y"], a + b)

    @REQUIRES_CUDA
    def test_runner_rejects_a_shape_outside_the_profile(self) -> None:
        """Test that a dynamic input outside the profile is refused with the profile's range.

        Input: the add engine, whose axis ``n`` goes from 1 to 8, fed ``n = 0`` (an empty
            frame), then ``n = 9``, then ``n = 4``.
        Expected: a RuntimeError naming the rejected shape and the profile's minimum and
            maximum for both, and the valid call still running.
        Check: the error messages and the last call's output.
        """
        runner = TensorRTEngineRunner(
            build_engine(add_graph(self.tmp_path), {"a": N_PROFILE, "b": N_PROFILE}),
            torch.device("cuda"),
        )
        for n in (0, 9):
            with self.assertRaisesRegex(
                RuntimeError,
                rf"rejected shape \({n}, 2\) for input 'a' .*"
                r"profile min \(1, 2\) max \(8, 2\)",
            ):
                runner.run(
                    {"a": torch.ones(n, 2, device="cuda"), "b": torch.ones(n, 2, device="cuda")}
                )
        ones = torch.ones(4, 2, device="cuda")
        torch.testing.assert_close(runner.run({"a": ones, "b": ones})["y"], ones * 2)

    @REQUIRES_CUDA
    def test_engine_survives_a_reload(self) -> None:
        """Test that a second load of an engine runs after the first one was collected.

        Input: the doubling Gemm engine loaded, run, deleted and collected, then loaded again.
        Expected: the second engine produces the first one's outputs.
        Check: the outputs.
        """
        path = build_engine(doubling_gemm_graph(self.tmp_path), {})
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4).cuda()

        first = TensorRTEngineRunner(path, torch.device("cuda"))
        expected = first.run({"x": x})["y"]
        del first
        gc.collect()

        second = TensorRTEngineRunner(path, torch.device("cuda"))
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
        runner = TensorRTEngineRunner(
            build_engine(doubling_gemm_graph(self.tmp_path), {}), torch.device("cuda")
        )
        x = torch.ones(2, 4, device="cuda")

        with self.assertRaisesRegex(ValueError, r"missing \['x'\], unexpected \['z'\]"):
            runner.run({"z": x})
        with self.assertRaisesRegex(ValueError, r"unexpected \['z'\]"):
            runner.run({"x": x, "z": x})
        with self.assertRaisesRegex(RuntimeError, r"rejected shape \(3, 4\) for input 'x'"):
            runner.run({"x": torch.ones(3, 4, device="cuda")})
        torch.testing.assert_close(runner.run({"x": x})["y"], 2 * x)

    @REQUIRES_CUDA
    def test_runner_rejects_inputs_that_disagree_on_a_shared_axis(self) -> None:
        """Test that inputs that are each inside the profile but disagree are refused.

        Input: the add engine fed ``a`` of ``[3, 2]`` and ``b`` of ``[5, 2]``, then ``[3, 2]`` both.
        Expected: a RuntimeError naming the output and both input shapes, and the consistent
            call still running.
        Check: the error message and the last call's output.
        """
        runner = TensorRTEngineRunner(
            build_engine(add_graph(self.tmp_path), {"a": N_PROFILE, "b": N_PROFILE}),
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
        runner = TensorRTEngineRunner(
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
            TensorRTEngineRunner(self.tmp_path / "missing.engine", torch.device("cpu"))

    @REQUIRES_CUDA
    def test_runner_refuses_a_missing_plugin_library(self) -> None:
        """Test that a plugin library that does not exist is refused before the engine loads.

        Input: the doubling Gemm engine and the path of a plugin library that was never built.
        Expected: a FileNotFoundError naming the library.
        Check: the error message.
        """
        path = build_engine(doubling_gemm_graph(self.tmp_path), {})
        missing = self.tmp_path / "libmissing_plugins.so"

        with self.assertRaisesRegex(FileNotFoundError, "plugin library not found.*libmissing"):
            TensorRTEngineRunner(path, torch.device("cuda"), plugin_libraries=[str(missing)])

    @REQUIRES_CUDA
    def test_pinned_cuda_device_fixes_the_index_of_a_bare_cuda(self) -> None:
        """Test that a CUDA device without an index becomes the current device.

        Input: ``"cuda"`` and ``torch.device("cuda")``.
        Expected: the CUDA device whose index is ``torch.cuda.current_device()``.
        Check: the returned devices.
        """
        current = torch.device("cuda", torch.cuda.current_device())

        self.assertEqual(pinned_cuda_device("cuda"), current)
        self.assertEqual(pinned_cuda_device(torch.device("cuda")), current)

    def test_pinned_cuda_device_keeps_a_given_index(self) -> None:
        """Test that a CUDA device with an index is returned as it is, whatever is current.

        Input: ``"cuda:1"``.
        Expected: ``torch.device("cuda", 1)``.
        Check: the returned device.
        """
        self.assertEqual(pinned_cuda_device("cuda:1"), torch.device("cuda", 1))


if __name__ == "__main__":
    unittest.main()
