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

"""TensorRT runner for one graph's engine."""

from __future__ import annotations

import ctypes
import functools
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import tensorrt as trt
import torch

from autoware_ml.deployment.graphs import GraphTensor

logger = logging.getLogger(__name__)

#: The torch dtype of each engine dtype of the same width and kind.
_TORCH_DTYPES: Mapping[trt.DataType, torch.dtype] = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.uint8: torch.uint8,
    trt.bool: torch.bool,
}


@functools.cache
def tensorrt_logger() -> trt.Logger:
    """Return the one TensorRT logger of the process.

    TensorRT keeps a raw pointer to the logger it was given: a logger created per engine load
    is destroyed before the runtime and the engine that log through it.

    Returns:
        The process-wide logger, at warning level.
    """
    return trt.Logger(trt.Logger.WARNING)


@functools.cache
def _load_plugin_library(library: str) -> None:
    """Load one plugin library into the process, once per path; it registers its own plugins.

    The library is loaded by its absolute path: the dynamic loader searches its library
    directories, not the working directory, for a bare file name.

    Args:
        library: Path of the shared library.

    Raises:
        FileNotFoundError: If the library does not exist.
    """
    library_path = Path(library)
    if not library_path.exists():
        raise FileNotFoundError(f"TensorRT plugin library not found: {library_path}")
    ctypes.CDLL(str(library_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger.info("Loaded TensorRT plugin library: %s", library_path)


@functools.cache
def _init_plugin_registry() -> None:
    """Register TensorRT's built-in plugins, once per process."""
    trt.init_libnvinfer_plugins(tensorrt_logger(), "")


def pinned_cuda_device(device: str | torch.device) -> torch.device:
    """Return ``device`` as a CUDA device with its index fixed.

    torch resolves a bare ``"cuda"`` to the current device on every use; an engine stays on
    the device it was deserialized on, so its runner fixes the index once.

    Args:
        device: A CUDA device; ``"cuda"`` is the current device at the call.

    Returns:
        The device with an index.

    Raises:
        ValueError: If ``device`` is not a CUDA device.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"TensorRT requires a CUDA device, got {device}.")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class TensorRTEngineRunner:
    """Run one serialized TensorRT engine with torch CUDA tensors as its I/O buffers.

    Inputs are bound in place, on the device and in the engine's dtype; outputs are allocated
    per call and belong to the caller. The engine runs on the current stream of its device,
    and a runner serves one thread at a time.

    The engine's I/O has to be what the export's builder and ``trtexec`` produce by default:
    tensors on the device in the linear format, none of them a shape tensor. The runner does
    not check this. An input shape outside the engine's profile is refused, an empty input
    included when the profile starts above zero.

    Args:
        engine_path: The serialized engine.
        device: The CUDA device the engine runs on; ``"cuda"`` is the current device at
            construction, and the runner stays on it.
        plugin_libraries: Custom plugin libraries the engine was built with.

    Raises:
        ValueError: If ``device`` is not a CUDA device.
        FileNotFoundError: If the engine file or a plugin library does not exist.
        RuntimeError: If deserialization or context creation fails.
        KeyError: If an engine tensor has a dtype torch has no counterpart of (FP8, INT4).
    """

    def __init__(
        self,
        engine_path: str | Path,
        device: str | torch.device,
        plugin_libraries: Sequence[str] = (),
    ) -> None:
        self.device = pinned_cuda_device(device)
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        for library in plugin_libraries:
            _load_plugin_library(str(library))
        _init_plugin_registry()
        # The runtime has to outlive the engine, so it is kept alongside it.
        self._runtime = trt.Runtime(tensorrt_logger())
        # An engine is bound to the device current when it is deserialized.
        with torch.cuda.device(self.device):
            self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
            if self._engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
            self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(
                f"Failed to create TensorRT execution context for {engine_path.name} "
                "(likely GPU out-of-memory)."
            )
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self._dtypes: dict[str, torch.dtype] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
            self._dtypes[name] = _TORCH_DTYPES[self._engine.get_tensor_dtype(name)]
        logger.info(
            "Loaded TensorRT engine %s (inputs=%s, outputs=%s)",
            engine_path.name,
            self.input_names,
            self.output_names,
        )

    def _as_binding(self, input_name: str, tensor: GraphTensor) -> GraphTensor:
        """Return ``tensor`` on :attr:`device`, contiguous, in the binding's dtype.

        Args:
            input_name: The engine input the tensor is bound to.
            tensor: The caller's tensor, on any device and of any dtype.

        Returns:
            The tensor as TensorRT reads it: float32 data bound to a ``HALF`` input would
            otherwise silently corrupt the activations, and int64 indices feed ``INT32``
            inputs.
        """
        return tensor.to(device=self.device, dtype=self._dtypes[input_name]).contiguous()

    def _bind_input(self, input_name: str, tensor: GraphTensor) -> tuple[int, ...]:
        """Declare an input's shape and address for this call.

        Args:
            input_name: The engine input.
            tensor: The tensor bound to it, as :meth:`_as_binding` returns it.

        Returns:
            The bound shape.

        Raises:
            RuntimeError: If the engine rejects the shape (wrong rank or outside the profile).
        """
        shape = tuple(tensor.shape)
        if not self._context.set_input_shape(input_name, shape):
            declared = tuple(self._engine.get_tensor_shape(input_name))
            allowed = f"engine shape {declared}"
            if -1 in declared:
                low, _, high = self._engine.get_tensor_profile_shape(
                    input_name, self._context.active_optimization_profile
                )
                allowed += f", profile min {tuple(low)} max {tuple(high)}"
            raise RuntimeError(
                f"TensorRT rejected shape {shape} for input {input_name!r} ({allowed})."
            )
        self._context.set_tensor_address(input_name, tensor.data_ptr())
        return shape

    def _allocate_output(
        self, output_name: str, input_shapes: Mapping[str, tuple[int, ...]]
    ) -> GraphTensor:
        """Allocate an output's buffer for the bound input shapes and bind it.

        Args:
            output_name: The engine output.
            input_shapes: The shapes bound to the inputs, for the error messages.

        Returns:
            The empty buffer on :attr:`device`.

        Raises:
            RuntimeError: If the output's shape does not follow from the input shapes: the
                inputs disagree, or the shape depends on the data (``NonZero``-like ops).
        """
        try:
            # TensorRT reports an unresolvable shape as a Dims of negative rank, which the
            # Python binding cannot iterate.
            shape = tuple(self._context.get_tensor_shape(output_name))
        except ValueError:
            raise RuntimeError(
                f"TensorRT could not infer the shape of output {output_name!r} from the input "
                f"shapes {dict(input_shapes)}; the inputs disagree with each other."
            ) from None
        if -1 in shape:
            raise RuntimeError(
                f"Output {output_name!r} has the data-dependent shape {shape}; the runner "
                "allocates outputs from the input shapes and does not support such engines."
            )
        buffer = torch.empty(shape, dtype=self._dtypes[output_name], device=self.device)
        self._context.set_tensor_address(output_name, buffer.data_ptr())
        return buffer

    def run(self, inputs: Mapping[str, GraphTensor]) -> dict[str, GraphTensor]:
        """Run the engine once.

        Args:
            inputs: The engine inputs by name, on any device and of any dtype.

        Returns:
            The outputs by name, newly allocated on :attr:`device`.

        Raises:
            ValueError: If the input names are not exactly the engine's inputs.
            RuntimeError: If TensorRT rejects an input shape, cannot resolve an output shape,
                or the execution fails.
        """
        if set(inputs) != set(self.input_names):
            missing = sorted(set(self.input_names) - set(inputs))
            unexpected = sorted(set(inputs) - set(self.input_names))
            raise ValueError(
                f"TensorRT engine inputs are {self.input_names}; "
                f"missing {missing}, unexpected {unexpected}."
            )
        with torch.cuda.device(self.device):
            # The bound tensors stay referenced until the execution has synchronized.
            bound = [self._as_binding(name, inputs[name]) for name in self.input_names]
            input_shapes = {
                name: self._bind_input(name, tensor)
                for name, tensor in zip(self.input_names, bound, strict=True)
            }
            outputs = {
                name: self._allocate_output(name, input_shapes) for name in self.output_names
            }
            stream = torch.cuda.current_stream(self.device)
            if not self._context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 returned failure status.")
            stream.synchronize()
        return outputs
