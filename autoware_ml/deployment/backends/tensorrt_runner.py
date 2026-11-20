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

"""TensorRT runner for one graph."""

from __future__ import annotations

import ctypes
import functools
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

from autoware_ml.deployment.graphs import GraphTensor

logger = logging.getLogger(__name__)


@functools.cache
def trt_logger() -> trt.Logger:
    """Return the one TensorRT logger of the process.

    TensorRT keeps a raw pointer to the logger it was given: a logger created per engine load
    is destroyed before the runtime and the engine that log through it.

    Returns:
        The process-wide logger, at warning level.
    """
    return trt.Logger(trt.Logger.WARNING)


@functools.cache
def _load_plugin_library(library: str) -> None:
    """Load one plugin library into the process, once per path.

    Args:
        library: Path of the shared library.

    Raises:
        FileNotFoundError: If the library does not exist.
    """
    library_path = Path(library)
    if not library_path.exists():
        raise FileNotFoundError(f"TensorRT plugin library not found: {library_path}")
    ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    logger.info("Loaded TensorRT plugin library: %s", library_path)


@functools.cache
def _init_plugin_registry() -> None:
    """Register the built-in and the loaded plugins with TensorRT, once per process."""
    trt.init_libnvinfer_plugins(trt_logger(), "")


def load_trt_engine(
    engine_path: str | Path, device: torch.device, plugin_libraries: Sequence[str] = ()
) -> tuple[trt.Runtime, trt.ICudaEngine, trt.IExecutionContext]:
    """Deserialize a TensorRT engine on ``device``.

    Args:
        engine_path: The serialized engine.
        device: The CUDA device the engine is bound to.
        plugin_libraries: Custom plugin libraries the engine was built with.

    Returns:
        The runtime, the engine it deserialized and one execution context of the engine; the
        runtime has to outlive the engine.

    Raises:
        FileNotFoundError: If the engine file does not exist.
        RuntimeError: If deserialization or context creation fails.
    """
    engine_path = Path(engine_path)
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    for library in plugin_libraries:
        _load_plugin_library(str(library))
    _init_plugin_registry()
    runtime = trt.Runtime(trt_logger())
    # An engine is bound to the device current when it is deserialized.
    with torch.cuda.device(device):
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(
            f"Failed to create TensorRT execution context for {engine_path.name} "
            "(likely GPU out-of-memory)."
        )
    return runtime, engine, context


def list_trt_io_names(engine: trt.ICudaEngine) -> tuple[list[str], list[str]]:
    """List the engine's tensor names by I/O mode.

    Args:
        engine: The deserialized engine.

    Returns:
        The input names and the output names, each in TensorRT tensor-index order.
    """
    inputs: list[str] = []
    outputs: list[str] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return inputs, outputs


def _trt_dtype_to_torch(trt_dtype: trt.DataType) -> torch.dtype:
    """Map a TensorRT dtype to torch; an unknown one raises instead of mis-sizing the buffer.

    Args:
        trt_dtype: The dtype of an engine tensor.

    Returns:
        The torch dtype of the same width and kind.
    """
    numpy_dtype = np.dtype(trt.nptype(trt_dtype))
    return torch.from_numpy(np.zeros(0, dtype=numpy_dtype)).dtype


class TensorRTModuleRunner:
    """Run one serialized TensorRT engine with torch CUDA tensors as its I/O buffers.

    Inputs are bound in place, on the device and in the engine's dtype; outputs are allocated
    per call and belong to the caller. The engine runs on the current stream of its device,
    and a runner serves one thread at a time.

    Args:
        engine_path: The serialized engine.
        device: The CUDA device the engine runs on.
        plugin_libraries: Custom plugin libraries the engine was built with.

    Raises:
        ValueError: If ``device`` is not a CUDA device.
    """

    def __init__(
        self, engine_path: str | Path, device: torch.device, plugin_libraries: Sequence[str] = ()
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"TensorRT requires a CUDA device, got {self.device}.")
        self.runtime, self.engine, self.context = load_trt_engine(
            engine_path, self.device, plugin_libraries
        )
        self.input_names, self.output_names = list_trt_io_names(self.engine)
        logger.info(
            "Loaded TensorRT engine %s (inputs=%s, outputs=%s)",
            Path(engine_path).name,
            self.input_names,
            self.output_names,
        )

    def _as_binding(self, name: str, tensor: GraphTensor) -> GraphTensor:
        """Return ``tensor`` on :attr:`device`, contiguous, in the binding's dtype.

        Args:
            name: The engine input the tensor is bound to.
            tensor: The caller's tensor, on any device and of any dtype.

        Returns:
            The tensor as TensorRT reads it; float32 data bound to a ``HALF`` input would
            otherwise silently corrupt the activations.
        """
        target_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
        if tensor.dtype != target_dtype:
            logger.debug("[trt-io] casting tensor %r: %s -> %s", name, tensor.dtype, target_dtype)
            tensor = tensor.to(target_dtype)
        return tensor.to(self.device).contiguous()

    def _bind_input(self, name: str, tensor: GraphTensor) -> None:
        """Declare an input's shape and address for this call.

        Args:
            name: The engine input.
            tensor: The tensor bound to it, as :meth:`_as_binding` returns it.

        Raises:
            RuntimeError: If the engine rejects the shape (wrong rank or outside the profile).
        """
        shape = tuple(tensor.shape)
        if not self.context.set_input_shape(name, shape):
            declared = tuple(self.engine.get_tensor_shape(name))
            allowed = f"engine shape {declared}"
            if -1 in declared:
                low, _, high = self.engine.get_tensor_profile_shape(
                    name, self.context.active_optimization_profile
                )
                allowed += f", profile min {tuple(low)} max {tuple(high)}"
            raise RuntimeError(f"TensorRT rejected shape {shape} for input {name!r} ({allowed}).")
        self.context.set_tensor_address(name, tensor.data_ptr())

    def _allocate_output(
        self, name: str, input_shapes: Mapping[str, tuple[int, ...]]
    ) -> GraphTensor:
        """Allocate an output's buffer for the bound input shapes and bind it.

        Args:
            name: The engine output.
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
            shape = tuple(self.context.get_tensor_shape(name))
        except ValueError:
            raise RuntimeError(
                f"TensorRT could not infer the shape of output {name!r} from the input shapes "
                f"{dict(input_shapes)}; the inputs disagree with each other."
            ) from None
        if -1 in shape:
            raise RuntimeError(
                f"Output {name!r} has the data-dependent shape {shape}; the runner allocates "
                "outputs from the input shapes and does not support such engines."
            )
        buffer = torch.empty(
            shape, dtype=_trt_dtype_to_torch(self.engine.get_tensor_dtype(name)), device=self.device
        )
        self.context.set_tensor_address(name, buffer.data_ptr())
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
        bound = {name: self._as_binding(name, inputs[name]) for name in self.input_names}
        with torch.cuda.device(self.device):
            for name, tensor in bound.items():
                self._bind_input(name, tensor)
            input_shapes = {name: tuple(tensor.shape) for name, tensor in bound.items()}
            outputs = {
                name: self._allocate_output(name, input_shapes) for name in self.output_names
            }
            stream = torch.cuda.current_stream(self.device)
            if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 returned failure status.")
            stream.synchronize()
        return outputs
