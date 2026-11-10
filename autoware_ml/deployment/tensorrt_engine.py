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

"""Run a serialized TensorRT engine on named torch tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_loaded_plugin_libraries: set[Path] = set()


def load_tensorrt_plugins(plugin_paths: Sequence[str | Path]) -> None:
    """Load plugin libraries into the process, once each.

    The libraries register their creators with TensorRT as they load. Loading them through
    ``IPluginRegistry.load_library`` would register them a second time, which fails.

    Args:
        plugin_paths: Plugin libraries, e.g. those of ``autoware_tensorrt_plugins``.

    Raises:
        FileNotFoundError: If a library does not exist.
    """
    import tensorrt  # noqa: F401, loads the libnvinfer the plugin libraries link against

    for plugin_path in plugin_paths:
        path = Path(plugin_path).resolve()
        if path in _loaded_plugin_libraries:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"TensorRT plugin library not found: {path}")
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        _loaded_plugin_libraries.add(path)
        logger.info("Loaded TensorRT plugin library %s", path)


def _torch_dtype(trt_dtype: Any) -> torch.dtype:
    import tensorrt as trt

    dtypes = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
    }
    return dtypes[trt_dtype]


class TensorRTEngine:
    """A TensorRT engine called with named torch tensors, returning its outputs by name.

    The engine is deserialized on the first call, on the CUDA device of the inputs, so it can
    be built from a config before CUDA is initialized. Inputs are read by the engine's input
    names, so a caller may pass a superset; integer inputs are cast to the engine's integer
    type. Outputs are allocated by torch and computed on the current CUDA stream.

    Args:
        engine_path: Serialized engine, e.g. built by ``trtexec``.
        plugin_paths: Plugin libraries the engine needs, loaded before it is deserialized.
    """

    def __init__(self, engine_path: str | Path, plugin_paths: Sequence[str | Path] = ()) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        self.plugin_paths = tuple(Path(path) for path in plugin_paths)
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._input_names: tuple[str, ...] = ()
        self._output_names: tuple[str, ...] = ()
        self._dtypes: dict[str, torch.dtype] = {}

    def __repr__(self) -> str:
        """Name the engine by its file."""
        return f"{type(self).__name__}({str(self.engine_path)!r})"

    @property
    def input_names(self) -> tuple[str, ...]:
        """Names of the engine inputs, in binding order."""
        self._load()
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """Names of the engine outputs, in binding order."""
        self._load()
        return self._output_names

    def _load(self) -> None:
        """Deserialize the engine and create its execution context, once."""
        if self._context is not None:
            return
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")
        load_tensorrt_plugins(self.plugin_paths)
        self._runtime = trt.Runtime(trt_logger)
        self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {self.engine_path}.")
        self._context = self._engine.create_execution_context()
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        is_input = {
            name: self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in names
        }
        self._input_names = tuple(name for name in names if is_input[name])
        self._output_names = tuple(name for name in names if not is_input[name])
        self._dtypes = {name: _torch_dtype(self._engine.get_tensor_dtype(name)) for name in names}
        logger.info(
            "Loaded TensorRT engine %s: inputs %s, outputs %s",
            self.engine_path,
            list(self._input_names),
            list(self._output_names),
        )

    def __call__(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the engine.

        Args:
            tensors: CUDA tensors keyed by name, holding at least every engine input.

        Returns:
            The engine outputs keyed by their names.

        Raises:
            KeyError: If an engine input is missing.
            TypeError: If an input is not on CUDA or needs a cast across kinds of dtype.
            ValueError: If an input shape is outside the engine's optimization profile.
            RuntimeError: If an output shape depends on the data or the execution fails.
        """
        self._load()
        missing = [name for name in self._input_names if name not in tensors]
        if missing:
            raise KeyError(f"{self.engine_path.name} needs the inputs {missing}.")

        context = self._context
        device: torch.device | None = None
        bound_inputs = []
        for name in self._input_names:
            tensor = tensors[name]
            if tensor.device.type != "cuda":
                raise TypeError(f"Input '{name}' of {self.engine_path.name} must be on CUDA.")
            device = tensor.device
            dtype = self._dtypes[name]
            if tensor.dtype != dtype:
                if tensor.dtype.is_floating_point or dtype.is_floating_point:
                    raise TypeError(
                        f"Input '{name}' of {self.engine_path.name} is {tensor.dtype}, the engine "
                        f"takes {dtype}."
                    )
                tensor = tensor.to(dtype)
            tensor = tensor.contiguous()
            bound_inputs.append(tensor)
            if not context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(
                    f"Input '{name}' of {self.engine_path.name} has shape {tuple(tensor.shape)}, "
                    "outside the engine's optimization profile."
                )
            context.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for name in self._output_names:
            shape = tuple(context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Output '{name}' of {self.engine_path.name} has the data-dependent shape "
                    f"{shape}, which this runner does not allocate."
                )
            output = torch.empty(shape, dtype=self._dtypes[name], device=device)
            context.set_tensor_address(name, output.data_ptr())
            outputs[name] = output

        if not context.execute_async_v3(torch.cuda.current_stream(device).cuda_stream):
            raise RuntimeError(f"TensorRT failed to execute {self.engine_path.name}.")
        return outputs


def load_engines(
    engine_dir: str | Path, names: Sequence[str], plugin_paths: Sequence[str | Path] = ()
) -> dict[str, TensorRTEngine]:
    """Return the engine ``<engine_dir>/<name>.engine`` of every exported module, in order.

    Args:
        engine_dir: Directory holding the engines.
        names: Exported module names, in dataflow order.
        plugin_paths: Plugin libraries the engines need.
    """
    return {
        name: TensorRTEngine(Path(engine_dir) / f"{name}.engine", plugin_paths) for name in names
    }
