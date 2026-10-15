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

"""ONNX graph modifiers used during deployment export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _import_onnx_tooling() -> tuple[Any, Any, Any, Any, Any]:
    """Import ONNX tooling lazily for deploy-only paths."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    return np, onnx, TensorProto, helper, numpy_helper


def _set_last_dim(value_info: Any, value: int) -> None:
    """Set the last dimension of a ValueInfoProto to a fixed value."""
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape") or len(tensor_type.shape.dim) == 0:
        return
    last_dim = tensor_type.shape.dim[-1]
    last_dim.ClearField("dim_param")
    last_dim.dim_value = value


@dataclass(slots=True)
class TopKConstantKModifier:
    """Replace a TopK node K input with a compile-time constant.

    TensorRT rejects exported ``argsort(...)[..., :k]`` patterns when ONNX keeps
    the internal TopK K tied to the flattened source dimension. This modifier
    rewrites the selected TopK node to use a fixed K so TensorRT can build the
    engine when ``k`` is below the parser limit.
    """

    k: int
    node_name_substring: str | None = None
    output_path: str | Path | None = None

    def _select_topk_node(self, model: Any) -> Any:
        matches = []
        for node in model.graph.node:
            if node.op_type != "TopK":
                continue
            node_name = node.name or ""
            if self.node_name_substring is not None and self.node_name_substring not in node_name:
                continue
            matches.append(node)

        if not matches:
            filter_text = (
                f" matching '{self.node_name_substring}'"
                if self.node_name_substring is not None
                else ""
            )
            raise RuntimeError(f"Could not find TopK node{filter_text} in exported ONNX graph.")
        if len(matches) > 1:
            node_names = [node.name for node in matches]
            raise RuntimeError(
                f"Expected exactly one TopK node to patch, found {len(matches)}: {node_names}"
            )
        return matches[0]

    def modify(self, onnx_path: str | Path) -> Path:
        """Rewrite the target TopK node to use the configured fixed K."""
        np, onnx, TensorProto, helper, numpy_helper = _import_onnx_tooling()

        source_path = Path(onnx_path)
        model = onnx.load(source_path.as_posix())
        graph = model.graph
        topk = self._select_topk_node(model)
        if len(topk.input) < 2:
            raise RuntimeError(
                f"TopK node '{topk.name or '<unnamed>'}' does not have the expected K input."
            )

        constant_name = f"{topk.name or 'TopK'}_K"

        stale_initializer_indices = [
            index
            for index, initializer in enumerate(graph.initializer)
            if initializer.name == constant_name
        ]
        for index in reversed(stale_initializer_indices):
            del graph.initializer[index]

        graph.initializer.append(
            numpy_helper.from_array(np.array([self.k], dtype=np.int64), name=constant_name)
        )
        graph.value_info.append(
            helper.make_tensor_value_info(constant_name, TensorProto.INT64, [1])
        )
        topk.input[1] = constant_name

        value_infos = list(graph.value_info) + list(graph.output)
        for value_info in value_infos:
            if value_info.name == topk.output[0]:
                _set_last_dim(value_info, self.k)
            elif value_info.name == topk.output[1]:
                _set_last_dim(value_info, self.k)
                value_info.type.tensor_type.elem_type = TensorProto.INT64

        destination_path = Path(self.output_path) if self.output_path is not None else source_path
        onnx.save_model(model, destination_path.as_posix())
        return destination_path

    __call__ = modify


@dataclass(slots=True)
class AttentionScaleToDivModifier:
    """Rewrite exported attention query scaling to the TensorRT-friendly form.

    PyTorch exports ``nn.MultiheadAttention`` query scaling as ``Mul(q, scale)``
    before the score ``MatMul``. The TransFusion deployment graph uses the
    equivalent ``Div(q, 1 / scale)`` form, which TensorRT handles more reliably
    for this TransHead graph.
    """

    node_name_substring: str = "/bbox_head/decoder"
    fail_if_no_match: bool = True
    output_path: str | Path | None = None

    @staticmethod
    def _constant_tensor_by_output(model: Any, numpy_helper: Any) -> dict[str, Any]:
        constants = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in model.graph.initializer
        }
        for node in model.graph.node:
            if node.op_type != "Constant" or len(node.output) != 1:
                continue
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)
                    break
        return constants

    @staticmethod
    def _node_consumers(model: Any) -> dict[str, list[Any]]:
        consumers: dict[str, list[Any]] = {}
        for node in model.graph.node:
            for input_name in node.input:
                consumers.setdefault(input_name, []).append(node)
        return consumers

    def modify(self, onnx_path: str | Path) -> Path:
        """Rewrite attention score scaling Mul nodes as equivalent Div nodes."""
        np, onnx, _TensorProto, _helper, numpy_helper = _import_onnx_tooling()

        source_path = Path(onnx_path)
        model = onnx.load(source_path.as_posix())
        constants = self._constant_tensor_by_output(model, numpy_helper)
        consumers = self._node_consumers(model)
        rewritten = 0

        for node in model.graph.node:
            if node.op_type != "Mul":
                continue
            if self.node_name_substring not in (node.name or ""):
                continue
            if len(node.input) != 2:
                continue
            if not any(
                consumer.op_type == "MatMul" for consumer in consumers.get(node.output[0], [])
            ):
                continue

            constant_inputs = [
                (index, constants[input_name])
                for index, input_name in enumerate(node.input)
                if input_name in constants and constants[input_name].size == 1
            ]
            if len(constant_inputs) != 1:
                continue

            constant_index, scale_tensor = constant_inputs[0]
            scale = float(scale_tensor.reshape(-1)[0])
            if not np.isfinite(scale) or scale == 0.0:
                continue

            tensor_input = node.input[1 - constant_index]
            divisor_name = f"{node.name or 'attention_scale'}_Divisor"
            divisor = np.array(1.0 / scale, dtype=scale_tensor.dtype)
            stale_initializer_indices = [
                index
                for index, initializer in enumerate(model.graph.initializer)
                if initializer.name == divisor_name
            ]
            for index in reversed(stale_initializer_indices):
                del model.graph.initializer[index]
            model.graph.initializer.append(numpy_helper.from_array(divisor, name=divisor_name))

            node.op_type = "Div"
            del node.input[:]
            node.input.extend([tensor_input, divisor_name])
            rewritten += 1

        if rewritten == 0 and self.fail_if_no_match:
            raise RuntimeError(
                f"Could not find attention scale Mul nodes matching '{self.node_name_substring}' to rewrite."
            )

        destination_path = Path(self.output_path) if self.output_path is not None else source_path
        onnx.save_model(model, destination_path.as_posix())
        return destination_path

    __call__ = modify


@dataclass(slots=True)
class TransHeadTensorRTModifier:
    """Apply the TransHead ONNX graph patches needed for TensorRT deployment."""

    k: int
    topk_node_name_substring: str | None = "/bbox_head/TopK"
    attention_node_name_substring: str = "/bbox_head/decoder"
    output_path: str | Path | None = None

    def modify(self, onnx_path: str | Path) -> Path:
        """Apply TransHead graph fixes in a stable order."""
        destination_path = (
            Path(self.output_path) if self.output_path is not None else Path(onnx_path)
        )
        patched_path = TopKConstantKModifier(
            k=self.k,
            node_name_substring=self.topk_node_name_substring,
            output_path=destination_path,
        ).modify(onnx_path)
        return AttentionScaleToDivModifier(
            node_name_substring=self.attention_node_name_substring,
            fail_if_no_match=False,
            output_path=destination_path,
        ).modify(patched_path)

    __call__ = modify
