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

"""Explicit-precision conversion for exported ONNX models.

``deploy.onnx.precision`` controls the float dtypes baked into the exported ONNX.
The TensorRT stage always builds strongly typed, so those dtypes are what the
engine uses: the exported ONNX alone decides the engine's numerics.

All float network inputs/outputs always stay FP32, so inference code works
with all model precisions without interface changes.
"""

from __future__ import annotations

import logging
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import onnx

# cspell:ignore onnxconverter
from onnxconverter_common import float16

logger = logging.getLogger(__name__)


class OnnxPrecision(StrEnum):
    """
    Float precision baked into an exported ONNX.

    Attributes:
      FP32: Export the model unchanged.
      FP16: Convert weights and internal tensors to half precision, keeping graph I/O fp32.
    """

    FP32 = auto()
    FP16 = auto()


def resolve_onnx_precision(onnx_cfg: Any) -> OnnxPrecision:
    """Read and validate ``deploy.onnx.precision`` (default fp32, the historical behavior)."""
    precision = str(onnx_cfg.get("precision", OnnxPrecision.FP32))
    try:
        return OnnxPrecision(precision)
    except ValueError:
        raise ValueError(
            f"Unsupported deploy.onnx.precision '{precision}'. Supported values: "
            f"{', '.join(OnnxPrecision)}. Precisions that require calibration (e.g. int8) "
            "are out of scope for export-time conversion."
        ) from None


def validate_module_onnx_precision(module: Any, onnx_cfg: Any) -> None:
    """Reject export modules whose declared precision requirement is not configured."""
    actual = resolve_onnx_precision(onnx_cfg)
    for submodule in module.modules():
        required = getattr(submodule, "required_onnx_precision", None)
        if required is not None and str(required) != actual:
            raise ValueError(
                f"{type(submodule).__name__} requires deploy.onnx.precision='{required}', "
                f"but the effective module config uses '{actual}'. Align the model's export "
                "options with deploy.onnx.precision."
            )


def should_convert_precision(onnx_cfg: Any) -> bool:
    """Return whether the exported ONNX needs a precision conversion pass."""
    return resolve_onnx_precision(onnx_cfg) != OnnxPrecision.FP32


def convert_onnx_precision(onnx_path: str | Path, precision: OnnxPrecision) -> Path:
    """Convert an exported fp32 ONNX to the requested explicit precision, in place."""
    if precision == OnnxPrecision.FP16:
        return _convert_to_fp16(Path(onnx_path))
    raise ValueError(f"No conversion implemented for precision '{precision}'.")


def _convert_to_fp16(onnx_path: Path) -> Path:
    model = onnx.load(onnx_path.as_posix())

    # The cast-fixup passes below only handle the top-level graph. No autoware-ml model uses
    # subgraphs, so refuse them explicitly instead of converting them incorrectly.
    subgraph_nodes = [
        node.name or node.op_type
        for node in model.graph.node
        for attribute in node.attribute
        if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    if subgraph_nodes:
        raise NotImplementedError(
            "fp16 conversion does not support models with subgraphs (If/Loop/Scan); "
            f"found: {', '.join(subgraph_nodes)}"
        )

    # keep_io_types keeps the graph boundary fp32 via cast layers inside the model, so engine
    # I/O and consumers' buffers stay fp32 regardless of the internal precision.
    #
    # The op block list must be empty: custom plugin ops negotiate their own precisions and
    # support fp16 I/O natively, and the default block list creates fp32 islands whose boundary
    # casts the converter places incorrectly around ops it cannot shape-infer.
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=[],
        disable_shape_infer=True,  # custom plugin ops break ONNX shape inference
    )

    n_rewired = _rewire_dual_use_outputs(converted)
    n_recast = _retarget_stray_float_casts(converted)
    logger.info(
        "Converted %s to fp16 with fp32 graph I/O (rewired %d dual-use output consumer(s), "
        "retargeted %d pre-existing float cast(s))",
        onnx_path,
        n_rewired,
        n_recast,
    )

    onnx.save(converted, onnx_path.as_posix())
    return onnx_path


def _rewire_dual_use_outputs(model: Any) -> int:
    """Rewire internal consumers of fp32 graph outputs to the pre-cast fp16 tensor.

    A graph output can double as an internal input (e.g. PTv3's point_feat_0 feeds the next
    encoder stage). keep_io_types casts such outputs back to fp32, which would re-enter the fp16
    interior as fp32 and fail a strongly typed parse; internal consumers must read the fp16
    tensor the output cast was fed from instead.
    """
    graph_outputs = {output.name for output in model.graph.output}
    producers = {output: node for node in model.graph.node for output in node.output}
    n_rewired = 0
    for name in graph_outputs:
        producer = producers.get(name)
        if producer is None or producer.op_type != "Cast":
            continue
        cast_target = next((a.i for a in producer.attribute if a.name == "to"), None)
        if cast_target != onnx.TensorProto.FLOAT:
            continue
        pre_cast_tensor = producer.input[0]
        consumers = [
            (node, index)
            for node in model.graph.node
            if node is not producer
            for index, node_input in enumerate(node.input)
            if node_input == name
        ]
        for node, index in consumers:
            node.input[index] = pre_cast_tensor
        n_rewired += len(consumers)
    return n_rewired


def _retarget_stray_float_casts(model: Any) -> int:
    """Retarget pre-existing ``Cast(to=FLOAT)`` nodes (and float ConstantOfShape values) to fp16.

    ``float16.convert_float_to_float16`` leaves the ``to`` attribute of Cast nodes untouched.
    Exports commonly contain fp32->fp32 no-op casts (e.g. the post-softmax ``.to(query.dtype)``
    in attention). After conversion these would cast fp16->fp32, feeding the wrong precision to
    consumers. Casts that produce a graph output are the fp32 boundary and stay untouched.
    """
    graph_outputs = {output.name for output in model.graph.output}
    n_recast = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            if any(output in graph_outputs for output in node.output):
                continue
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                    attribute.i = onnx.TensorProto.FLOAT16
                    n_recast += 1
        elif node.op_type == "ConstantOfShape":
            for attribute in node.attribute:
                if attribute.name == "value" and attribute.t.data_type == onnx.TensorProto.FLOAT:
                    value = onnx.numpy_helper.to_array(attribute.t)
                    attribute.t.CopyFrom(onnx.numpy_helper.from_array(value.astype("float16")))
    return n_recast
