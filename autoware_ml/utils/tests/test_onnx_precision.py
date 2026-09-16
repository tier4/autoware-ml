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

"""Tests for export-time ONNX precision conversion."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from omegaconf import OmegaConf
from onnx import TensorProto, helper, numpy_helper

from autoware_ml.utils.onnx_precision import (
    OnnxPrecision,
    convert_onnx_precision,
    resolve_onnx_precision,
    should_convert_precision,
    validate_module_onnx_precision,
)


def _make_fp32_model() -> onnx.ModelProto:
    """A minimal fp32 graph exercising the conversion pitfalls.

    MatMul with an fp32 weight initializer, followed by a pre-existing no-op Cast(to=FLOAT)
    (the pattern torch exports for ``.to(query.dtype)``), whose output is both a graph output
    and an internal input to a second MatMul — mirroring PTv3's dual-use point_feat_0.
    """
    weight = numpy_helper.from_array(np.eye(4, dtype=np.float32), name="weight")
    nodes = [
        helper.make_node("MatMul", ["x", "weight"], ["mm0"], name="mm0"),
        helper.make_node("Cast", ["mm0"], ["feat"], name="noop_cast", to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["feat", "weight"], ["out"], name="mm1"),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])],
        outputs=[
            helper.make_tensor_value_info("feat", TensorProto.FLOAT, [None, 4]),
            helper.make_tensor_value_info("out", TensorProto.FLOAT, [None, 4]),
        ],
        initializer=[weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_resolve_precision_defaults_to_fp32() -> None:
    assert resolve_onnx_precision(OmegaConf.create({})) is OnnxPrecision.FP32
    assert not should_convert_precision(OmegaConf.create({}))


def test_resolve_precision_reads_config_string() -> None:
    """The enum's values are the spellings used in deploy.onnx.precision."""
    cfg = OmegaConf.create({"precision": "fp16"})
    assert resolve_onnx_precision(cfg) is OnnxPrecision.FP16
    assert should_convert_precision(cfg)


def test_resolve_precision_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="int8"):
        resolve_onnx_precision(OmegaConf.create({"precision": "int8"}))


def test_validate_module_precision_requirement() -> None:
    class PrecisionAwareModule:
        required_onnx_precision = "fp16"

        def modules(self):
            return [self]

    module = PrecisionAwareModule()
    validate_module_onnx_precision(module, OmegaConf.create({"precision": "fp16"}))
    with pytest.raises(ValueError, match="requires deploy.onnx.precision='fp16'"):
        validate_module_onnx_precision(module, OmegaConf.create({"precision": "fp32"}))


def test_fp16_conversion(tmp_path) -> None:
    onnx_path = tmp_path / "tiny.onnx"
    onnx.save(_make_fp32_model(), onnx_path.as_posix())

    result = convert_onnx_precision(onnx_path, OnnxPrecision.FP16)
    model = onnx.load(result.as_posix())

    initializer_types = {i.name: i.data_type for i in model.graph.initializer}
    assert initializer_types["weight"] == TensorProto.FLOAT16

    # The graph boundary stays fp32: reduced precision is internal, behind cast layers, so
    # consumers keep their fp32 buffers.
    for value_info in list(model.graph.input) + list(model.graph.output):
        assert value_info.type.tensor_type.elem_type == TensorProto.FLOAT

    # 'feat' is both a graph output and an internal input of the second MatMul. The output must
    # be produced by an fp16->fp32 cast, while the internal consumer must read the pre-cast fp16
    # tensor - otherwise a strongly typed parse rejects the mixed Float/Half MatMul.
    producers = {output: node for node in model.graph.node for output in node.output}
    feat_producer = producers["feat"]
    assert feat_producer.op_type == "Cast"
    assert next(a.i for a in feat_producer.attribute if a.name == "to") == TensorProto.FLOAT
    internal_consumers = [
        n for n in model.graph.node if "feat" in n.input and n is not feat_producer
    ]
    assert not internal_consumers, "internal consumers must be rewired to the fp16 tensor"
    mm1 = next(n for n in model.graph.node if n.name == "mm1")
    assert mm1.input[0] == feat_producer.input[0]

    onnx.checker.check_model(model)


def test_fp16_conversion_rejects_subgraphs(tmp_path) -> None:
    """Models with If/Loop/Scan subgraphs are refused: the cast fixups only handle the top level."""
    const = helper.make_node(
        "Constant",
        [],
        ["y"],
        value=helper.make_tensor("v", TensorProto.FLOAT, [1], [1.0]),
    )
    branch = helper.make_graph(
        [const],
        "branch",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    if_node = helper.make_node(
        "If", ["cond"], ["y"], name="if0", then_branch=branch, else_branch=branch
    )
    graph = helper.make_graph(
        [if_node],
        "with_subgraph",
        inputs=[helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_path = tmp_path / "subgraph.onnx"
    onnx.save(model, onnx_path.as_posix())

    with pytest.raises(NotImplementedError, match="if0"):
        convert_onnx_precision(onnx_path, OnnxPrecision.FP16)
