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

"""Tests for ONNX graph modifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autoware_ml.utils.onnx_modifiers import AttentionScaleToDivModifier, TopKConstantKModifier

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def _write_topk_graph(path: Path) -> None:
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 4096])
    dynamic_k = helper.make_tensor_value_info("dynamic_k", TensorProto.INT64, [1])
    top_values = helper.make_tensor_value_info("top_values", TensorProto.FLOAT, [1, 4096])
    top_indices = helper.make_tensor_value_info("top_indices", TensorProto.INT64, [1, 4096])
    topk = helper.make_node(
        "TopK",
        inputs=["scores", "dynamic_k"],
        outputs=["top_values", "top_indices"],
        name="/bbox_head/TopK",
        axis=-1,
        largest=1,
        sorted=1,
    )
    graph = helper.make_graph(
        nodes=[topk],
        name="topk_graph",
        inputs=[scores, dynamic_k],
        outputs=[top_values, top_indices],
        initializer=[numpy_helper.from_array(np.array([4096], dtype=np.int64), name="dynamic_k")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save_model(model, path.as_posix())


def _write_attention_scale_graph(path: Path) -> None:
    query = helper.make_tensor_value_info("query", TensorProto.FLOAT, [1, 8, 200, 16])
    key = helper.make_tensor_value_info("key", TensorProto.FLOAT, [1, 8, 16, 200])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 8, 200, 200])
    scale = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["scale"],
        name="/bbox_head/decoder.0/self_attn/Constant_18",
        value=numpy_helper.from_array(np.array(0.25, dtype=np.float32)),
    )
    mul = helper.make_node(
        "Mul",
        inputs=["query", "scale"],
        outputs=["scaled_query"],
        name="/bbox_head/decoder.0/self_attn/Mul",
    )
    matmul = helper.make_node(
        "MatMul",
        inputs=["scaled_query", "key"],
        outputs=["scores"],
        name="/bbox_head/decoder.0/self_attn/MatMul_1",
    )
    graph = helper.make_graph(
        nodes=[scale, mul, matmul],
        name="attention_scale_graph",
        inputs=[query, key],
        outputs=[scores],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save_model(model, path.as_posix())


def _last_dim(value_info: onnx.ValueInfoProto) -> int:
    return value_info.type.tensor_type.shape.dim[-1].dim_value


def test_topk_constant_k_modifier_rewrites_target_node(tmp_path: Path) -> None:
    onnx_path = tmp_path / "transhead.onnx"
    _write_topk_graph(onnx_path)

    modified_path = TopKConstantKModifier(k=200, node_name_substring="/bbox_head/TopK").modify(
        onnx_path
    )

    assert modified_path == onnx_path

    model = onnx.load(modified_path.as_posix())
    topk_nodes = [node for node in model.graph.node if node.op_type == "TopK"]
    assert len(topk_nodes) == 1

    topk = topk_nodes[0]
    assert topk.input[1] == "/bbox_head/TopK_K"

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    assert "/bbox_head/TopK_K" in initializers
    np.testing.assert_array_equal(
        numpy_helper.to_array(initializers["/bbox_head/TopK_K"]),
        np.array([200], dtype=np.int64),
    )

    outputs = {output.name: output for output in model.graph.output}
    assert _last_dim(outputs["top_values"]) == 200
    assert _last_dim(outputs["top_indices"]) == 200
    assert outputs["top_indices"].type.tensor_type.elem_type == TensorProto.INT64


def test_attention_scale_to_div_modifier_rewrites_attention_mul(tmp_path: Path) -> None:
    onnx_path = tmp_path / "transhead_attention.onnx"
    _write_attention_scale_graph(onnx_path)

    modified_path = AttentionScaleToDivModifier().modify(onnx_path)

    assert modified_path == onnx_path

    model = onnx.load(modified_path.as_posix())
    nodes_by_name = {node.name: node for node in model.graph.node}
    scale_node = nodes_by_name["/bbox_head/decoder.0/self_attn/Mul"]
    assert scale_node.op_type == "Div"
    assert list(scale_node.input) == ["query", "/bbox_head/decoder.0/self_attn/Mul_Divisor"]

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    np.testing.assert_array_equal(
        numpy_helper.to_array(initializers["/bbox_head/decoder.0/self_attn/Mul_Divisor"]),
        np.array(4.0, dtype=np.float32),
    )
