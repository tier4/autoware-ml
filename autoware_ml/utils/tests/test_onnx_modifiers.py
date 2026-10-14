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

from autoware_ml.utils.onnx_modifiers import (
    AttentionScaleToDivModifier,
    IdentityEliminationModifier,
    TransHeadTensorRTModifier,
    TopKConstantKModifier,
)

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


def _write_identity_alias_graph(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])
    bias_out = helper.make_tensor_value_info("bias_out", TensorProto.FLOAT, [2])
    initializers = [
        numpy_helper.from_array(np.array([1.5, 0.5], dtype=np.float32), name="scale"),
        numpy_helper.from_array(np.array([0.1, -0.2], dtype=np.float32), name="bias"),
        numpy_helper.from_array(np.array([0.3, 0.7], dtype=np.float32), name="mean"),
        numpy_helper.from_array(np.array([2.0, 4.0], dtype=np.float32), name="var"),
    ]
    nodes = [
        helper.make_node("Identity", ["scale"], ["scale_alias"], name="/bn/Identity_scale"),
        helper.make_node("Identity", ["bias"], ["bias_alias"], name="/bn/Identity_bias"),
        helper.make_node("Identity", ["bias"], ["bias_out"], name="/bn/Identity_output"),
        helper.make_node("Identity", ["x"], ["x_alias"], name="/bn/Identity_input"),
        helper.make_node(
            "BatchNormalization",
            ["x_alias", "scale_alias", "bias_alias", "mean", "var"],
            ["y"],
            name="/bn/BatchNormalization",
            epsilon=1e-5,
        ),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="identity_alias_graph",
        inputs=[x],
        outputs=[y, bias_out],
        initializer=initializers,
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


def test_identity_elimination_modifier_rewires_consumers(tmp_path: Path) -> None:
    onnx_path = tmp_path / "identity_alias.onnx"
    _write_identity_alias_graph(onnx_path)
    output_path = tmp_path / "identity_alias_stripped.onnx"

    modified_path = IdentityEliminationModifier(output_path=output_path).modify(onnx_path)

    assert modified_path == output_path
    # The source file is left untouched when an explicit output path is given.
    assert len(onnx.load(onnx_path.as_posix()).graph.node) == 5

    model = onnx.load(modified_path.as_posix())
    onnx.checker.check_model(model)
    nodes_by_name = {node.name: node for node in model.graph.node}
    assert set(nodes_by_name) == {"/bn/Identity_output", "/bn/BatchNormalization"}
    # Consumers are rewired to the Identity inputs, graph input and initializers
    # alike; the Identity producing a graph output is kept as it was.
    assert list(nodes_by_name["/bn/BatchNormalization"].input) == [
        "x",
        "scale",
        "bias",
        "mean",
        "var",
    ]
    assert list(nodes_by_name["/bn/Identity_output"].input) == ["bias"]
    assert [output.name for output in model.graph.output] == ["y", "bias_out"]
    assert {initializer.name for initializer in model.graph.initializer} == {
        "scale",
        "bias",
        "mean",
        "var",
    }


def test_identity_elimination_modifier_is_lossless(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    onnx_path = tmp_path / "identity_alias.onnx"
    _write_identity_alias_graph(onnx_path)
    stripped_path = IdentityEliminationModifier(
        output_path=tmp_path / "identity_alias_stripped.onnx"
    ).modify(onnx_path)

    x = np.random.default_rng(0).standard_normal((1, 2, 4, 4)).astype(np.float32)
    outputs = {}
    for path in (onnx_path, stripped_path):
        session = onnxruntime.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
        outputs[path] = session.run(None, {"x": x})

    for before, after in zip(outputs[onnx_path], outputs[stripped_path]):
        np.testing.assert_array_equal(before, after)


def test_identity_elimination_modifier_keeps_graph_without_aliases(tmp_path: Path) -> None:
    onnx_path = tmp_path / "transhead.onnx"
    _write_topk_graph(onnx_path)
    before = onnx.load(onnx_path.as_posix()).SerializeToString()

    modified_path = IdentityEliminationModifier().modify(onnx_path)

    assert modified_path == onnx_path
    assert onnx.load(modified_path.as_posix()).SerializeToString() == before


def test_identity_elimination_modifier_collapses_identity_chains(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["scale"], ["scale_1"], name="Identity_1"),
            helper.make_node("Identity", ["scale_1"], ["scale_2"], name="Identity_2"),
            helper.make_node("Add", ["x", "scale_2"], ["y"], name="Add"),
        ],
        name="identity_chain_graph",
        inputs=[x],
        outputs=[y],
        initializer=[numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="scale")],
    )
    onnx_path = tmp_path / "identity_chain.onnx"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), onnx_path
    )

    model = onnx.load(IdentityEliminationModifier().modify(onnx_path).as_posix())

    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == ["Add"]
    assert list(model.graph.node[0].input) == ["x", "scale"]


def test_trans_head_modifier_removes_identity_nodes_first(tmp_path: Path) -> None:
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 4096])
    top_values = helper.make_tensor_value_info("top_values", TensorProto.FLOAT, [1, 4096])
    top_indices = helper.make_tensor_value_info("top_indices", TensorProto.INT64, [1, 4096])
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["scores"], ["scores_alias"], name="Identity_scores"),
            helper.make_node("Identity", ["dynamic_k"], ["k_alias"], name="Identity_k"),
            helper.make_node(
                "TopK",
                ["scores_alias", "k_alias"],
                ["top_values", "top_indices"],
                name="/bbox_head/TopK",
                axis=-1,
                largest=1,
                sorted=1,
            ),
        ],
        name="transhead_graph",
        inputs=[scores],
        outputs=[top_values, top_indices],
        initializer=[numpy_helper.from_array(np.array([4096], dtype=np.int64), name="dynamic_k")],
    )
    onnx_path = tmp_path / "transhead_identity.onnx"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), onnx_path
    )

    model = onnx.load(TransHeadTensorRTModifier(k=200).modify(onnx_path).as_posix())

    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == ["TopK"]
    assert list(model.graph.node[0].input) == ["scores", "/bbox_head/TopK_K"]
    assert _last_dim({o.name: o for o in model.graph.output}["top_values"]) == 200
