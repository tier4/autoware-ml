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
    CenterPointBoxEncodingModifier,
    OutputChannelPermuteModifier,
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


def _write_centerpoint_head_graph(path: Path) -> None:
    """Write a stand-in for the exported CenterPoint head outputs."""
    dim_features = helper.make_tensor_value_info("dim_features", TensorProto.FLOAT, [1, 3, 2, 2])
    rot_features = helper.make_tensor_value_info("rot_features", TensorProto.FLOAT, [1, 2, 2, 2])
    dim_output = helper.make_tensor_value_info("dim", TensorProto.FLOAT, [1, 3, 2, 2])
    rot_output = helper.make_tensor_value_info("rot", TensorProto.FLOAT, [1, 2, 2, 2])
    nodes = [
        helper.make_node(
            "Identity", inputs=["dim_features"], outputs=["dim"], name="/bbox_head/dim/Identity"
        ),
        helper.make_node(
            "Identity", inputs=["rot_features"], outputs=["rot"], name="/bbox_head/rot/Identity"
        ),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="centerpoint_head_graph",
        inputs=[dim_features, rot_features],
        outputs=[dim_output, rot_output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save_model(model, path.as_posix())


def _run_centerpoint_head_graph(
    path: Path, dim_features: np.ndarray, rot_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    onnxruntime = pytest.importorskip("onnxruntime")
    session = onnxruntime.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    dim, rot = session.run(
        ["dim", "rot"], {"dim_features": dim_features, "rot_features": rot_features}
    )
    return dim, rot


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


def test_centerpoint_box_encoding_modifier_swaps_dim_and_rot_channels(tmp_path: Path) -> None:
    onnx_path = tmp_path / "centerpoint_head.onnx"
    _write_centerpoint_head_graph(onnx_path)
    dim_features = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    rot_features = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)

    modified_path = CenterPointBoxEncodingModifier().modify(onnx_path)

    assert modified_path == onnx_path
    onnx.checker.check_model(onnx.load(modified_path.as_posix()))

    dim, rot = _run_centerpoint_head_graph(modified_path, dim_features, rot_features)
    # Length and width swap; height stays last. Sine and cosine swap.
    np.testing.assert_array_equal(dim, dim_features[:, [1, 0, 2]])
    np.testing.assert_array_equal(rot, rot_features[:, [1, 0]])


def test_centerpoint_box_encoding_modifier_preserves_output_interface(tmp_path: Path) -> None:
    onnx_path = tmp_path / "centerpoint_head.onnx"
    _write_centerpoint_head_graph(onnx_path)

    model = onnx.load(CenterPointBoxEncodingModifier().modify(onnx_path).as_posix())

    outputs = {output.name: output for output in model.graph.output}
    assert set(outputs) == {"dim", "rot"}
    assert [dim.dim_value for dim in outputs["dim"].type.tensor_type.shape.dim] == [1, 3, 2, 2]
    assert [dim.dim_value for dim in outputs["rot"].type.tensor_type.shape.dim] == [1, 2, 2, 2]

    nodes_by_name = {node.name: node for node in model.graph.node}
    assert nodes_by_name["/bbox_head/dim/Identity"].output[0] == "dim_PrePermute"
    dim_gather = nodes_by_name["dim_ChannelPermute"]
    assert dim_gather.op_type == "Gather"
    assert list(dim_gather.input) == ["dim_PrePermute", "dim_PermuteIndices"]

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    np.testing.assert_array_equal(
        numpy_helper.to_array(initializers["dim_PermuteIndices"]),
        np.array([1, 0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        numpy_helper.to_array(initializers["rot_PermuteIndices"]),
        np.array([1, 0], dtype=np.int64),
    )


def test_centerpoint_box_encoding_modifier_is_idempotent(tmp_path: Path) -> None:
    onnx_path = tmp_path / "centerpoint_head.onnx"
    _write_centerpoint_head_graph(onnx_path)
    dim_features = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    rot_features = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)

    CenterPointBoxEncodingModifier().modify(onnx_path)
    CenterPointBoxEncodingModifier().modify(onnx_path)

    dim, rot = _run_centerpoint_head_graph(onnx_path, dim_features, rot_features)
    np.testing.assert_array_equal(dim, dim_features[:, [1, 0, 2]])
    np.testing.assert_array_equal(rot, rot_features[:, [1, 0]])


def test_centerpoint_box_encoding_modifier_writes_to_output_path(tmp_path: Path) -> None:
    onnx_path = tmp_path / "centerpoint_head.onnx"
    destination_path = tmp_path / "centerpoint_head_swapped.onnx"
    _write_centerpoint_head_graph(onnx_path)
    dim_features = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    rot_features = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)

    modified_path = CenterPointBoxEncodingModifier(output_path=destination_path).modify(onnx_path)

    assert modified_path == destination_path
    dim, _rot = _run_centerpoint_head_graph(onnx_path, dim_features, rot_features)
    np.testing.assert_array_equal(dim, dim_features)


def test_output_channel_permute_modifier_rejects_mismatched_channel_count(tmp_path: Path) -> None:
    onnx_path = tmp_path / "centerpoint_head.onnx"
    _write_centerpoint_head_graph(onnx_path)

    with pytest.raises(ValueError, match="does not match"):
        OutputChannelPermuteModifier(output_name="rot", permutation=(1, 0, 2)).modify(onnx_path)


def test_output_channel_permute_modifier_rejects_invalid_permutation(tmp_path: Path) -> None:
    onnx_path = tmp_path / "centerpoint_head.onnx"
    _write_centerpoint_head_graph(onnx_path)

    with pytest.raises(ValueError, match="must be a permutation"):
        OutputChannelPermuteModifier(output_name="dim", permutation=(1, 1, 2)).modify(onnx_path)


def test_output_channel_permute_modifier_missing_output_is_optional(tmp_path: Path) -> None:
    onnx_path = tmp_path / "centerpoint_head.onnx"
    _write_centerpoint_head_graph(onnx_path)

    with pytest.raises(RuntimeError, match="Could not find graph output 'vel'"):
        OutputChannelPermuteModifier(output_name="vel", permutation=(1, 0)).modify(onnx_path)

    modified_path = OutputChannelPermuteModifier(
        output_name="vel", permutation=(1, 0), fail_if_missing=False
    ).modify(onnx_path)
    assert modified_path == onnx_path
