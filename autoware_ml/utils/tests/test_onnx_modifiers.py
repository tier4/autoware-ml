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
    TopKConstantKModifier,
    TopKPerRowModifier,
    TransHeadTensorRTModifier,
)

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

# Scores shape (batch, rows, columns) and K of the per-row TopK tests; K must not exceed columns.
PER_ROW_SHAPE = (2, 3, 64)
PER_ROW_K = 12


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


def _write_flattened_topk_graph(path: Path) -> None:
    """Write scores of shape ``PER_ROW_SHAPE`` flattened per sample into an unsorted TopK."""
    batch_size, rows, columns = PER_ROW_SHAPE
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, list(PER_ROW_SHAPE))
    top_values = helper.make_tensor_value_info(
        "top_values", TensorProto.FLOAT, [batch_size, PER_ROW_K]
    )
    top_indices = helper.make_tensor_value_info(
        "top_indices", TensorProto.INT64, [batch_size, PER_ROW_K]
    )
    nodes = [
        helper.make_node(
            "Reshape", ["scores", "flat_shape"], ["flat_scores"], name="/bbox_head/Reshape"
        ),
        helper.make_node(
            "TopK",
            ["flat_scores", "dynamic_k"],
            ["top_values", "top_indices"],
            name="/bbox_head/TopK",
            axis=-1,
            largest=1,
            sorted=0,
        ),
    ]
    initializers = [
        numpy_helper.from_array(
            np.array([batch_size, rows * columns], dtype=np.int64), "flat_shape"
        ),
        numpy_helper.from_array(np.array([rows * columns], dtype=np.int64), "dynamic_k"),
    ]
    graph = helper.make_graph(
        nodes, "flattened_topk", [scores], [top_values, top_indices], initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save_model(model, path.as_posix())


def _build_tie_heavy_scores() -> dict[str, np.ndarray]:
    """Return scores of shape ``PER_ROW_SHAPE`` with ties inside rows, across rows and at the cut."""
    generator = np.random.default_rng(0)
    straddling = np.full(PER_ROW_SHAPE, 0.25, dtype=np.float32)
    straddling[:, 2, :5] = 0.5
    one_row_equal = np.ones(PER_ROW_SHAPE, dtype=np.float32)
    one_row_equal[:, 0] = generator.random(PER_ROW_SHAPE[2])
    crowded = generator.random(PER_ROW_SHAPE).astype(np.float32)
    crowded[:, 1] += 1.0
    return {
        "random": generator.random(PER_ROW_SHAPE).astype(np.float32),
        "one_row_wins": crowded,
        "all_equal": np.zeros(PER_ROW_SHAPE, dtype=np.float32),
        "few_levels": (generator.integers(0, 4, PER_ROW_SHAPE) / 4).astype(np.float32),
        "ties_across_rows": straddling,
        "one_row_random": one_row_equal,
    }


def test_topk_per_row_modifier_equals_one_sorted_topk(tmp_path: Path) -> None:
    """
    Input: scores flattened over rows into an unsorted TopK with K fixed by
        TopKConstantKModifier, rewritten by TopKPerRowModifier, on random and tie-heavy scores.
    Expected: two TopK nodes and no Reshape of the scores left; the values and flat indices
        of one sorted TopK, equal scores ranked by ascending flat index as ONNX does.
    Check: the TopK and Reshape counts, and the onnxruntime outputs against a stable sort.
    """
    onnx_path = tmp_path / "flattened_topk.onnx"
    _write_flattened_topk_graph(onnx_path)
    TopKConstantKModifier(k=PER_ROW_K, node_name_substring="/bbox_head/TopK").modify(onnx_path)

    TopKPerRowModifier(node_name_substring="/bbox_head/TopK").modify(onnx_path)

    model = onnx.load(onnx_path.as_posix())
    op_types = [node.op_type for node in model.graph.node]
    assert op_types.count("TopK") == 2
    assert "/bbox_head/Reshape" not in [node.name for node in model.graph.node]
    session = onnxruntime.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    for case, scores in _build_tie_heavy_scores().items():
        flat_scores = scores.reshape(PER_ROW_SHAPE[0], -1)
        expected_indices = np.argsort(-flat_scores, axis=-1, kind="stable")[:, :PER_ROW_K]
        values, indices = session.run(None, {"scores": scores})
        np.testing.assert_array_equal(indices, expected_indices, err_msg=case)
        np.testing.assert_array_equal(
            values, np.take_along_axis(flat_scores, expected_indices, -1), err_msg=case
        )


def test_topk_per_row_modifier_refuses_a_dynamic_k(tmp_path: Path) -> None:
    """
    Input: the flattened TopK graph with its K fed as a graph input instead of a constant.
    Expected: a RuntimeError asking for TopKConstantKModifier first.
    Check: the error message.
    """
    onnx_path = tmp_path / "flattened_topk.onnx"
    _write_flattened_topk_graph(onnx_path)
    model = onnx.load(onnx_path.as_posix())
    model.graph.input.append(helper.make_tensor_value_info("dynamic_k", TensorProto.INT64, [1]))
    model.graph.initializer.pop()
    onnx.save_model(model, onnx_path.as_posix())

    with pytest.raises(RuntimeError, match="no constant K; run TopKConstantKModifier first"):
        TopKPerRowModifier(node_name_substring="/bbox_head/TopK").modify(onnx_path)


def test_transhead_modifier_splits_the_topk_only_when_asked(tmp_path: Path) -> None:
    """
    Input: the flattened TopK graph through TransHeadTensorRTModifier with and without
        ``topk_per_class``.
    Expected: two TopK nodes with the flag and one without.
    Check: the TopK count of each result.
    """
    for topk_per_class, topk_count in ((True, 2), (False, 1)):
        onnx_path = tmp_path / f"transhead_{topk_per_class}.onnx"
        _write_flattened_topk_graph(onnx_path)

        TransHeadTensorRTModifier(k=PER_ROW_K, topk_per_class=topk_per_class).modify(onnx_path)

        op_types = [node.op_type for node in onnx.load(onnx_path.as_posix()).graph.node]
        assert op_types.count("TopK") == topk_count
