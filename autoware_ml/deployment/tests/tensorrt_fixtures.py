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

"""Toy ONNX graphs and the engine build the TensorRT tests share."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnx
from omegaconf import OmegaConf
from onnx import TensorProto, helper

from autoware_ml.utils.deploy import build_tensorrt_engine, engine_path

#: One dynamic input's profile, ``{"min_shape": ..., "opt_shape": ..., "max_shape": ...}``.
Profile = Mapping[str, Sequence[int]]

#: The ``[n, 2]`` profile of the add graph: n from 1 to 8.
N_PROFILE: Profile = {"min_shape": [1, 2], "opt_shape": [4, 2], "max_shape": [8, 2]}


def _save_graph(
    path: Path,
    nodes: Sequence[onnx.NodeProto],
    inputs: Sequence[onnx.ValueInfoProto],
    outputs: Sequence[onnx.ValueInfoProto],
    initializers: Sequence[onnx.TensorProto] = (),
) -> Path:
    """Save a one-graph ONNX model at opset 17.

    Args:
        path: Where the model goes.
        nodes: The graph's nodes.
        inputs: The graph's inputs, with their dtype and shape.
        outputs: The graph's outputs, with their dtype and shape.
        initializers: The graph's constants.

    Returns:
        ``path``.
    """
    graph = helper.make_graph(list(nodes), "g", list(inputs), list(outputs), list(initializers))
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    return path


def doubling_gemm_graph(directory: Path) -> Path:
    """Save ``y = x @ 2I``, twice its static ``[2, 4]`` input.

    Args:
        directory: Where ``gemm.onnx`` goes.

    Returns:
        The model's path.
    """
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4], 2 * np.eye(4, dtype=np.float32).flatten()
    )
    return _save_graph(
        directory / "gemm.onnx",
        [helper.make_node("Gemm", ["x", "w"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        [weight],
    )


def add_graph(directory: Path) -> Path:
    """Save ``y = a + b`` over two ``[n, 2]`` inputs sharing the dynamic axis ``n``.

    Args:
        directory: Where ``add.onnx`` goes.

    Returns:
        The model's path.
    """
    return _save_graph(
        directory / "add.onnx",
        [helper.make_node("Add", ["a", "b"], ["y"])],
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, ["n", 2]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, ["n", 2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 2])],
    )


def nonzero_graph(directory: Path) -> Path:
    """Save ``y = NonZero(x)``, whose output shape depends on the data.

    Args:
        directory: Where ``nonzero.onnx`` goes.

    Returns:
        The model's path.
    """
    return _save_graph(
        directory / "nonzero.onnx",
        [helper.make_node("NonZero", ["x"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n"])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [1, "k"])],
    )


def build_engine(onnx_path: Path, input_shapes: Mapping[str, Profile]) -> Path:
    """Build the engine of an ONNX model next to it, with the export's builder.

    Args:
        onnx_path: The ONNX model; the engine takes its stem.
        input_shapes: The profile of each dynamic input; empty for a static model.

    Returns:
        The engine's path.
    """
    path = engine_path(onnx_path.parent, onnx_path.stem)
    deploy_cfg = OmegaConf.create(
        {
            "tensorrt": {
                "workspace_size": 1 << 28,
                "modules": {onnx_path.stem: {"input_shapes": dict(input_shapes)}},
            }
        }
    )
    build_tensorrt_engine(onnx_path, deploy_cfg, path, onnx_path.stem)
    return path
