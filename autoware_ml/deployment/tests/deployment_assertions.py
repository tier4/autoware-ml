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

"""Assertions the deployed models' tests share.

The deployment reproduces ``forward()``, and every spec derived from the graphs exports an ONNX
that carries its graph's names and dynamic axes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import tempfile
from typing import NamedTuple
import unittest

from jaxtyping import Shaped
from omegaconf import OmegaConf
import onnx
import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.deployment.graphs import GraphTensor, run_deployment_in_torch
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.utils.deploy import export_to_onnx


class DeploymentForwardCheck(NamedTuple):
    """What :func:`assert_deployment_reproduces_forward` ran, for checks beyond the outputs.

    Attributes:
        forward_outputs: What ``forward()`` returned on the batch.
        deployment_outputs: The model outputs of the deployment run on the same batch, in PyTorch.
    """

    forward_outputs: ModelOutputs
    deployment_outputs: dict[str, GraphTensor]


def assert_deployment_reproduces_forward(
    model: ModuleBaseModel,
    multi_task_batch_inputs: ModelBatchInputs,
    device: torch.device,
    forward_tensors: Callable[[ModelOutputs], Mapping[str, Shaped[torch.Tensor, "*shape"]]],
    *,
    rtol: float,
    atol: float,
) -> DeploymentForwardCheck:
    """Run ``forward()`` and the deployed function on one batch and compare every model output.

    ``forward()`` and ``deployment_forward()`` compose the same modules twice; this check is
    what keeps the two compositions equal. The runs share the weights, so the tolerances only
    have to cover kernel-selection noise.

    Args:
        model: Model in eval mode declaring ``deployment_graphs()`` and ``deployment_forward()``;
            the deployment is what its ``deployment_declaration()`` returns.
        multi_task_batch_inputs: The preprocessed batch both runs start from.
        device: Device the graphs run on.
        forward_tensors: Maps the outputs ``forward()`` returns to the model outputs the
            deployed function returns, by name.
        rtol: Relative tolerance of :func:`torch.testing.assert_close`.
        atol: Absolute tolerance of :func:`torch.testing.assert_close`.

    Returns:
        What ``forward()`` returned and what the deployment returned.

    Raises:
        AssertionError: When the model declares no graphs, ``forward_tensors`` does not name
            exactly the model outputs, or a model output differs from ``forward()``'s.
    """
    case = unittest.TestCase()
    deployment = model.deployment_declaration()
    case.assertIsNotNone(deployment, f"{type(model).__name__} declares no deployment graphs.")
    with torch.no_grad():
        forward_outputs = model(multi_task_batch_inputs)
    reference = forward_tensors(forward_outputs)
    outputs = run_deployment_in_torch(deployment, multi_task_batch_inputs, device)
    case.assertEqual(
        sorted(reference),
        sorted(outputs),
        "forward_tensors must name exactly the model outputs of the deployed function: "
        f"reference {sorted(reference)}, deployed {sorted(outputs)}.",
    )
    for name, tensor in outputs.items():
        torch.testing.assert_close(
            tensor,
            reference[name],
            rtol=rtol,
            atol=atol,
            msg=lambda detail, name=name: f"Model output {name!r} differs from forward(): {detail}",
        )
    return DeploymentForwardCheck(forward_outputs, outputs)


def assert_specs_export_onnx_with_the_declared_io(
    model: ModuleBaseModel, multi_task_batch_inputs: ModelBatchInputs
) -> None:
    """Export every spec derived from a model's graphs to ONNX and check it against its graph.

    The export runs with the base configs' ONNX settings: the TorchScript exporter, opset 17,
    no constant folding.

    Args:
        model: Model in eval mode declaring its deployment graphs.
        multi_task_batch_inputs: One batch, on the model's device, to trace on.

    Raises:
        AssertionError: When an ONNX's input or output names are not its graph's, or a declared
            dynamic axis is not the ``dim_param`` of the ONNX tensor.
    """
    case = unittest.TestCase()
    graphs = {graph.name: graph for graph in model.deployment_graphs()}
    onnx_cfg = OmegaConf.create(
        {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
    )
    with tempfile.TemporaryDirectory() as directory:
        for name, spec in model.build_export_specs(multi_task_batch_inputs).items():
            path = Path(directory) / f"{name}.onnx"
            export_to_onnx(spec, onnx_cfg, path)
            onnx_graph = onnx.load(path).graph
            declared = graphs[name]
            case.assertEqual(
                [tensor.name for tensor in onnx_graph.input], list(declared.input_names), name
            )
            case.assertEqual(
                [tensor.name for tensor in onnx_graph.output], list(declared.output_names), name
            )
            for tensor in [*onnx_graph.input, *onnx_graph.output]:
                axes = declared.onnx_dynamic_axes.get(tensor.name, {})
                dims = tensor.type.tensor_type.shape.dim
                case.assertEqual({i: dims[i].dim_param for i in axes}, dict(axes), tensor.name)
