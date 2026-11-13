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

"""The check every deployed model's tests run: the deployment reproduces ``forward()``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple
import unittest

from jaxtyping import Shaped
import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.deployment.graphs import DeploymentGraph, DeploymentRun, run_deployment_in_torch
from autoware_ml.models.module_base_model import ModuleBaseModel


class DeploymentForwardCheck(NamedTuple):
    """What :func:`assert_deployment_reproduces_forward` ran, for checks beyond the outputs."""

    graphs: tuple[DeploymentGraph, ...]
    forward_outputs: ModelOutputs
    run: DeploymentRun


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

    ``forward()`` and ``deployment_forward()`` compose the same modules twice, once for training
    and once for deployment, and nothing but this check keeps the two compositions equal.
    Both runs share the weights, so only kernel-selection noise remains; a wrong dataflow is
    off by O(1).

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
        The graphs, what ``forward()`` returned and the deployment run.

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
    run = run_deployment_in_torch(deployment, multi_task_batch_inputs, device)
    case.assertEqual(
        set(reference),
        set(run.outputs),
        "forward_tensors must name exactly the model outputs of the deployed function.",
    )
    for name, tensor in run.outputs.items():
        torch.testing.assert_close(
            tensor,
            reference[name],
            rtol=rtol,
            atol=atol,
            msg=lambda detail, name=name: f"Model output {name!r} differs from forward(): {detail}",
        )
    return DeploymentForwardCheck(deployment.graphs, forward_outputs, run)
