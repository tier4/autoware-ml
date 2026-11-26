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

"""Run a model's deployed function on TensorRT.

Each graph runs on its exported engine; the deployed function's own code, the PyTorch code
between the graphs, runs as it does in
:func:`~autoware_ml.deployment.graphs.run_deployment_in_torch`. A run starts from a
preprocessed batch and ends at the model outputs the function returns.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.deployment.backends.tensorrt_engine_runner import (
    TensorRTEngineRunner,
    pinned_cuda_device,
)
from autoware_ml.deployment.graphs import (
    DeploymentDeclaration,
    DeploymentGraph,
    GraphRunner,
    GraphTensor,
    run_deployment,
)
from autoware_ml.utils.deploy import engine_path


class TensorRTDeploymentRunner:
    """Run a model's deployed function on TensorRT, loading each graph's engine once.

    Only the graphs run on TensorRT: the deployed function's own code runs in PyTorch on
    ``device``. A runner serves one thread at a time.

    Args:
        deployment: The model's graphs and deployed function (``deployment_declaration()``).
        artifacts_dir: Directory of the ``<graph name>.engine`` artifacts.
        device: The CUDA device the engines and the deployed function run on; ``"cuda"`` is
            the current device at construction, and the runner stays on it.
        plugin_libraries: Custom plugin libraries the engines were built with.

    Raises:
        ValueError: If ``device`` is not a CUDA device.
        FileNotFoundError: If a graph's engine or a plugin library does not exist.
        RuntimeError: If an engine fails to load.
    """

    def __init__(
        self,
        deployment: DeploymentDeclaration,
        artifacts_dir: str | Path,
        device: str | torch.device,
        plugin_libraries: Sequence[str] = (),
    ) -> None:
        self.deployment = deployment
        self.device = pinned_cuda_device(device)
        self._engines = {
            graph.name: TensorRTEngineRunner(
                engine_path(artifacts_dir, graph.name), self.device, plugin_libraries
            )
            for graph in deployment.graphs
        }

    def _graph_runner(self, graph: DeploymentGraph) -> GraphRunner:
        """Give the runner of one graph: its engine, fed under the declared input names.

        Args:
            graph: The graph to run.

        Returns:
            The runner that names the graph's positional inputs and runs the engine on them.
        """
        engine = self._engines[graph.name]
        return lambda args: engine.run(dict(zip(graph.input_names, args, strict=True)))

    def run(self, batch: ModelBatchInputs) -> dict[str, GraphTensor]:
        """Run the deployed function once.

        Args:
            batch: The preprocessed batch the deployed function takes.

        Returns:
            The model outputs by name, owned by the caller.
        """
        return run_deployment(self.deployment, self._graph_runner, batch, self.device)
