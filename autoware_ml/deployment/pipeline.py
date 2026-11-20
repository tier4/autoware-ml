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

"""Run a model's deployed function on one backend.

Each graph runs its module (pytorch) or its exported engine (tensorrt); the deployed function's
own code, the PyTorch code between the graphs, runs the same on every backend. A run starts from
a preprocessed batch and ends at the model outputs the function returns.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Final

import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.deployment.graphs import (
    DeploymentDeclaration,
    DeploymentGraph,
    GraphRunner,
    GraphTensor,
    run_deployment,
    run_graph_module,
)
from autoware_ml.types.backend import Backend

#: File suffix of the artifact a backend runs; pytorch runs the modules themselves.
_ARTIFACT_SUFFIX: Final = {Backend.TENSORRT: ".engine"}


def artifact_path(output_dir: str | Path, graph_name: str, backend: Backend) -> Path:
    """Path of a graph's exported artifact: ``<output_dir>/<graph_name><suffix>``.

    Args:
        output_dir: Directory the export wrote to.
        graph_name: The graph's name, the artifact stem.
        backend: The backend whose artifact is meant.

    Returns:
        The artifact path.

    Raises:
        ValueError: For the pytorch backend, which runs the modules and has no artifact.
    """
    suffix = _ARTIFACT_SUFFIX.get(backend)
    if suffix is None:
        raise ValueError(f"The {backend} backend has no exported artifact.")
    return Path(output_dir) / f"{graph_name}{suffix}"


class DeployedPipeline:
    """Run a model's deployed function on one backend, loading each graph's artifact once.

    The graph modules run as they are, on their own device and in their current mode: the
    model is expected in eval mode on ``device``. A pipeline serves one thread at a time.

    Args:
        deployment: The model's graphs and deployed function (``deployment_declaration()``).
        backend: Backend that runs the graphs.
        device: Device the graphs run on.
        artifacts_dir: Directory of the ``<graph>.engine`` artifacts, required by the
            tensorrt backend.
        plugin_libraries: TensorRT plugin libraries the engines were built with.

    Raises:
        ValueError: When the tensorrt backend is asked for without ``artifacts_dir``.
    """

    def __init__(
        self,
        deployment: DeploymentDeclaration,
        backend: Backend,
        device: str | torch.device,
        artifacts_dir: str | Path | None = None,
        plugin_libraries: Sequence[str] = (),
    ) -> None:
        self.deployment = deployment
        self.backend = backend
        self.device = torch.device(device)
        if backend is Backend.TENSORRT and artifacts_dir is None:
            raise ValueError("artifacts_dir is required for the tensorrt backend.")
        self._runners: dict[str, GraphRunner] = {
            graph.name: self._graph_runner(graph, artifacts_dir, plugin_libraries)
            for graph in deployment.graphs
        }

    def _graph_runner(
        self,
        graph: DeploymentGraph,
        artifacts_dir: str | Path | None,
        plugin_libraries: Sequence[str],
    ) -> GraphRunner:
        """Build the runner of one graph on this backend.

        Args:
            graph: The graph to run.
            artifacts_dir: Directory of the artifacts; ``None`` only for the pytorch backend.
            plugin_libraries: TensorRT plugin libraries the engine was built with.

        Returns:
            The graph's module for pytorch, or its TensorRT engine fed under the declared
            input names.
        """
        if self.backend is Backend.PYTORCH:
            return partial(run_graph_module, graph)
        # Imported here: a pytorch pipeline needs no tensorrt.
        from autoware_ml.deployment.backends.tensorrt_runner import (  # noqa: PLC0415
            TensorRTModuleRunner,
        )

        engine = TensorRTModuleRunner(
            artifact_path(artifacts_dir, graph.name, self.backend), self.device, plugin_libraries
        )
        return lambda args: engine.run(dict(zip(graph.input_names, args, strict=True)))

    def run(self, batch: ModelBatchInputs) -> dict[str, GraphTensor]:
        """Run the deployed function once.

        Args:
            batch: The preprocessed batch the deployed function takes.

        Returns:
            The model outputs by name, owned by the caller.
        """
        return run_deployment(
            self.deployment, lambda graph: self._runners[graph.name], batch, self.device
        )
