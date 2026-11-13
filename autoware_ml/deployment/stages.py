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

"""Deployment declared as graphs plus one function.

A model declares the graphs it exports (:class:`DeployGraph`) and writes its deployed inference
as one function, ``deployed_forward(multi_task_batch_inputs, graphs)``: the glue is plain code
in that function, and each graph is called by name. ``multi_task_batch_inputs`` is the
preprocessed batch ``forward()`` takes. The framework hands the function its graphs as callables
(:class:`GraphCall`), so one function serves export and every later run, and the model outputs
are what the function returns. :func:`build_export_specs_from_deployment` derives the export
specs from one PyTorch run of that function.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple

from jaxtyping import Shaped
import torch
from torch import nn

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.utils.deploy import ExportSpec

#: A graph input or output: any dtype and shape, fixed by the graph's module.
GraphTensor = Shaped[torch.Tensor, "*shape"]


@dataclass(frozen=True)
class DeployGraph:
    """One exported ONNX / TensorRT graph.

    Attributes:
        name: Unique graph name, also the ``deploy.onnx.modules`` key and the artifact stem.
        module: The traced module, export-ready as declared (the live submodule or an
            export copy); its positional arguments are the ``inputs`` tensors. The framework
            runs and traces it and never alters it.
        inputs: The names the deployed function passes the tensors under, in the module's
            argument order; they are the ONNX input names.
        outputs: ONNX output names, in the module's return order; a call of the graph returns
            its tensors under these names.
        onnx_dynamic_axes: Axes dynamic by construction, ``{tensor: {dim: name}}``. They
            become the spec's ``dynamic_axes``, the one declaration both ONNX exporters
            derive from.
    """

    name: str
    module: nn.Module
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    onnx_dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The exporter renames a repeated name (``x`` -> ``x.1``), so the artifact would not
        # carry the declared names.
        for kind, names in (("inputs", self.inputs), ("outputs", self.outputs)):
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"DeployGraph {self.name!r} lists {repeated} more than once in its {kind}; "
                    "each is one ONNX tensor name."
                )


def validate_graphs(graphs: Sequence[DeployGraph]) -> tuple[DeployGraph, ...]:
    """Check a model's graph declaration and return it as a tuple.

    Whether the deployed function calls each graph with its declared inputs is checked when
    it runs.

    Raises:
        ValueError: On no graph, or on two graphs with one name (the artifact's file name).
    """
    graphs = tuple(graphs)
    if not graphs:
        raise ValueError("A deployment needs at least one exportable DeployGraph.")
    names = [graph.name for graph in graphs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate graph names: {duplicates}.")
    return graphs


def run_graph_stage(stage: DeployGraph, args: Sequence[GraphTensor]) -> dict[str, GraphTensor]:
    """Call a graph's module on its positional inputs and name its outputs.

    Raises:
        ValueError: When the module returns another number of tensors than ``outputs``.
    """
    raw = stage.module(*args)
    if isinstance(raw, torch.Tensor):
        raw = (raw,)
    return dict(zip(stage.outputs, raw, strict=True))


#: Runs one graph on its positional inputs and returns its outputs by name.
GraphRunner = Callable[[tuple[GraphTensor, ...]], Mapping[str, GraphTensor]]


class GraphCall:
    """One declared graph as the deployed function calls it: tensors in and out by name.

    Attributes:
        stage: The graph's declaration.
        inputs: The graph's input names, in declaration order.
        outputs: The graph's output names, in declaration order.
        args: The positional inputs of the call, in declaration order, on the run device;
            ``None`` until the graph is called. Export traces the graph on them.
    """

    def __init__(self, stage: DeployGraph, run: GraphRunner, device: torch.device) -> None:
        self.stage = stage
        self.inputs = stage.inputs
        self.outputs = stage.outputs
        self.args: tuple[GraphTensor, ...] | None = None
        self._run = run
        self._device = device
        self._versions: tuple[int, ...] = ()

    def __call__(self, **tensors: GraphTensor) -> dict[str, GraphTensor]:
        """Run the graph on exactly its declared inputs and return its declared outputs.

        Raises:
            RuntimeError: On a second call in one run; a graph is one artifact, run once.
            TypeError: When the keyword names are not the declared inputs, or a value is not
                a tensor.
        """
        name = self.stage.name
        if self.args is not None:
            raise RuntimeError(f"Graph {name!r} was called twice in one run.")
        if set(tensors) != set(self.inputs):
            raise TypeError(
                f"Graph {name!r} takes {list(self.inputs)}, got {sorted(tensors)}: missing "
                f"{sorted(set(self.inputs) - set(tensors))}, unknown "
                f"{sorted(set(tensors) - set(self.inputs))}."
            )
        others = {
            key: type(value).__name__
            for key, value in tensors.items()
            if not torch.is_tensor(value)
        }
        if others:
            raise TypeError(
                f"Graph {name!r} binds tensors only; got {others}. The deployed function has "
                "to pass each input as a tensor."
            )
        self.args = tuple(tensors[key].to(self._device) for key in self.inputs)
        self._versions = tuple(arg._version for arg in self.args)
        produced = self._run(self.args)
        return {key: produced[key] for key in self.outputs}

    def modified_inputs(self) -> list[str]:
        """Names of the inputs modified in place since the call; export traces on them."""
        if self.args is None:
            return []
        return [
            key
            for key, arg, version in zip(self.inputs, self.args, self._versions)
            if arg._version != version
        ]


class Graphs(Mapping[str, GraphCall]):
    """The graphs of one run, by name, as the deployed function receives them.

    Attributes:
        device: Device the graphs run on; the function moves the tensors it makes there.
    """

    def __init__(self, calls: Mapping[str, GraphCall], device: torch.device) -> None:
        self._calls = dict(calls)
        self.device = device

    def __getitem__(self, name: str) -> GraphCall:
        try:
            return self._calls[name]
        except KeyError:
            raise KeyError(
                f"No graph {name!r}; the model declares {sorted(self._calls)}."
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._calls)

    def __len__(self) -> int:
        return len(self._calls)


#: ``deployed_forward(multi_task_batch_inputs, graphs) -> {output name: tensor}``.
DeployedForward = Callable[[ModelBatchInputs, Graphs], Mapping[str, GraphTensor]]


@dataclass(frozen=True)
class DeploymentDeclaration:
    """A model's deployment: the graphs it exports and the function that runs them.

    Attributes:
        graphs: The exported graphs, checked by :func:`validate_graphs`.
        forward: The deployed inference,
            ``forward(multi_task_batch_inputs, graphs) -> model outputs``.
    """

    graphs: tuple[DeployGraph, ...]
    forward: DeployedForward

    def __post_init__(self) -> None:
        object.__setattr__(self, "graphs", validate_graphs(self.graphs))


def call_deployed_forward(
    deployment: DeploymentDeclaration,
    runner_for: Callable[[DeployGraph], GraphRunner],
    multi_task_batch_inputs: ModelBatchInputs,
    device: torch.device,
) -> tuple[dict[str, GraphTensor], dict[str, GraphCall]]:
    """Run the deployed function once, each graph on ``runner_for(graph)``; check what it did.

    Args:
        deployment: The model's graphs and deployed function.
        runner_for: Gives the callable that runs a graph on its positional inputs.
        multi_task_batch_inputs: The preprocessed batch, as ``forward()`` takes it.
        device: Device the graphs run on.

    Returns:
        The model outputs, and the call of each graph by name.

    Raises:
        TypeError: When the function returns something other than a non-empty mapping of
            tensors.
        RuntimeError: When the function did not call a declared graph.
    """
    calls = {stage.name: GraphCall(stage, runner_for(stage), device) for stage in deployment.graphs}
    with torch.no_grad():
        outputs = deployment.forward(multi_task_batch_inputs, Graphs(calls, device))
    if not isinstance(outputs, Mapping) or not outputs:
        raise TypeError(
            "deployed_forward returns the model outputs as a non-empty mapping of name -> "
            f"tensor, got {type(outputs).__name__}."
        )
    others = sorted(name for name, value in outputs.items() if not torch.is_tensor(value))
    if others:
        raise TypeError(f"deployed_forward returned model outputs that are not tensors: {others}.")
    uncalled = [name for name, call in calls.items() if call.args is None]
    if uncalled:
        raise RuntimeError(
            f"deployed_forward did not call the declared graphs {uncalled}; every declared "
            "graph is exported, so each has to run."
        )
    return dict(outputs), calls


class DeploymentRun(NamedTuple):
    """One PyTorch run of a deployment.

    Attributes:
        outputs: The model outputs the deployed function returned, by name.
        graph_inputs: The positional inputs each graph ran on, by graph name: the tensors the
            function passed, which export traces each graph on.
    """

    outputs: dict[str, GraphTensor]
    graph_inputs: dict[str, tuple[GraphTensor, ...]]


def run_deployment_in_torch(
    deployment: DeploymentDeclaration,
    multi_task_batch_inputs: ModelBatchInputs,
    device: torch.device,
) -> DeploymentRun:
    """Run the deployed function once in PyTorch, every graph on its own module.

    Export takes its trace inputs from ``graph_inputs``, so a graph input the function modifies
    in place after the graph ran is rejected. Modules run in their current mode; export
    switches them to eval first.

    Raises:
        TypeError: When a graph gets a non-tensor input, or the function returns something
            other than a mapping of tensors.
        ValueError: When a graph returns another number of tensors than it declares, or the
            function modifies a graph input in place.
        RuntimeError: When the function skips a declared graph or calls one twice.
    """
    outputs, calls = call_deployed_forward(
        deployment,
        lambda stage: partial(run_graph_stage, stage),
        multi_task_batch_inputs,
        device,
    )
    modified = {name: call.modified_inputs() for name, call in calls.items()}
    modified = {name: names for name, names in modified.items() if names}
    if modified:
        raise ValueError(
            f"deployed_forward modified graph inputs in place after the graphs ran: {modified}; "
            "export traces each graph on the tensors it ran on."
        )
    return DeploymentRun(outputs, {name: call.args for name, call in calls.items()})


def export_spec_for_graph(stage: DeployGraph, args: tuple[GraphTensor, ...]) -> ExportSpec:
    """Build one graph's spec from the tensors it ran on.

    The trace inputs are ``args``, the tensors the deployed function passed the graph. The spec
    takes the graph's declared input and output names. Its ``dynamic_axes`` are the graph's
    ``onnx_dynamic_axes``, or ``None`` when the graph declares none.
    """
    dynamic_axes = (
        {name: dict(axes) for name, axes in stage.onnx_dynamic_axes.items()}
        if stage.onnx_dynamic_axes
        else None
    )
    return ExportSpec(
        module=stage.module,
        args=args,
        input_names=list(stage.inputs),
        output_names=list(stage.outputs),
        dynamic_axes=dynamic_axes,
    )


def build_export_specs_from_deployment(
    deployment: DeploymentDeclaration,
    multi_task_batch_inputs: ModelBatchInputs,
    device: torch.device,
) -> dict[str, ExportSpec]:
    """Run the deployed function once and return ``{graph name: ExportSpec}`` per graph.

    The graph modules are switched to eval first. An export wrapper starts in training
    mode, and ``torch.onnx.export`` restores the module's mode recursively when it is done:
    a wrapper left in training mode would switch the submodules it shares with the model,
    BatchNorm included, to training mode.

    Args:
        deployment: The model's graphs and deployed function.
        multi_task_batch_inputs: One preprocessed example batch.
        device: Device of the tracing run.
    """
    for stage in deployment.graphs:
        stage.module.eval()
    run = run_deployment_in_torch(deployment, multi_task_batch_inputs, device)
    return {
        stage.name: export_spec_for_graph(stage, run.graph_inputs[stage.name])
        for stage in deployment.graphs
    }
