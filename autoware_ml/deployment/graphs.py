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

"""A model's deployment: the computation graphs it exports and the function that runs them.

A model declares the computation graphs it exports, one :class:`DeploymentGraph` per ONNX file
and TensorRT engine. It also writes ``deployment_forward(multi_task_batch_inputs, executors,
device)``, the PyTorch code between those graphs: it takes the preprocessed batch ``forward()``
takes, runs each graph as ``executors[name].run(**inputs)`` and returns the model outputs.
The same function serves export, where :func:`build_export_specs_from_deployment` runs it once
in PyTorch and traces each graph on the tensors it was passed, and every later run on a backend.

:class:`DeploymentGraph` and :class:`DeploymentDeclaration` are what the model declares. A
:class:`GraphExecutor` belongs to one run: it is what the function calls, by name and checked.
A :data:`GraphRunner` is what a backend provides to it: positional tensors in, named tensors
out; :func:`run_graph_module` is the PyTorch one. :class:`~autoware_ml.utils.deploy.ExportSpec`
is derived from a graph and the tensors it ran on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial

from jaxtyping import Shaped
import torch
from torch import nn

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.utils.deploy import ExportSpec

GraphTensor = Shaped[torch.Tensor, "*shape"]


@dataclass(frozen=True)
class DeploymentGraph:
    """One exported computation graph: one ONNX file and one TensorRT engine.

    Attributes:
        name: Unique graph name, also the ``deploy.onnx.modules`` key and the artifact stem.
        module: The traced module, export-ready as declared (the live submodule or an
            export copy); its positional arguments are the ``input_names`` tensors. The
            framework runs and traces it and never alters it.
        input_names: The names the deployed function passes the tensors under, in the
            module's argument order; they are the ONNX input names.
        output_names: ONNX output names, in the module's return order; a run of the graph
            returns its tensors under these names.
        onnx_dynamic_axes: Axes dynamic by construction, ``{tensor: {dim: name}}``. They
            become the spec's ``dynamic_axes``, which the TorchScript and the ``torch.export``
            ONNX exporter both take.
    """

    name: str
    module: nn.Module
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    onnx_dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for kind, names in (("input_names", self.input_names), ("output_names", self.output_names)):
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"DeploymentGraph {self.name!r} lists {repeated} more than once in its "
                    f"{kind}; each is one ONNX tensor name."
                )


def run_graph_module(graph: DeploymentGraph, args: Sequence[GraphTensor]) -> dict[str, GraphTensor]:
    """Run a graph's PyTorch module on its positional inputs and name its outputs.

    This is the PyTorch :data:`GraphRunner`; a backend runner runs the graph's artifact instead.
    The module returns its tensors as one tensor, a tuple in ``output_names`` order, or a
    mapping whose keys are ``output_names`` in that order: ONNX export names the outputs by
    position, so a mapping in another order would name the exported tensors wrongly.

    Raises:
        ValueError: When the module returns another number of tensors than ``output_names``,
            or a mapping whose keys are not ``output_names`` in the declared order.
    """
    raw = graph.module(*args)
    if isinstance(raw, Mapping):
        if list(raw) != list(graph.output_names):
            raise ValueError(
                f"Graph {graph.name!r} returned the outputs {list(raw)}; it declares "
                f"{list(graph.output_names)}, and ONNX export names outputs by position, so "
                "a mapping has to list them in the declared order."
            )
        return dict(raw)
    if isinstance(raw, torch.Tensor):
        raw = (raw,)
    raw = tuple(raw)
    if len(raw) != len(graph.output_names):
        raise ValueError(
            f"Graph {graph.name!r} returned {len(raw)} tensors; it declares "
            f"{len(graph.output_names)} outputs {list(graph.output_names)}."
        )
    return dict(zip(graph.output_names, raw, strict=True))


#: What a backend provides to run one graph: its positional inputs in, its outputs by name.
GraphRunner = Callable[[tuple[GraphTensor, ...]], Mapping[str, GraphTensor]]


class GraphExecutor:
    """Runs one declared graph on this run's backend, with the tensors in and out by name.

    ``run(**inputs)`` takes exactly ``input_names`` as tensors, moves them to the run device,
    runs the graph and returns its ``output_names`` tensors by name. A graph runs once per
    run.

    Attributes:
        name: The graph's name, the artifact stem.
        input_names: The graph's input names, in declaration order.
        output_names: The graph's output names, in declaration order.
    """

    def __init__(self, graph: DeploymentGraph, runner: GraphRunner, device: torch.device) -> None:
        self.name = graph.name
        self.input_names = graph.input_names
        self.output_names = graph.output_names
        self._runner = runner
        self._device = device
        self._output_name_set = frozenset(graph.output_names)
        self._ran = False

    @property
    def ran(self) -> bool:
        """Whether the graph ran in this run."""
        return self._ran

    def run(self, **inputs: GraphTensor) -> dict[str, GraphTensor]:
        """Run the graph on exactly its declared inputs and return its declared outputs.

        Raises:
            RuntimeError: On a second run in one deployment run; a graph is one artifact,
                run once.
            TypeError: When the keyword names are not the declared ``input_names``, or a
                value is not a tensor; when the runner returns something other than a
                mapping of tensors.
            ValueError: When the runner's output names are not exactly the declared
                ``output_names``: the artifact does not match the declaration.
        """
        name = self.name
        if self._ran:
            raise RuntimeError(f"Graph {name!r} was run twice in one run.")
        if set(inputs) != set(self.input_names):
            raise TypeError(
                f"Graph {name!r} takes {list(self.input_names)}, got {sorted(inputs)}: missing "
                f"{sorted(set(self.input_names) - set(inputs))}, unknown "
                f"{sorted(set(inputs) - set(self.input_names))}."
            )
        others = {
            key: type(value).__name__ for key, value in inputs.items() if not torch.is_tensor(value)
        }
        if others:
            raise TypeError(
                f"Graph {name!r} binds tensors only; got {others}. The deployed function has "
                "to pass each input as a tensor."
            )
        args = tuple(inputs[key].to(self._device) for key in self.input_names)
        self._ran = True
        return self._check_outputs(self._runner(args))

    def _check_outputs(self, produced: Mapping[str, GraphTensor]) -> dict[str, GraphTensor]:
        """Return the runner's outputs in ``output_names`` order; any other set of names, or a
        non-tensor, means the artifact does not match the declaration and is refused by name.
        """
        name = self.name
        if not isinstance(produced, Mapping):
            raise TypeError(
                f"Graph {name!r}: the runner returned {type(produced).__name__}, not a mapping "
                "of output name -> tensor."
            )
        if set(produced) != self._output_name_set:
            raise ValueError(
                f"Graph {name!r} declares the outputs {list(self.output_names)}, the runner "
                f"returned {sorted(produced)}: missing "
                f"{sorted(self._output_name_set - set(produced))}, unknown "
                f"{sorted(set(produced) - self._output_name_set)}."
            )
        others = {
            key: type(value).__name__
            for key, value in produced.items()
            if not torch.is_tensor(value)
        }
        if others:
            raise TypeError(f"Graph {name!r} returned outputs that are not tensors: {others}.")
        return {key: produced[key] for key in self.output_names}


#: The deployed inference: ``forward(multi_task_batch_inputs, executors, device) -> model outputs``.
DeploymentForward = Callable[
    [ModelBatchInputs, Mapping[str, GraphExecutor], torch.device], Mapping[str, GraphTensor]
]


@dataclass(frozen=True)
class DeploymentDeclaration:
    """A model's deployment: the graphs it exports and the function that runs them.

    Attributes:
        graphs: The exported graphs, at least one, with unique names (each name is an
            artifact's file name).
        forward: The deployed inference,
            ``forward(multi_task_batch_inputs, executors, device) -> model outputs``.
    """

    graphs: tuple[DeploymentGraph, ...]
    forward: DeploymentForward

    def __post_init__(self) -> None:
        """Keep the graphs as a tuple and refuse an empty declaration or a repeated name.

        Raises:
            ValueError: On no graph, or on two graphs with one name.
        """
        graphs = tuple(self.graphs)
        if not graphs:
            raise ValueError("A deployment needs at least one exportable DeploymentGraph.")
        names = [graph.name for graph in graphs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate graph names: {duplicates}.")
        object.__setattr__(self, "graphs", graphs)


def run_deployment(
    deployment: DeploymentDeclaration,
    runner_for: Callable[[DeploymentGraph], GraphRunner],
    multi_task_batch_inputs: ModelBatchInputs,
    device: torch.device,
) -> dict[str, GraphTensor]:
    """Run the deployed function once, each graph on ``runner_for(graph)``; check what it did.

    ``runner_for`` is the backend seam: :func:`run_deployment_in_torch` runs each graph's
    module, and a backend runner hands the graph's positional inputs to its ONNX Runtime
    session or TensorRT engine instead. The function does not know which.

    Args:
        deployment: The model's graphs and deployed function.
        runner_for: Gives the runner of a graph on this backend.
        multi_task_batch_inputs: The preprocessed batch, as ``forward()`` takes it.
        device: Device the graphs run on.

    Returns:
        The model outputs the deployed function returned, by name.

    Raises:
        TypeError: When a graph gets a non-tensor input, or the function returns something
            other than a non-empty mapping of tensors.
        RuntimeError: When the function skips a declared graph or runs one twice.
    """
    executors = {
        graph.name: GraphExecutor(graph, runner_for(graph), device) for graph in deployment.graphs
    }
    with torch.no_grad():
        outputs = deployment.forward(multi_task_batch_inputs, executors, device)
    if not isinstance(outputs, Mapping) or not outputs:
        raise TypeError(
            "deployment_forward returns the model outputs as a non-empty mapping of name -> "
            f"tensor, got {type(outputs).__name__}."
        )
    others = sorted(name for name, value in outputs.items() if not torch.is_tensor(value))
    if others:
        raise TypeError(
            f"deployment_forward returned model outputs that are not tensors: {others}."
        )
    uncalled = [name for name, executor in executors.items() if not executor.ran]
    if uncalled:
        raise RuntimeError(
            f"deployment_forward did not run the declared graphs {uncalled}; every declared "
            "graph is exported, so each has to run."
        )
    return dict(outputs)


def run_deployment_in_torch(
    deployment: DeploymentDeclaration,
    multi_task_batch_inputs: ModelBatchInputs,
    device: torch.device,
) -> dict[str, GraphTensor]:
    """Run the deployed function once in PyTorch, every graph on its own module.

    Modules run in their current mode; export switches them to eval first.

    Args:
        deployment: The model's graphs and deployed function.
        multi_task_batch_inputs: The preprocessed batch, as ``forward()`` takes it.
        device: Device the graphs run on; the inputs of each graph are moved there.

    Returns:
        The model outputs the deployed function returned, by name.

    Raises:
        ValueError: When a graph's module returns another number of tensors than it
            declares, or a mapping in another order.
        TypeError: As :func:`run_deployment`.
        RuntimeError: As :func:`run_deployment`.
    """
    return run_deployment(
        deployment,
        lambda graph: partial(run_graph_module, graph),
        multi_task_batch_inputs,
        device,
    )


def export_spec_for_graph(graph: DeploymentGraph, args: tuple[GraphTensor, ...]) -> ExportSpec:
    """Build one graph's spec from the tensors it ran on.

    The trace inputs are ``args``, the tensors the deployed function passed the graph. The spec
    takes the graph's declared input and output names. Its ``dynamic_axes`` are the graph's
    ``onnx_dynamic_axes``, or ``None`` when the graph declares none.
    """
    dynamic_axes = (
        {name: dict(axes) for name, axes in graph.onnx_dynamic_axes.items()}
        if graph.onnx_dynamic_axes
        else None
    )
    return ExportSpec(
        module=graph.module,
        args=args,
        input_names=list(graph.input_names),
        output_names=list(graph.output_names),
        dynamic_axes=dynamic_axes,
    )


def build_export_specs_from_deployment(
    deployment: DeploymentDeclaration,
    multi_task_batch_inputs: ModelBatchInputs,
    device: torch.device,
) -> dict[str, ExportSpec]:
    """Run the deployed function once in PyTorch and return ``{graph name: ExportSpec}``.

    Each graph is traced on a copy of the tensors the function passed it, taken when the
    graph ran, so what the function does with those tensors afterwards does not reach the
    trace.

    The graph modules are switched to eval first. An export wrapper starts in training
    mode, and ``torch.onnx.export`` restores the module's mode recursively when it is done:
    a wrapper left in training mode would switch the submodules it shares with the model,
    BatchNorm included, to training mode.

    Args:
        deployment: The model's graphs and deployed function.
        multi_task_batch_inputs: One preprocessed example batch.
        device: Device of the tracing run.

    Raises:
        ValueError: As :func:`run_deployment_in_torch`.
        TypeError: As :func:`run_deployment_in_torch`.
        RuntimeError: As :func:`run_deployment_in_torch`.
    """
    trace_inputs: dict[str, tuple[GraphTensor, ...]] = {}

    def runner_for(graph: DeploymentGraph) -> GraphRunner:
        def run(args: tuple[GraphTensor, ...]) -> dict[str, GraphTensor]:
            trace_inputs[graph.name] = tuple(arg.detach().clone() for arg in args)
            return run_graph_module(graph, args)

        return run

    for graph in deployment.graphs:
        graph.module.eval()
    run_deployment(deployment, runner_for, multi_task_batch_inputs, device)
    return {
        graph.name: export_spec_for_graph(graph, trace_inputs[graph.name])
        for graph in deployment.graphs
    }
