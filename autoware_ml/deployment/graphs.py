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

A model declares the graphs it exports (:class:`DeploymentGraph`) and writes its deployed
inference as one function, ``deployment_forward(multi_task_batch_inputs, graphs)``: the glue is
plain code in that function, and each graph is run by name. ``multi_task_batch_inputs`` is
the preprocessed batch ``forward()`` takes. The framework hands the function its graphs as
executors (:class:`GraphExecutor`, ``graphs[name].run(**inputs)``), so one function serves
export and every later run, and the model outputs are what the function returns.
:func:`build_export_specs_from_deployment` derives the export specs from one PyTorch run of
that function.

The vocabulary, by lifetime: ``Deployment*`` / ``deployment_*`` is what the model declares
(:class:`DeploymentGraph`, :class:`DeploymentDeclaration`, ``deployment_graphs()``,
``deployment_forward()``); :class:`GraphExecutor` and :class:`DeploymentRun` belong to one run
of it; :class:`~autoware_ml.utils.deploy.ExportSpec` is the export artifact derived from both.
Two things run a graph: a :class:`GraphExecutor` is what the deployed function calls (inputs
by name, checked and recorded), and a :data:`GraphRunner` is what a backend provides to it
(positional tensors in, named tensors out; :func:`run_graph_module` is the PyTorch one).
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
class DeploymentGraph:
    """One exported ONNX / TensorRT graph.

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
            become the spec's ``dynamic_axes``, the one declaration both ONNX exporters
            derive from. One name is one size: an output whose axis differs from an
            input's gets another name.
    """

    name: str
    module: nn.Module
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    onnx_dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The exporter renames a repeated name (``x`` -> ``x.1``), so the artifact would not
        # carry the declared names.
        for kind, names in (("input_names", self.input_names), ("output_names", self.output_names)):
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"DeploymentGraph {self.name!r} lists {repeated} more than once in its "
                    f"{kind}; each is one ONNX tensor name."
                )


def validate_graphs(graphs: Sequence[DeploymentGraph]) -> tuple[DeploymentGraph, ...]:
    """Check a model's graph declaration and return it as a tuple.

    Whether the deployed function runs each graph with its declared inputs is checked when
    it runs.

    Raises:
        ValueError: On no graph, or on two graphs with one name (the artifact's file name).
    """
    graphs = tuple(graphs)
    if not graphs:
        raise ValueError("A deployment needs at least one exportable DeploymentGraph.")
    names = [graph.name for graph in graphs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate graph names: {duplicates}.")
    return graphs


def run_graph_module(graph: DeploymentGraph, args: Sequence[GraphTensor]) -> dict[str, GraphTensor]:
    """Run a graph's PyTorch module on its positional inputs and name its outputs.

    This is the PyTorch :data:`GraphRunner`; a backend runner runs the graph's artifact instead.

    Raises:
        TypeError: When the module returns a mapping; a graph returns its tensors
            positionally, in the order of its declared ``output_names``.
        ValueError: When the module returns another number of tensors than ``output_names``.
    """
    raw = graph.module(*args)
    if isinstance(raw, torch.Tensor):
        raw = (raw,)
    if isinstance(raw, Mapping):
        # zip() would iterate the keys and name the outputs after them.
        raise TypeError(
            f"Graph {graph.name!r} returned a mapping; a graph returns its tensors "
            f"positionally, in the order of its output_names {list(graph.output_names)}."
        )
    raw = tuple(raw)
    if len(raw) != len(graph.output_names):
        raise ValueError(
            f"Graph {graph.name!r} returned {len(raw)} tensors; it declares "
            f"{len(graph.output_names)} outputs {list(graph.output_names)}."
        )
    return dict(zip(graph.output_names, raw, strict=True))


#: What a backend provides to run one graph: its positional inputs in, its outputs by name.
GraphRunner = Callable[[tuple[GraphTensor, ...]], Mapping[str, GraphTensor]]


@dataclass
class _GraphRunRecord:
    """What one graph ran on; written by :meth:`GraphExecutor.run`, read by :func:`run_deployment`.

    Attributes:
        args: The positional inputs of the run, in ``input_names`` order, on the run device;
            ``None`` until the graph ran.
        versions: The in-place version counter of each input when the graph ran.
    """

    args: tuple[GraphTensor, ...] | None = None
    versions: tuple[int, ...] = ()

    def modified_inputs(self, input_names: tuple[str, ...]) -> list[str]:
        """Names of the inputs modified in place since the graph ran."""
        if self.args is None:
            return []
        return [
            name
            for name, arg, version in zip(input_names, self.args, self.versions, strict=True)
            if arg._version != version
        ]


class GraphExecutor:
    """Runs one declared graph on this run's backend, with the tensors in and out by name.

    ``run(**inputs)`` takes exactly ``input_names`` as tensors, moves them to the run device,
    runs the graph and returns its ``output_names`` tensors by name. A graph runs once per
    run; the framework records what it ran on, for export tracing and the run's checks.

    Attributes:
        name: The graph's name, the artifact stem.
        input_names: The graph's input names, in declaration order.
        output_names: The graph's output names, in declaration order.
    """

    def __init__(
        self,
        graph: DeploymentGraph,
        runner: GraphRunner,
        device: torch.device,
        record: _GraphRunRecord,
    ) -> None:
        self.name = graph.name
        self.input_names = graph.input_names
        self.output_names = graph.output_names
        self._runner = runner
        self._device = device
        self._record = record

    def run(self, **inputs: GraphTensor) -> dict[str, GraphTensor]:
        """Run the graph on exactly its declared inputs and return its declared outputs.

        Raises:
            RuntimeError: On a second run in one deployment run; a graph is one artifact,
                run once.
            TypeError: When the keyword names are not the declared ``input_names``, or a
                value is not a tensor.
        """
        name = self.name
        if self._record.args is not None:
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
        self._record.args = args
        self._record.versions = tuple(arg._version for arg in args)
        produced = self._runner(args)
        return {key: produced[key] for key in self.output_names}


class Graphs(Mapping[str, GraphExecutor]):
    """The graphs of one run, by name, as the deployed function receives them.

    Attributes:
        device: Device the graphs run on; the function moves the tensors it makes there.
    """

    def __init__(self, executors: Mapping[str, GraphExecutor], device: torch.device) -> None:
        self._executors = dict(executors)
        self.device = device

    def __getitem__(self, name: str) -> GraphExecutor:
        try:
            return self._executors[name]
        except KeyError:
            raise KeyError(
                f"No graph {name!r}; the model declares {sorted(self._executors)}."
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._executors)

    def __len__(self) -> int:
        return len(self._executors)


#: ``deployment_forward(multi_task_batch_inputs, graphs) -> {output name: tensor}``.
DeploymentForward = Callable[[ModelBatchInputs, Graphs], Mapping[str, GraphTensor]]


@dataclass(frozen=True)
class DeploymentDeclaration:
    """A model's deployment: the graphs it exports and the function that runs them.

    Attributes:
        graphs: The exported graphs, checked by :func:`validate_graphs`.
        forward: The deployed inference,
            ``forward(multi_task_batch_inputs, graphs) -> model outputs``.
    """

    graphs: tuple[DeploymentGraph, ...]
    forward: DeploymentForward

    def __post_init__(self) -> None:
        object.__setattr__(self, "graphs", validate_graphs(self.graphs))


class DeploymentRun(NamedTuple):
    """One run of a deployment, on any backend.

    Attributes:
        outputs: The model outputs the deployed function returned, by name.
        graph_inputs: The logical inputs of each graph run, by graph name: the tensors the
            deployed function passed ``run()``, moved to the run device, in ``input_names``
            order. Export traces each graph on them. They are not necessarily the tensors a
            backend finally bound to its artifact: an ONNX Runtime or TensorRT runner may
            feed a host, contiguous or re-typed copy of them.
    """

    outputs: dict[str, GraphTensor]
    graph_inputs: dict[str, tuple[GraphTensor, ...]]


def run_deployment(
    deployment: DeploymentDeclaration,
    runner_for: Callable[[DeploymentGraph], GraphRunner],
    multi_task_batch_inputs: ModelBatchInputs,
    device: torch.device,
) -> DeploymentRun:
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
        The model outputs and the logical inputs each graph ran on.

    Raises:
        TypeError: When a graph gets a non-tensor input, or the function returns something
            other than a non-empty mapping of tensors.
        ValueError: When the function modifies a graph input in place after the graph ran;
            export traces the graph on that tensor.
        RuntimeError: When the function skips a declared graph or runs one twice.
    """
    records = {graph.name: _GraphRunRecord() for graph in deployment.graphs}
    graphs = Graphs(
        {
            graph.name: GraphExecutor(graph, runner_for(graph), device, records[graph.name])
            for graph in deployment.graphs
        },
        device,
    )
    with torch.no_grad():
        outputs = deployment.forward(multi_task_batch_inputs, graphs)
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
    uncalled = [name for name, record in records.items() if record.args is None]
    if uncalled:
        raise RuntimeError(
            f"deployment_forward did not run the declared graphs {uncalled}; every declared "
            "graph is exported, so each has to run."
        )
    modified = {
        graph.name: records[graph.name].modified_inputs(graph.input_names)
        for graph in deployment.graphs
    }
    modified = {name: names for name, names in modified.items() if names}
    if modified:
        raise ValueError(
            f"deployment_forward modified graph inputs in place after the graphs ran: "
            f"{modified}; export traces each graph on the tensors it ran on."
        )
    return DeploymentRun(dict(outputs), {name: record.args for name, record in records.items()})


def run_deployment_in_torch(
    deployment: DeploymentDeclaration,
    multi_task_batch_inputs: ModelBatchInputs,
    device: torch.device,
) -> DeploymentRun:
    """Run the deployed function once in PyTorch, every graph on its own module.

    Export takes its trace inputs from ``graph_inputs``. Modules run in their current mode;
    export switches them to eval first.

    Raises:
        TypeError: When a graph returns a mapping, or as :func:`run_deployment`.
        ValueError: When a graph returns another number of tensors than it declares, or as
            :func:`run_deployment`.
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
    for graph in deployment.graphs:
        graph.module.eval()
    run = run_deployment_in_torch(deployment, multi_task_batch_inputs, device)
    return {
        graph.name: export_spec_for_graph(graph, run.graph_inputs[graph.name])
        for graph in deployment.graphs
    }
