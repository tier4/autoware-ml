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

"""Stage graph: one declaration of how a model splits into deployable graphs.

Inference is an ordered list of stages over named tensors (:class:`RunState`): a
:class:`GraphStage` is one exported ONNX / TensorRT graph, a :class:`TorchStage` is glue
that runs in PyTorch on every backend. Each stage declares the names it reads and writes,
so the dataflow is checked from the declaration and the export specs derive from it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn

from autoware_ml.types.backend import Backend


@dataclass
class RunState:
    """Named tensors of one stage-graph run.

    Attributes:
        batch: The preprocessed batch the run started from; only glue reads it.
        device: Device the graph stages run on.
        tensors: Tensors produced so far, by name.
        graph_inputs: The positional inputs each graph stage ran on, by stage name, copied
            when the stage ran: later glue cannot change what export traces.
    """

    batch: Any
    device: torch.device
    tensors: dict[str, Any] = field(default_factory=dict)
    graph_inputs: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Any:
        """Return a produced tensor; a miss means a stage read a name it did not declare."""
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(
                f"Run state has no tensor {name!r}; available: {sorted(self.tensors)}. "
                "A stage reads a name it did not declare as an input."
            ) from error

    def narrowed(self, names: Sequence[str]) -> RunState:
        """The view a glue stage runs on: the same batch and device, only ``names``."""
        return RunState(self.batch, self.device, {name: self[name] for name in names})

    def add(self, produced: Mapping[str, Any]) -> None:
        """Merge one stage's named results; a name is written once per run.

        Raises:
            ValueError: On a name the run already holds.
        """
        repeated = sorted(set(produced) & set(self.tensors))
        if repeated:
            raise ValueError(
                f"Run state already holds {repeated}; every name is written by one stage."
            )
        self.tensors.update(produced)


def _reject_bare_string(stage: Stage, kind: str, names: Sequence[str]) -> None:
    """Refuse a ``str`` for a tuple of names: iterated, it would declare one-letter names."""
    if isinstance(names, str):
        raise TypeError(
            f"{type(stage).__name__} {stage.name!r} declares its {kind} as the string "
            f"{names!r}; declare a tuple of names, e.g. ({names!r},)."
        )


@dataclass(frozen=True)
class TorchStage:
    """Glue that is not exported and runs in PyTorch on every backend.

    Attributes:
        name: Unique stage name.
        run: ``fn(state) -> {name: value}`` returning exactly ``outputs``. It must not
            modify what it reads in place: earlier graphs were traced with those tensors.
        inputs: Run-state names the stage reads (``()`` when it reads only
            ``state.batch``); it runs on a view holding only these.
        outputs: Run-state names the stage writes.
    """

    name: str
    run: Callable[[RunState], Mapping[str, Any]]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        _reject_bare_string(self, "inputs", self.inputs)
        _reject_bare_string(self, "outputs", self.outputs)
        if not self.outputs:
            raise ValueError(f"TorchStage {self.name!r} must declare outputs.")


@dataclass(frozen=True)
class GraphStage:
    """One exported ONNX / TensorRT graph.

    Attributes:
        name: Unique stage name, also the ``deploy.onnx.modules`` key and the artifact stem.
        module: The traced module; its positional arguments are the ``inputs`` tensors.
        inputs: Run-state names fed to the module, in order; they are the ONNX input names.
        outputs: ONNX output names, in the module's return order. Those no later stage
            reads are the model outputs (:func:`model_output_names`).
        onnx_dynamic_axes: Axes dynamic by construction, ``{tensor: {dim: name}}``. They
            become the spec's ``dynamic_axes``; the module's config ``dynamic_axes`` apply
            only when the stage declares none.
    """

    name: str
    module: nn.Module
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    onnx_dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _reject_bare_string(self, "inputs", self.inputs)
        _reject_bare_string(self, "outputs", self.outputs)
        if not self.inputs or not self.outputs:
            raise ValueError(f"GraphStage {self.name!r} must declare inputs and outputs.")
        # The exporter renames a repeated name (``x`` -> ``x.1``), so the artifact would not
        # carry the declared names.
        for kind, names in (("inputs", self.inputs), ("outputs", self.outputs)):
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"GraphStage {self.name!r} lists {repeated} more than once in its {kind}; "
                    "each is one ONNX tensor name."
                )
        # The legacy exporter only warns about an axis of an unknown tensor and exports it
        # static.
        unknown = sorted(set(self.onnx_dynamic_axes) - set(self.inputs) - set(self.outputs))
        if unknown:
            raise ValueError(
                f"GraphStage {self.name!r} declares onnx_dynamic_axes for {unknown}, which are "
                f"not among its inputs {list(self.inputs)} or outputs {list(self.outputs)}."
            )


Stage = TorchStage | GraphStage


def validate_stages(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """Check a stage declaration's dataflow and return it as a tuple.

    Every input must be written by an earlier stage, every name by exactly one stage, and at
    least one graph output must be left as a model output. Whether a stage returns what it
    declares is checked when it runs; whether the model outputs are what
    ``build_deployed_eval_output`` reads is left to each model's tests.

    Raises:
        TypeError: On an entry that is not a stage.
        ValueError: On a dataflow violation.
    """
    stages = tuple(stages)
    others = [type(stage).__name__ for stage in stages if not isinstance(stage, Stage)]
    if others:
        raise TypeError(f"A stage graph holds TorchStage and GraphStage entries, got {others}.")
    names = [stage.name for stage in stages]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate stage names: {duplicates}.")
    written: dict[str, str] = {}
    for stage in stages:
        missing = [name for name in stage.inputs if name not in written]
        if missing:
            raise ValueError(
                f"Stage {stage.name!r} reads {missing}, which no earlier stage writes; "
                f"written so far: {sorted(written)}."
            )
        for name in stage.outputs:
            if name in written:
                raise ValueError(
                    f"Stage {stage.name!r} writes {name!r}, which stage {written[name]!r} "
                    "already writes. Every run-state name is written by exactly one stage."
                )
            written[name] = stage.name
    if not graph_stages(stages):
        raise ValueError("A stage graph needs at least one exportable GraphStage.")
    if not model_output_names(stages):
        raise ValueError(
            "No graph output is left as a model output: later glue consumes every one of "
            "them. The declaration has to end with what the model hands to "
            "build_deployed_eval_output."
        )
    return stages


def graph_stages(stages: Sequence[Stage]) -> tuple[GraphStage, ...]:
    """Return the exportable stages in order."""
    return tuple(stage for stage in stages if isinstance(stage, GraphStage))


def model_output_names(stages: Sequence[Stage]) -> tuple[str, ...]:
    """Return the graph outputs no later stage reads, in stage and declaration order.

    These are the model outputs: the last graph's outputs for a chain, every head's outputs
    for several heads over one encoder.
    """
    stages = tuple(stages)
    names: list[str] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, GraphStage):
            continue
        consumed = {name for later in stages[index + 1 :] for name in later.inputs}
        names.extend(name for name in stage.outputs if name not in consumed)
    return tuple(names)


#: File suffix of the artifact each backend runs; pytorch runs the modules themselves.
_ARTIFACT_SUFFIX: Final = {Backend.ONNX: ".onnx", Backend.TENSORRT: ".engine"}


def artifact_path(output_dir: str | Path, stage_name: str, backend: Backend) -> Path:
    """Path of a graph stage's exported artifact: ``<output_dir>/<stage_name><suffix>``."""
    suffix = _ARTIFACT_SUFFIX.get(backend)
    if suffix is None:
        raise ValueError(f"The {backend} backend has no exported artifact.")
    return Path(output_dir) / f"{stage_name}{suffix}"


def graph_input_tensors(stage: GraphStage, state: RunState) -> tuple[torch.Tensor, ...]:
    """Return the stage's positional inputs on the run device, in declaration order.

    Every runner of a stage graph takes a graph's inputs through here.

    Raises:
        TypeError: When an input is not a tensor; a graph binds tensors only.
    """
    tensors = []
    for name in stage.inputs:
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(
                f"GraphStage {stage.name!r} input {name!r} is a {type(value).__name__}; a graph "
                "stage binds tensors only, so the glue that writes it has to produce a tensor."
            )
        tensors.append(value.to(state.device))
    return tuple(tensors)


def run_torch_stage(stage: TorchStage, state: RunState) -> dict[str, Any]:
    """Run a glue stage on its declared inputs and check its results against ``outputs``.

    Every runner of a stage graph runs its glue through here.

    Raises:
        TypeError: When ``run`` returns something other than a mapping.
        ValueError: When the returned names are not exactly the declared ``outputs``.
    """
    produced = stage.run(state.narrowed(stage.inputs))
    if not isinstance(produced, Mapping):
        raise TypeError(
            f"TorchStage {stage.name!r} returned {type(produced).__name__}; a glue stage's "
            "results are a mapping of name -> value."
        )
    if set(produced) != set(stage.outputs):
        raise ValueError(
            f"TorchStage {stage.name!r} returned {sorted(produced)} but declares outputs "
            f"{list(stage.outputs)}."
        )
    return {name: produced[name] for name in stage.outputs}


def run_stages_in_torch(stages: Sequence[Stage], batch: Any, device: torch.device) -> RunState:
    """Run every stage in PyTorch and return the filled run state.

    Export takes its trace inputs from ``RunState.graph_inputs``, filled by this run. A graph
    stage gets its ``inputs`` as positional arguments and returns one tensor per ``outputs``
    name, in order. Modules run in their current mode; export switches them to eval first.

    Raises:
        TypeError: When a graph stage gets a non-tensor input or returns a mapping or a
            non-tensor, or a glue stage returns something other than a mapping.
        ValueError: When a stage returns other names, or another number of tensors, than
            it declares.
    """
    stages = validate_stages(stages)
    state = RunState(batch=batch, device=device)
    with torch.no_grad():
        for stage in stages:
            if isinstance(stage, TorchStage):
                state.add(run_torch_stage(stage, state))
                continue
            args = graph_input_tensors(stage, state)
            state.graph_inputs[stage.name] = tuple(arg.detach().clone() for arg in args)
            raw = stage.module(*args)
            if isinstance(raw, Mapping):
                # len() and zip() would accept a dict and name the outputs by its keys.
                raise TypeError(
                    f"GraphStage {stage.name!r} returned a mapping; a graph stage returns "
                    f"its tensors positionally, in the order of outputs {list(stage.outputs)}."
                )
            if isinstance(raw, torch.Tensor):
                raw = (raw,)
            if len(raw) != len(stage.outputs):
                raise ValueError(
                    f"GraphStage {stage.name!r} returned {len(raw)} tensor(s) but declares "
                    f"outputs {list(stage.outputs)}."
                )
            # The exporter flattens a nested tuple into extra outputs and drops Python values,
            # so neither would match the declared names.
            others = sorted({type(value).__name__ for value in raw if not torch.is_tensor(value)})
            if others:
                raise TypeError(
                    f"GraphStage {stage.name!r} returned {others} among its outputs; a graph "
                    f"stage returns one tensor per declared output {list(stage.outputs)}."
                )
            state.add(dict(zip(stage.outputs, raw)))
    return state
