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

A model describes its inference as an ordered list of *stages* over a shared bag of
named tensors (the :class:`RunState`):

- :class:`GraphStage` — an exportable sub-graph: an ``nn.Module`` plus the run-state
  names it reads (its ONNX input names) and writes (its ONNX output names). One
  ``GraphStage`` is one ``deploy.onnx.modules.<name>`` entry, one ``<name>.onnx`` and
  one ``<name>.engine``.
- :class:`TorchStage` — glue that is not exported (pillar decoration, BEV scatter, shape
  bookkeeping ...). It always runs in PyTorch, on every backend, and declares the names
  it reads and writes like a graph does, so the whole dataflow is static.

From the declaration, generic code derives the export specs and their trace inputs
(:mod:`.export_specs`) and the artifact names (:func:`artifact_path`); later consumers
(a per-backend pipeline, verification, evaluation) read the same declaration. The graph
outputs no later stage consumes (:func:`model_output_names`) are the *model outputs*: what
a backend hands to the model's ``build_deployed_eval_output``, which turns them into the
flat dict ``build_eval_output`` produces from ``forward()``'s typed outputs. The deployed
graphs may emit something else than ``forward`` does (labels and scores where it returns
logits, decoded boxes where it returns head maps), so that mapping is the model's, next to
its ``build_stages``. Nothing model-specific lives outside the model.
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
    """The dataflow state of one stage-graph run: the named tensors stages read and write.

    Attributes:
        batch: The preprocessed batch the run started from (the ``ModelBatchInputs`` the
            model's ``on_after_batch_transfer`` produced). Only glue reads it; the graph
            stages see run-state tensors.
        device: Device exportable stages execute on; glue stages place their outputs
            here so the next graph stage finds its inputs in place.
        tensors: Name -> value produced so far (graph inputs/outputs and glue results).
    """

    batch: Any
    device: torch.device
    tensors: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Any:
        """Read a produced tensor by name.

        :func:`validate_stages` already checked that every declared input is produced;
        a miss here is a glue stage reading a name it did not declare (it runs on a view
        narrowed to its declared inputs, :meth:`narrowed`), and the error says so.
        """
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(
                f"Run state has no tensor {name!r}; available: {sorted(self.tensors)}. "
                "A stage reads a name it did not declare as an input."
            ) from error

    def narrowed(self, names: Sequence[str]) -> RunState:
        """The view a glue stage runs on: the same batch and device, only ``names``."""
        return RunState(self.batch, self.device, {name: self.tensors[name] for name in names})

    def add(self, produced: Mapping[str, Any]) -> None:
        """Merge one stage's named results (already checked against its declaration when
        the stage ran; :func:`validate_stages` made sure no two stages write the same
        name)."""
        self.tensors.update(produced)


@dataclass(frozen=True)
class TorchStage:
    """A non-exportable glue stage that always runs in PyTorch.

    Attributes:
        name: Unique stage name (latency breakdown key).
        run: ``fn(state) -> {name: value}``; the returned mapping is merged into the
            run state. Reads earlier results via ``state[name]`` and the raw batch via
            ``state.batch``.
        inputs: Run-state names the stage reads (``()`` for glue that reads only the
            batch). It runs on a view narrowed to these, so an undeclared read fails.
        outputs: Run-state names the stage writes; ``run`` must return exactly these.
    """

    name: str
    run: Callable[[RunState], Mapping[str, Any]]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError(f"TorchStage {self.name!r} must declare outputs.")


@dataclass(frozen=True)
class GraphStage:
    """An exportable sub-graph: one ONNX / TensorRT artifact.

    Attributes:
        name: Unique stage name; also the ``deploy.onnx.modules`` key and the artifact
            stem (``<name>.onnx``).
        module: The traced module. Its positional forward arguments are the run-state
            tensors named by ``inputs``, in order.
        inputs: Run-state names fed to the module — these ARE the ONNX input names.
        outputs: Names the module's outputs are written under — the ONNX output names,
            in the module's return order (a single tensor return maps to one name). The
            ones no later stage reads are model outputs (:func:`model_output_names`).
        torch_fallback_backends: Backends on which this stage runs its PyTorch module
            instead of an artifact — for graphs a backend cannot execute, e.g. a
            plugin-op graph on ONNX Runtime.
        onnx_dynamic_axes: Axes this graph makes dynamic *by construction*
            (``{tensor_name: {dim_index: dim_name}}``), for graphs whose dynamic axes are
            a property of the declaration rather than a per-config choice — a point
            model where every tensor is indexed by a point count, say. Becomes the
            derived spec's ``dynamic_axes``; a ``dynamic_axes`` under
            ``deploy.onnx.modules.<name>`` applies when the stage declares none.
    """

    name: str
    module: nn.Module
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    torch_fallback_backends: tuple[Backend, ...] = ()
    onnx_dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.inputs or not self.outputs:
            raise ValueError(f"GraphStage {self.name!r} must declare inputs and outputs.")


Stage = TorchStage | GraphStage


def validate_stages(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """Check a stage declaration's dataflow and return it as a tuple.

    Every stage declares what it reads and writes, so the dataflow is checked here, once,
    from the declaration alone: every input is produced by an earlier stage, every name
    is written by exactly one stage, and at least one graph output is left for the model
    to hand to ``build_deployed_eval_output``. What the run checks is the other half —
    that a stage's module or ``run`` returns what it declares.

    Not checked: that :func:`model_output_names` are what the model's
    ``build_deployed_eval_output`` accepts. Only the model knows that, so each model's own
    tests pin it.

    Raises:
        ValueError: On duplicate stage names, a name written by two stages, a stage
            reading a name no earlier stage wrote, no exportable stage, or no model
            output (every graph output consumed by later glue).
    """
    stages = tuple(stages)
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
    """The model outputs: the graph outputs no later stage (graph or glue) consumes.

    Stage order, then declared order. A chain (``glue -> graph -> glue -> graph``) yields
    its last graph's outputs; several heads over one encoder (``glue -> encoder -> seg ->
    det``) yield every head's outputs, in the order the heads are declared, and not the
    encoder's features the heads read.
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


def run_torch_stage(stage: TorchStage, state: RunState) -> dict[str, Any]:
    """Run a glue stage on the view of the run state it declared and name its results.

    The one place a glue ``run`` is checked against its declaration, shared by the
    reference run below and by every pipeline backend.

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
    """Run every stage with its PyTorch module and return the filled run state.

    This is the reference execution of a stage graph — what export uses to obtain each
    graph stage's trace inputs, and what the ``pytorch`` backend of the inference
    pipeline runs. Graph stages receive the run-state tensors named by their ``inputs``
    (moved to ``device``) as positional arguments, in ``inputs`` order, and must return
    one tensor per declared output, in ``outputs`` order.

    Raises:
        TypeError: When a graph stage returns a mapping (its outputs are positional), or
            a glue stage returns something other than a mapping.
        ValueError: When a stage returns something other than what it declares as
            ``outputs``.
    """
    stages = validate_stages(stages)
    state = RunState(batch=batch, device=device)
    with torch.no_grad():
        for stage in stages:
            if isinstance(stage, TorchStage):
                state.add(run_torch_stage(stage, state))
                continue
            args = tuple(state[name].to(device) for name in stage.inputs)
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
            state.add(dict(zip(stage.outputs, raw)))
    return state
