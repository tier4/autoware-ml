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
  bookkeeping ...). It always runs in PyTorch, on every backend.

From the declaration, generic code derives the export specs and their trace inputs
(:mod:`.export_specs`) and the artifact names (:func:`artifact_path`); later consumers
(a per-backend pipeline, verification, evaluation) read the same declaration. The final
``GraphStage``'s ``outputs`` are the keys of the model's ``forward()`` output, so a
backend's raw outputs feed the same ``build_eval_output`` as the PyTorch forward.
Nothing model-specific lives outside the model's own ``build_stages``.
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

        This is where the graph's dataflow is actually checked. It cannot be done
        statically in :func:`validate_stages`, because a ``TorchStage`` declares no
        outputs — so the error here names what the run state does hold, which is the
        information a wrong name needs.
        """
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(
                f"Run state has no tensor {name!r}; available: {sorted(self.tensors)}. "
                "A stage reads a name no earlier stage produced."
            ) from error

    def add(self, stage_name: str, produced: Any) -> None:
        """Merge one stage's named results.

        Every name is written once. A stage that re-writes an earlier name would
        silently replace what later stages — and the exported graphs' trace inputs —
        read, so that is an error, as is a result that is not a mapping.

        Raises:
            TypeError: When ``produced`` is not a mapping.
            ValueError: When ``produced`` names a tensor an earlier stage already wrote.
        """
        if not isinstance(produced, Mapping):
            raise TypeError(
                f"Stage {stage_name!r} returned {type(produced).__name__}; a stage's results "
                "are a mapping of name -> value."
            )
        rewritten = sorted(set(produced) & set(self.tensors))
        if rewritten:
            raise ValueError(
                f"Stage {stage_name!r} writes {rewritten}, which an earlier stage already "
                "produced. Every run-state name is written by exactly one stage."
            )
        self.tensors.update(produced)


@dataclass(frozen=True)
class TorchStage:
    """A non-exportable glue stage that always runs in PyTorch.

    Attributes:
        name: Unique stage name (latency breakdown key).
        run: ``fn(state) -> {name: value}``; the returned mapping is merged into the
            run state. Reads earlier results via ``state[name]`` and the raw batch via
            ``state.batch``.
    """

    name: str
    run: Callable[[RunState], Mapping[str, Any]]


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
            in the module's return order (a single tensor return maps to one name). On
            the final graph stage these are also the keys of the model's ``forward()``
            output dict.
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
    """Check a stage declaration and return it as a tuple.

    Reads are *not* checked against writes here, and cannot be: a ``TorchStage``
    declares no outputs, so from the first one onwards what the run state holds is only
    knowable at run time. :meth:`RunState.__getitem__` is where a name that no stage
    produced is caught, and its error lists what *is* available; :meth:`RunState.add`
    is where a glue stage re-writing an earlier name is caught. The static checks are
    the ones the declaration alone decides: the opening stage, whose run state is empty
    by construction, and the graph stages' declared outputs, which must not repeat.

    Not checked: that the final ``GraphStage``'s ``outputs`` are the keys of the model's
    ``forward()`` output. The consumers (pipeline, verification, evaluation) rely on it,
    but only the model knows its forward keys, so each model's own tests pin it.

    Raises:
        ValueError: On duplicate stage names, a name two graph stages both declare as an
            output, no exportable stage, or an opening ``GraphStage`` (nothing has
            produced its inputs yet).
    """
    stages = tuple(stages)
    names = [stage.name for stage in stages]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate stage names: {duplicates}.")
    outputs = [name for stage in graph_stages(stages) for name in stage.outputs]
    repeated = sorted({name for name in outputs if outputs.count(name) > 1})
    if repeated:
        raise ValueError(f"Graph-stage outputs declared more than once: {repeated}.")
    if stages and isinstance(stages[0], GraphStage):
        raise ValueError(
            f"The first stage {stages[0].name!r} is a GraphStage reading "
            f"{list(stages[0].inputs)}, but the run state starts empty. A glue "
            "TorchStage has to put a graph stage's inputs there first."
        )
    if not graph_stages(stages):
        raise ValueError("A stage graph needs at least one exportable GraphStage.")
    return stages


def graph_stages(stages: Sequence[Stage]) -> tuple[GraphStage, ...]:
    """Return the exportable stages in order."""
    return tuple(stage for stage in stages if isinstance(stage, GraphStage))


def final_stage(stages: Sequence[Stage]) -> GraphStage:
    """Return the last exportable stage — its ``outputs`` are the model's forward output keys."""
    return graph_stages(stages)[-1]


#: File suffix of the artifact each backend runs; pytorch runs the modules themselves.
_ARTIFACT_SUFFIX: Final = {Backend.ONNX: ".onnx", Backend.TENSORRT: ".engine"}


def artifact_path(output_dir: str | Path, stage_name: str, backend: Backend) -> Path:
    """Path of a graph stage's exported artifact: ``<output_dir>/<stage_name><suffix>``."""
    suffix = _ARTIFACT_SUFFIX.get(backend)
    if suffix is None:
        raise ValueError(f"The {backend} backend has no exported artifact.")
    return Path(output_dir) / f"{stage_name}{suffix}"


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
        ValueError: When a graph stage returns a different number of tensors than it
            declares as ``outputs``, or a stage re-writes an earlier name.
    """
    stages = validate_stages(stages)
    state = RunState(batch=batch, device=device)
    with torch.no_grad():
        for stage in stages:
            if isinstance(stage, TorchStage):
                state.add(stage.name, stage.run(state))
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
            state.add(stage.name, dict(zip(stage.outputs, raw)))
    return state
