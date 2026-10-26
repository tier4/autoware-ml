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

"""``build_stages()`` derives ``build_export_specs()``; models without stages are untouched."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from autoware_ml.deployment.export_specs import build_export_specs_from_stages
from autoware_ml.deployment.stages import GraphStage, Stage, TorchStage
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec, export_to_onnx
from autoware_ml.types.backend import Backend


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2) * 2)

    def forward(self, features):
        return self.linear(features)


class _Head(nn.Module):
    def forward(self, bev):
        return bev + 1, bev - 1


class _ToyModel(ModuleBaseModel):
    """The toy models take a plain dict batch; the base class only needs the two ctor args."""

    def __init__(self) -> None:
        super().__init__(
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        )

    def decode_outputs(self, multi_task_batch_inputs: Mapping[str, Any], outputs: Any) -> Any:
        del multi_task_batch_inputs
        return outputs


class _StagedModel(_ToyModel):
    """Two exported graphs with glue between them, declared once as a stage graph."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.head = _Head()

    def forward(self, multi_task_batch_inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        bev = self.encoder(multi_task_batch_inputs["points"].float()).sum(dim=0, keepdim=True)
        plus, minus = self.head(bev)
        return {"plus": plus, "minus": minus}

    def compute_metrics(self, multi_task_batch_inputs: Mapping[str, Any], outputs: Any):
        del multi_task_batch_inputs
        return {"loss": outputs["plus"].sum()}

    def build_stages(self) -> Sequence[Stage]:
        def prep(state):
            return {"features": state.batch["points"].float()}

        def scatter(state):
            return {"bev": state["encoded"].sum(dim=0, keepdim=True)}

        return (
            TorchStage("prep", run=prep, inputs=(), outputs=("features",)),
            GraphStage("encoder", module=self.encoder, inputs=("features",), outputs=("encoded",)),
            TorchStage("scatter", run=scatter, inputs=("encoded",), outputs=("bev",)),
            GraphStage("head", module=self.head, inputs=("bev",), outputs=("plus", "minus")),
        )


class _PlainModel(_ToyModel):
    def forward(self, multi_task_batch_inputs: Mapping[str, Any]) -> torch.Tensor:
        return multi_task_batch_inputs["x"] + 1.0

    def compute_metrics(self, multi_task_batch_inputs: Mapping[str, Any], outputs: Any):
        del multi_task_batch_inputs
        return {"loss": outputs.sum()}

    def build_export_spec(self, batch_inputs: Mapping[str, Any]) -> ExportSpec:
        return ExportSpec(
            module=self, args=(batch_inputs["x"],), input_names=["x"], output_names=["y"]
        )


def _batch() -> dict[str, torch.Tensor]:
    return {"points": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "unused": torch.zeros(1)}


def test_specs_are_derived_one_per_graph_stage_in_order() -> None:
    model = _StagedModel().eval()
    specs = model.build_export_specs(_batch())

    assert list(specs) == ["encoder", "head"]
    encoder, head = specs["encoder"], specs["head"]
    assert encoder.input_names == ["features"]
    assert encoder.output_names == ["encoded"]
    assert head.input_names == ["bev"]
    assert head.output_names == ["plus", "minus"]
    assert encoder.module is model.encoder
    assert head.module is model.head
    assert encoder.dynamic_axes is None
    assert encoder.supported_stages == frozenset({"onnx", "tensorrt"})


def test_trace_inputs_are_the_run_state_tensors_the_glue_produced() -> None:
    model = _StagedModel().eval()
    specs = model.build_export_specs(_batch())

    (features,) = specs["encoder"].args
    assert torch.equal(features, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    (bev,) = specs["head"].args
    # encoder doubles, scatter sums the rows: [[2,4],[6,8]] -> [[8,12]]
    assert torch.equal(bev, torch.tensor([[8.0, 12.0]]))
    # Running the exported modules on their own trace inputs reproduces forward().
    plus, minus = specs["head"].module(*specs["head"].args)
    forward = model(_batch())
    assert torch.equal(plus, forward["plus"]) and torch.equal(minus, forward["minus"])


def test_stage_declared_axes_land_in_the_spec_and_fallback_keeps_export_targets() -> None:
    stages = (
        TorchStage("seed", run=lambda state: {"x": state.batch["x"]}, inputs=(), outputs=("x",)),
        GraphStage(
            "points",
            module=nn.Identity(),
            inputs=("x",),
            outputs=("y",),
            onnx_dynamic_axes={"x": {0: "num_points"}, "y": {0: "num_points"}},
            torch_fallback_backends=(Backend.ONNX,),
        ),
    )
    specs = build_export_specs_from_stages(stages, {"x": torch.ones(3, 2)}, torch.device("cpu"))
    spec = specs["points"]
    assert spec.dynamic_axes == {"x": {0: "num_points"}, "y": {0: "num_points"}}
    # A runtime fallback says where the artifact is *run*, not whether it is exported.
    assert spec.supported_stages == frozenset({"onnx", "tensorrt"})


def test_a_stage_graph_refuses_to_trace_in_training_mode() -> None:
    model = _StagedModel().train()
    with pytest.raises(RuntimeError, match="eval mode"):
        model.build_export_specs(_batch())


def test_a_stage_graph_refuses_to_trace_with_a_submodule_in_training_mode() -> None:
    # The model's own flag says eval, one of its submodules says training.
    model = _StagedModel().eval()
    model.encoder.train()
    with pytest.raises(RuntimeError, match=r"eval mode.*\['encoder', 'encoder.linear'\]"):
        model.build_export_specs(_batch())


class _Wrapper(nn.Module):
    """An export wrapper around a model submodule; it starts in nn.Module's training mode."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def test_export_leaves_the_submodules_a_wrapper_shares_in_eval_mode(tmp_path: Path) -> None:
    shared = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).eval()
    stages = (
        TorchStage("seed", run=lambda state: {"x": state.batch["x"]}, inputs=(), outputs=("x",)),
        GraphStage("wrapped", module=_Wrapper(shared), inputs=("x",), outputs=("y",)),
    )
    spec = build_export_specs_from_stages(stages, {"x": torch.rand(4, 2)}, torch.device("cpu"))[
        "wrapped"
    ]
    assert not spec.module.training

    export_to_onnx(
        spec, OmegaConf.create({"dynamo": False, "opset_version": 17}), tmp_path / "wrapped.onnx"
    )
    # The exporter hands the wrapper its own mode back, recursively: a wrapper left in
    # training mode would have put the shared BatchNorm in training mode as well.
    assert not any(module.training for module in shared.modules())


def test_multi_input_spec_args_follow_the_declared_input_order() -> None:
    # ONNX binds inputs by position: args, input names and the module's parameters must
    # all follow ``inputs``, whatever order the glue produced the tensors in.
    class _Subtract(nn.Module):
        def forward(self, a, b):
            return a - b

    stages = (
        TorchStage(
            "seed",
            run=lambda state: {"b": torch.tensor([1.0]), "a": torch.tensor([10.0])},
            inputs=(),
            outputs=("b", "a"),
        ),
        GraphStage("sub", module=_Subtract(), inputs=("a", "b"), outputs=("diff",)),
    )
    spec = build_export_specs_from_stages(stages, None, torch.device("cpu"))["sub"]
    assert spec.input_names == ["a", "b"]
    assert [arg.item() for arg in spec.args] == [10.0, 1.0]
    assert torch.equal(spec.module(*spec.args), torch.tensor([9.0]))


def test_models_without_a_stage_graph_keep_the_end_to_end_default() -> None:
    model = _PlainModel()
    assert model.build_stages() is None
    specs = model.build_export_specs({"x": torch.tensor([1.0])})
    assert list(specs) == ["end_to_end"]
    assert specs["end_to_end"].input_names == ["x"]


def test_a_stage_reading_an_unwritten_name_fails_before_anything_runs() -> None:
    stages = (
        TorchStage("seed", run=lambda state: {"x": state.batch["x"]}, inputs=(), outputs=("x",)),
        GraphStage("g", module=nn.Identity(), inputs=("typo",), outputs=("y",)),
    )
    with pytest.raises(ValueError, match="'g' reads \\['typo'\\].*written so far: \\['x'\\]"):
        build_export_specs_from_stages(stages, {"x": torch.ones(1)}, torch.device("cpu"))
