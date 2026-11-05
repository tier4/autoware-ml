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

"""Unit tests for deployment helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import re

import onnx
import pytest
import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.datamodule.tests.corpus import build_dataset, write_corpus
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
import autoware_ml.utils.deploy as deploy
from autoware_ml.utils.deploy import (
    ExportSpec,
    apply_spec_modifiers,
    dynamic_shapes_from_spec,
    export_to_onnx,
    merge_module_onnx_cfg,
    normalize_dynamic_shapes_for_model,
    resolve_export_specs,
    should_modify_graph,
    validate_onnx_modules_exported,
    validate_onnx_names_not_configured,
    supports_export_stage,
)


class _DummyModel(torch.nn.Module):
    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        return voxels + num_points.unsqueeze(-1)


def test_export_to_onnx_names_the_tensors_after_the_spec(tmp_path: Path) -> None:
    """The ONNX inputs and outputs carry the spec's names."""
    output_path = tmp_path / "model.onnx"
    spec = ExportSpec(
        module=torch.nn.Identity(),
        args=(torch.ones(2, 3),),
        input_names=["spec_input"],
        output_names=["spec_output"],
    )

    export_to_onnx(spec, OmegaConf.create({"opset_version": 17, "dynamo": False}), output_path)

    graph = onnx.load(str(output_path)).graph
    assert [tensor.name for tensor in graph.input] == ["spec_input"]
    assert [tensor.name for tensor in graph.output] == ["spec_output"]


def test_configured_names_are_refused_when_the_config_is_read() -> None:
    """A config that names tensors or axes fails up front, naming the entry, shared or per module."""
    validate_onnx_names_not_configured(
        OmegaConf.create({"opset_version": 17, "modules": {"m": {}}})
    )
    for cfg, entry in (
        ({"input_names": ["x"]}, "deploy.onnx.input_names"),
        ({"modules": {"head": {"output_names": ["y"]}}}, "deploy.onnx.modules.head.output_names"),
        (
            {"modules": {"head": {"dynamic_axes": {"x": {0: "n"}}}}},
            "deploy.onnx.modules.head.dynamic_axes",
        ),
        ({"dynamic_shapes": {"x": {0: "n"}}}, "deploy.onnx.dynamic_shapes"),
    ):
        with pytest.raises(ValueError, match=re.escape(entry)):
            validate_onnx_names_not_configured(OmegaConf.create(cfg))


def _identity_spec(dynamic_axes: dict[str, dict[int, str]] | None = None) -> ExportSpec:
    return ExportSpec(
        module=torch.nn.Identity(),
        args=(torch.ones(2, 3),),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


def test_export_to_onnx_applies_the_spec_dynamic_axes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The TorchScript exporter gets the axes the spec declares; the config has none."""
    captured_kwargs: dict[str, object] = {}
    monkeypatch.setattr(torch.onnx, "export", lambda **kwargs: captured_kwargs.update(kwargs))
    axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    export_to_onnx(
        _identity_spec(axes),
        OmegaConf.create({"opset_version": 17, "dynamo": False}),
        tmp_path / "m.onnx",
    )

    assert captured_kwargs["dynamic_axes"] == axes


def test_export_to_onnx_derives_the_dynamo_shapes_from_the_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """torch.export gets one Dim per input axis the spec declares; the config has none."""
    captured_kwargs: dict[str, object] = {}
    monkeypatch.setattr(torch.onnx, "export", lambda **kwargs: captured_kwargs.update(kwargs))

    export_to_onnx(
        _identity_spec({"input": {0: "batch"}, "output": {0: "batch"}}),
        OmegaConf.create({"opset_version": 17, "dynamo": True}),
        tmp_path / "m.onnx",
    )

    (shapes,) = captured_kwargs["dynamic_shapes"]
    assert list(shapes) == [0]
    assert shapes[0].__name__ == "batch"
    assert "dynamic_axes" not in captured_kwargs


def test_export_spec_refuses_dynamic_axes_of_unknown_tensors() -> None:
    """The TorchScript exporter only warns about such an axis and exports it static."""
    with pytest.raises(ValueError, match=r"for \['inputs'\], which are not among"):
        _identity_spec({"input": {0: "batch"}, "inputs": {0: "batch"}})


def test_export_spec_names_every_input_and_some_output() -> None:
    """A spec that leaves an input or every output unnamed is refused when it is built."""
    args = (torch.ones(1, 1), torch.ones(1))
    for input_names, output_names, message in (
        (["voxels"], ["out"], "names 1 inputs for 2 args"),
        (["voxels", "num_points", "extra"], ["out"], "names 3 inputs for 2 args"),
        (["voxels", "num_points"], [], "names no outputs"),
    ):
        with pytest.raises(ValueError, match=message):
            ExportSpec(
                module=_DummyModel(),
                args=args,
                input_names=input_names,
                output_names=output_names,
            )


def test_supports_export_stage_uses_export_spec_capabilities() -> None:
    spec = ExportSpec(
        module=_DummyModel(),
        args=(torch.ones(1, 1), torch.ones(1)),
        input_names=["voxels", "num_points"],
        output_names=["out"],
        supported_stages=frozenset({"onnx"}),
    )

    assert supports_export_stage(spec, "onnx") is True
    assert supports_export_stage(spec, "tensorrt") is False


def test_dynamic_shapes_from_spec_follow_the_input_order_and_share_a_dim_per_name() -> None:
    spec = ExportSpec(
        module=torch.nn.Identity(),
        args=(torch.ones(3, 4), torch.ones(3, 3), torch.ones(2, 3), torch.ones(3)),
        input_names=["points", "coors", "voxel_coors", "inverse_map"],
        output_names=["pred_probs"],
        dynamic_axes={
            "points": {0: "num_points"},
            "coors": {0: "num_points"},
            "inverse_map": {0: "num_points"},
            "pred_probs": {0: "num_points"},
        },
    )

    shapes = dynamic_shapes_from_spec(spec)

    assert shapes is not None
    assert len(shapes) == 4
    assert shapes[2] is None
    assert shapes[0] is not None and shapes[0][0].__name__ == "num_points"
    assert shapes[0][0] is shapes[1][0] is shapes[3][0]
    assert dynamic_shapes_from_spec(_identity_spec()) is None


def test_normalize_dynamic_shapes_wraps_varargs_forward() -> None:
    class _VarArgsModel(torch.nn.Module):
        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return args[0]

    dynamic_shapes = ({0: "dim0"}, {0: "dim1"})

    assert normalize_dynamic_shapes_for_model(_VarArgsModel(), dynamic_shapes) == (dynamic_shapes,)


def test_apply_spec_modifiers_runs_them_in_order_and_returns_the_last_path(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path]] = []

    def _modifier(tag: str):
        def modify(onnx_path: Path) -> Path:
            calls.append((tag, onnx_path))
            return onnx_path.with_name(f"{onnx_path.stem}.{tag}.onnx")

        return modify

    spec = ExportSpec(
        module=torch.nn.Identity(),
        args=(torch.zeros(1),),
        input_names=["x"],
        output_names=["y"],
        onnx_modifiers=(_modifier("first"), _modifier("second")),
    )

    final_path = apply_spec_modifiers(spec, tmp_path / "head.onnx")

    assert calls == [("first", tmp_path / "head.onnx"), ("second", tmp_path / "head.first.onnx")]
    assert final_path == tmp_path / "head.first.second.onnx"
    assert (
        apply_spec_modifiers(
            ExportSpec(torch.nn.Identity(), (torch.zeros(1),), ["x"], ["y"]), tmp_path / "head.onnx"
        )
        == tmp_path / "head.onnx"
    )


def test_should_modify_graph_handles_none_and_config() -> None:
    assert should_modify_graph(None) is False
    assert should_modify_graph(OmegaConf.create({"_target_": "pkg.Modifier"})) is True


def test_merge_module_onnx_cfg_overlays_shared_and_module_settings() -> None:
    onnx_cfg = OmegaConf.create(
        {
            "opset_version": 17,
            "dynamo": False,
            "modules": {
                "backbone": {"input_names": ["grid_coord", "feat"]},
                "det3d_head": {"dynamo": True, "input_names": ["point_feat"]},
            },
        }
    )

    backbone = merge_module_onnx_cfg(onnx_cfg, "backbone")
    assert backbone.opset_version == 17  # shared retained
    assert backbone.dynamo is False  # shared retained
    assert backbone.input_names == ["grid_coord", "feat"]  # module-specific
    assert "modules" not in backbone

    head = merge_module_onnx_cfg(onnx_cfg, "det3d_head")
    assert head.opset_version == 17  # shared retained
    assert head.dynamo is True  # module override wins over shared
    assert head.input_names == ["point_feat"]
    assert "modules" not in head


def test_merge_module_onnx_cfg_gives_the_shared_settings_to_a_module_without_an_entry() -> None:
    """A module only needs an entry when it overrides something."""
    for onnx_cfg in (
        OmegaConf.create({"opset_version": 17, "dynamo": False}),
        OmegaConf.create({"opset_version": 17, "dynamo": False, "modules": {"other": {}}}),
    ):
        head = merge_module_onnx_cfg(onnx_cfg, "head")
        assert head.opset_version == 17
        assert head.dynamo is False
        assert "modules" not in head


def test_module_entries_of_modules_the_model_does_not_export_are_refused() -> None:
    """A misspelled entry would silently keep the shared settings."""
    validate_onnx_modules_exported(OmegaConf.create({"opset_version": 17}), ["head"])
    validate_onnx_modules_exported(
        OmegaConf.create({"modules": {"head": {"precision": "fp16"}}}), ["encoder", "head"]
    )
    with pytest.raises(ValueError, match=r"\['heads'\].*\['encoder', 'head'\]"):
        validate_onnx_modules_exported(
            OmegaConf.create({"modules": {"heads": {"precision": "fp16"}}}), ["encoder", "head"]
        )


def test_resolve_export_specs_forwards_batch_and_preserves_order(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _MultiModuleModel(torch.nn.Module):
        def build_export_specs(self, batch: object) -> dict[str, str]:
            captured["batch"] = batch
            return {"backbone": "spec_backbone", "det3d_head": "spec_det3d_head"}

    sentinel_batch = {"feat": torch.zeros(1)}
    monkeypatch.setattr(deploy, "get_predict_batch", lambda dm, model, device: sentinel_batch)

    specs = resolve_export_specs(object(), _MultiModuleModel(), torch.device("cpu"))

    assert list(specs.keys()) == ["backbone", "det3d_head"]  # stable, ordered
    assert captured["batch"] is sentinel_batch  # batch forwarded to the model


def test_get_predict_batch_keeps_the_typed_batch_through_the_device_transfer(
    tmp_path: Path,
) -> None:
    """A collated batch of the real pipeline reaches preprocessing as a ModelGTBatch.

    The record tables are prepared before setup, and the batch is moved with its own transfer,
    so preprocessing reads its named fields.
    """

    dataset = build_dataset(tmp_path, write_corpus(tmp_path, num_records=2))
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    preprocessing = DataPreprocessor(preprocessor_modules=[])

    class _Datamodule:
        def __init__(self) -> None:
            self.prepared = False

        def prepare_data(self) -> None:
            self.prepared = True

        def setup(self, stage: str) -> None:
            assert stage == "predict"
            assert self.prepared

        def predict_dataloader(self) -> list:
            return [batch]

    class _Model:
        def on_after_batch_transfer(self, batch, dataloader_idx: int) -> ModelBatchInputs:
            return preprocessing(batch, is_training=False)

    inputs = deploy.get_predict_batch(_Datamodule(), _Model(), torch.device("cpu"))

    assert inputs.batch_size() == 2
    assert inputs.multi_task_gt_batch.point_cloud_gt_batch is not None
    assert torch.equal(
        inputs.multi_task_gt_batch.point_cloud_gt_batch.points,
        batch.point_cloud_gt_batch.points,
    )
