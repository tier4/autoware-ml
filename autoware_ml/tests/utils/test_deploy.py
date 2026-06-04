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
import pytest
import torch

import autoware_ml.utils.deploy as deploy
from autoware_ml.utils.deploy import (
    ExportSpec,
    build_dynamic_shapes,
    build_dynamic_axes,
    export_to_onnx,
    get_export_parameter_names,
    merge_module_onnx_cfg,
    normalize_dynamic_shapes_for_model,
    resolve_export_specs,
    should_modify_graph,
    supports_export_stage,
)


class _DummyModel(torch.nn.Module):
    def forward(
        self, voxels: torch.Tensor, num_points: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        return voxels + num_points.unsqueeze(-1)


def test_get_export_parameter_names_ignores_variadic_parameters() -> None:
    model = _DummyModel()

    assert get_export_parameter_names(model) == ["voxels", "num_points"]


def test_export_to_onnx_prefers_export_spec_output_names(tmp_path: Path) -> None:
    class _SingleOutput(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    output_path = tmp_path / "model.onnx"
    onnx_cfg = OmegaConf.create(
        {
            "opset_version": 17,
            "dynamo": False,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["configured_output"],
        }
    )

    export_to_onnx(
        model=_SingleOutput(),
        input_sample=(torch.ones(2, 3),),
        onnx_cfg=onnx_cfg,
        input_param_names=["input"],
        output_names_override=["exported_output"],
        output_path=output_path,
    )

    assert output_path.exists()


def test_supports_export_stage_uses_export_spec_capabilities() -> None:
    spec = ExportSpec(
        module=_DummyModel(),
        args=(torch.ones(1, 1), torch.ones(1)),
        input_param_names=["voxels", "num_points"],
        supported_stages=frozenset({"onnx"}),
    )

    assert supports_export_stage(spec, "onnx") is True
    assert supports_export_stage(spec, "tensorrt") is False


def test_build_dynamic_axes_supports_legacy_export_config() -> None:
    deploy_cfg = OmegaConf.create(
        {
            "dynamic_axes": {
                "feat": {0: "voxels_num"},
                "pred_probs": {0: "voxels_num"},
            }
        }
    )

    assert build_dynamic_axes(deploy_cfg) == {
        "feat": {0: "voxels_num"},
        "pred_probs": {0: "voxels_num"},
    }


def test_build_dynamic_axes_falls_back_to_dynamic_shapes() -> None:
    deploy_cfg = OmegaConf.create(
        {
            "dynamic_shapes": {
                "points": {0: {"name": "num_points", "min": 2}},
                "inverse_map": {0: {"name": "num_points", "min": 2}},
            }
        }
    )

    assert build_dynamic_axes(deploy_cfg) == {
        "points": {0: "num_points"},
        "inverse_map": {0: "num_points"},
    }


def test_build_dynamic_shapes_matches_positional_export_inputs() -> None:
    deploy_cfg = OmegaConf.create(
        {
            "dynamic_shapes": {
                "points": {0: {"name": "num_points", "min": 2}},
                "coors": {0: {"name": "num_points", "min": 2}},
                "inverse_map": {0: {"name": "num_points", "min": 2}},
            }
        }
    )

    dynamic_shapes = build_dynamic_shapes(
        deploy_cfg,
        ["points", "coors", "voxel_coors", "inverse_map"],
    )

    assert dynamic_shapes is not None
    assert len(dynamic_shapes) == 4
    assert dynamic_shapes[0] is not None
    assert dynamic_shapes[1] is not None
    assert dynamic_shapes[2] is None
    assert dynamic_shapes[3] is not None


def test_normalize_dynamic_shapes_wraps_varargs_forward() -> None:
    class _VarArgsModel(torch.nn.Module):
        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return args[0]

    dynamic_shapes = ({0: "dim0"}, {0: "dim1"})

    assert normalize_dynamic_shapes_for_model(_VarArgsModel(), dynamic_shapes) == (dynamic_shapes,)


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


def test_merge_module_onnx_cfg_raises_for_unknown_module() -> None:
    onnx_cfg = OmegaConf.create({"opset_version": 17, "modules": {"backbone": {}}})
    with pytest.raises(KeyError):
        merge_module_onnx_cfg(onnx_cfg, "missing")


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
