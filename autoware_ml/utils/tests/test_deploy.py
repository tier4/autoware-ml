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
from types import SimpleNamespace

from autoware_ml.utils.deploy import (
    create_optimization_profile,
    ExportSpec,
    build_dynamic_shapes,
    build_dynamic_axes,
    export_to_onnx,
    merge_module_onnx_cfg,
    normalize_dynamic_shapes_for_model,
    resolve_export_specs,
    should_modify_graph,
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
    """A config that names tensors fails up front, naming the entry, shared or per module."""
    validate_onnx_names_not_configured(
        OmegaConf.create({"opset_version": 17, "modules": {"m": {}}})
    )
    for cfg, entry in (
        ({"input_names": ["x"]}, "deploy.onnx.input_names"),
        ({"modules": {"head": {"output_names": ["y"]}}}, "deploy.onnx.modules.head.output_names"),
    ):
        with pytest.raises(ValueError, match=re.escape(entry)):
            validate_onnx_names_not_configured(OmegaConf.create(cfg))


def test_export_to_onnx_prefers_export_spec_dynamic_axes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        torch.onnx,
        "export",
        lambda **kwargs: captured_kwargs.update(kwargs),
    )

    dynamic_axes_override = {"input": {0: "export_spec_batch"}}
    onnx_cfg = OmegaConf.create(
        {
            "opset_version": 17,
            "dynamo": False,
            "dynamic_axes": {"input": {0: "configured_batch"}},
        }
    )

    spec = ExportSpec(
        module=torch.nn.Identity(),
        args=(torch.ones(2, 3),),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_override,
    )
    export_to_onnx(spec, onnx_cfg, tmp_path / "model.onnx")

    assert captured_kwargs["dynamic_axes"] == dynamic_axes_override


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


def test_export_to_onnx_refuses_declared_axes_under_dynamo(tmp_path: Path) -> None:
    # Under dynamo=true these axes would silently be exported static.
    with pytest.raises(ValueError, match="dynamo=false"):
        export_to_onnx(
            ExportSpec(
                module=torch.nn.Identity(),
                args=(torch.ones(2, 3),),
                input_names=["x"],
                output_names=["y"],
                dynamic_axes={"x": {0: "n"}},
            ),
            OmegaConf.create({"dynamo": True}),
            tmp_path / "static.onnx",
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


class _FakeProfile:
    def __init__(self) -> None:
        self.shapes: dict[str, tuple[list[int], list[int], list[int]]] = {}

    def set_shape(self, name: str, min: list[int], opt: list[int], max: list[int]) -> None:
        self.shapes[name] = (min, opt, max)


class _FakeBuilder:
    def create_optimization_profile(self) -> _FakeProfile:
        return _FakeProfile()


class _FakeNetwork:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    @property
    def num_inputs(self) -> int:
        return len(self._names)

    def get_input(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(name=self._names[index])


def test_create_optimization_profile_resolves_the_table_for_the_network_inputs() -> None:
    cfg = OmegaConf.create(
        {
            "max_voxels": 96000,
            "tensorrt": {
                "input_shapes": {
                    "input_features": {
                        "min_shape": [1, 32, 11],
                        "opt_shape": [20000, 32, 11],
                        "max_shape": ["${max_voxels}", 32, 11],
                    },
                    "spatial_features": {
                        "min_shape": [1, 32, 8, 8],
                        "opt_shape": [1, 32, 8, 8],
                        "max_shape": [1, 32, 8, 8],
                    },
                }
            },
        }
    )
    profile = create_optimization_profile(
        _FakeBuilder(), cfg.tensorrt, _FakeNetwork(["input_features"])
    )
    assert profile.shapes == {"input_features": ([1, 32, 11], [20000, 32, 11], [96000, 32, 11])}
    assert all(isinstance(dim, int) for dim in profile.shapes["input_features"][2])
    # A network whose inputs have no entry gets no profile at all.
    assert (
        create_optimization_profile(_FakeBuilder(), cfg.tensorrt, _FakeNetwork(["other"])) is None
    )
    with pytest.raises(ValueError, match="incomplete"):
        create_optimization_profile(
            _FakeBuilder(),
            OmegaConf.create({"input_shapes": {"x": {"min_shape": [1]}}}),
            _FakeNetwork(["x"]),
        )
