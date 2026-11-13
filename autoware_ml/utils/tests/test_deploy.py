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

from omegaconf import DictConfig, OmegaConf
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


def test_merge_module_onnx_cfg_gives_the_shared_settings_to_an_empty_entry() -> None:
    """``{}`` and a bare key both mean "no overrides"."""
    for modules in ({"head": {}}, {"head": None}):
        onnx_cfg = OmegaConf.create({"opset_version": 17, "dynamo": False, "modules": modules})
        head = merge_module_onnx_cfg(onnx_cfg, "head")
        assert head.opset_version == 17
        assert head.dynamo is False
        assert "modules" not in head


def test_merge_module_onnx_cfg_refuses_a_module_without_an_entry() -> None:
    """A module added to the model must be listed, or it would silently lose its overrides."""
    for onnx_cfg in (
        OmegaConf.create({"opset_version": 17, "dynamo": False}),
        OmegaConf.create({"opset_version": 17, "dynamo": False, "modules": {"other": {}}}),
    ):
        with pytest.raises(KeyError, match="no entry for exported module 'head'"):
            merge_module_onnx_cfg(onnx_cfg, "head")


def test_every_exported_module_needs_an_entry() -> None:
    """A missing entry drops a module's overrides; a misspelled one shows up as the missing one.

    Entries of modules the model does not export pass: BEVFusion lists the entries of both its
    lidar-only and camera-lidar variants in one base config.
    """
    validate_onnx_modules_exported(
        OmegaConf.create({"modules": {"encoder": {}, "head": {"precision": "fp16"}}}),
        ["encoder", "head"],
    )
    validate_onnx_modules_exported(
        OmegaConf.create({"modules": {"lidar": {}, "image_backbone": {}, "camera_lidar": {}}}),
        ["lidar"],
    )
    with pytest.raises(ValueError, match=r"no entry for the exported modules \['head'\]"):
        validate_onnx_modules_exported(OmegaConf.create({"opset_version": 17}), ["head"])
    with pytest.raises(ValueError, match=r"\['head'\].*entries are \['encoder', 'heads'\]"):
        validate_onnx_modules_exported(
            OmegaConf.create({"modules": {"encoder": {}, "heads": {"precision": "fp16"}}}),
            ["encoder", "head"],
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


class _FakeInput:
    """One input of a parsed TensorRT network: its name and shape, ``-1`` for a dynamic axis."""

    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape


class _FakeNetwork:
    """A parsed TensorRT network, as ``create_optimization_profile`` reads it: its inputs."""

    def __init__(self, inputs: list[_FakeInput]) -> None:
        self._inputs = inputs

    @property
    def num_inputs(self) -> int:
        return len(self._inputs)

    def get_input(self, index: int) -> _FakeInput:
        return self._inputs[index]


class _FakeProfile:
    """A TensorRT optimization profile that records the shapes set on it."""

    def __init__(self) -> None:
        self.shapes: dict[str, tuple[list[int], list[int], list[int]]] = {}

    def set_shape(self, name: str, min: list[int], opt: list[int], max: list[int]) -> None:
        self.shapes[name] = (min, opt, max)


class _FakeBuilder:
    """A TensorRT builder that hands out ``_FakeProfile`` and refuses to build an engine."""

    def create_optimization_profile(self) -> _FakeProfile:
        return _FakeProfile()

    def build_serialized_network(self, network: _FakeNetwork, config: None) -> None:
        del network, config
        pytest.fail("the engine was built")


def _tensorrt_cfg(module_name: str, input_shapes: dict[str, dict[str, list]]) -> DictConfig:
    """The ``deploy.tensorrt`` section with one module's ``input_shapes``."""
    return OmegaConf.create({"modules": {module_name: {"input_shapes": input_shapes}}})


def test_create_optimization_profile_refuses_the_dynamic_inputs_without_an_entry() -> None:
    """Test that a dynamic input without an entry is refused, static inputs are not.

    Input: a network with two dynamic inputs without an entry, one with, one static;
        then a static-only network whose module has no entry at all.
    Expected: ValueError naming ``['coors', 'mask']`` in network order and the config key;
        no profile for the static-only network.
    Check: the error message and the ``None`` return.
    """
    network = _FakeNetwork(
        [
            _FakeInput("voxels", (-1, 32, 5)),
            _FakeInput("coors", (-1, 3)),
            _FakeInput("canvas", (1, 64, 180, 180)),
            _FakeInput("mask", (-1,)),
        ]
    )
    tensorrt_cfg = _tensorrt_cfg(
        "lidar",
        {"voxels": {"min_shape": [1, 32, 5], "opt_shape": [8, 32, 5], "max_shape": [9, 32, 5]}},
    )

    with pytest.raises(
        ValueError,
        match=r"\['coors', 'mask'\] of the 'lidar' graph have no deploy\.tensorrt\.modules\.lidar\.input_shapes",
    ):
        deploy.create_optimization_profile(_FakeBuilder(), tensorrt_cfg, network, "lidar")
    static = _FakeNetwork([_FakeInput("x", (1, 2))])
    assert (
        deploy.create_optimization_profile(_FakeBuilder(), OmegaConf.create({}), static, "lidar")
        is None
    )


def test_create_optimization_profile_refuses_an_entry_of_no_input_and_the_old_table() -> None:
    """Test that a misspelled entry and a table at the old place are refused by key.

    Input: a module entry ``spatial_featues`` for a network whose input is ``spatial_features``;
        then a ``deploy.tensorrt.input_shapes`` table at the top level.
    Expected: ValueError naming the unknown entry and the graph's inputs; ValueError naming
        the new key for the old table.
    Check: the two error messages.
    """
    shapes = {"min_shape": [1, 32, 8, 8], "opt_shape": [1, 32, 8, 8], "max_shape": [1, 32, 8, 8]}
    network = _FakeNetwork([_FakeInput("spatial_features", (-1, 32, -1, -1))])
    with pytest.raises(ValueError, match=r"names \['spatial_featues'\], which are not inputs"):
        deploy.create_optimization_profile(
            _FakeBuilder(), _tensorrt_cfg("head", {"spatial_featues": shapes}), network, "head"
        )
    with pytest.raises(ValueError, match="deploy.tensorrt.modules.head.input_shapes"):
        deploy.create_optimization_profile(
            _FakeBuilder(),
            OmegaConf.create({"input_shapes": {"spatial_features": shapes}}),
            network,
            "head",
        )


def test_build_tensorrt_engine_names_the_missing_profile_before_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a dynamic input without an entry stops the build before the builder runs.

    Input: a parsed network with one dynamic input and a ``deploy.tensorrt`` without entries.
    Expected: ValueError naming the input; the builder is never asked to build.
    Check: the error message, and the fake builder fails the test if it is called.
    """
    builder = _FakeBuilder()
    network = _FakeNetwork([_FakeInput("voxels", (-1, 32, 5))])
    monkeypatch.setattr(
        deploy, "create_tensorrt_builder_config", lambda cfg: (builder, network, None, None)
    )
    monkeypatch.setattr(deploy, "parse_onnx_file", lambda parser, path: None)

    with pytest.raises(ValueError, match=r"dynamic inputs \['voxels'\] of the 'lidar' graph"):
        deploy.build_tensorrt_engine(
            tmp_path / "model.onnx",
            OmegaConf.create({"tensorrt": {}}),
            tmp_path / "model.engine",
            "lidar",
        )


def test_create_optimization_profile_applies_the_module_entries_resolved() -> None:
    """Test that a module's entries build its profile, resolved, and an incomplete one fails.

    Input: a config with a ``${max_voxels}`` reference in one module's entry; the same
        config for a second module that has no entry; an entry missing its max shape.
    Expected: a profile with the entry's shapes as integers; ``None`` for the module without
        an entry (its input is static); ValueError for the incomplete entry.
    Check: the recorded shapes, their types, the return value and the error message.
    """
    builder = _FakeBuilder()
    cfg = OmegaConf.create(
        {
            "max_voxels": 96000,
            "tensorrt": {
                "modules": {
                    "encoder": {
                        "input_shapes": {
                            "input_features": {
                                "min_shape": [1, 32, 11],
                                "opt_shape": [20000, 32, 11],
                                "max_shape": ["${max_voxels}", 32, 11],
                            }
                        }
                    }
                }
            },
        }
    )
    profile = deploy.create_optimization_profile(
        builder, cfg.tensorrt, _FakeNetwork([_FakeInput("input_features", (-1, 32, 11))]), "encoder"
    )
    assert profile.shapes == {"input_features": ([1, 32, 11], [20000, 32, 11], [96000, 32, 11])}
    assert all(isinstance(dim, int) for dim in profile.shapes["input_features"][2])
    head = _FakeNetwork([_FakeInput("spatial_features", (1, 32, 8, 8))])
    assert deploy.create_optimization_profile(builder, cfg.tensorrt, head, "head") is None
    with pytest.raises(ValueError, match=r"modules\.head\.input_shapes\.x is incomplete"):
        deploy.create_optimization_profile(
            builder,
            _tensorrt_cfg("head", {"x": {"min_shape": [1]}}),
            _FakeNetwork([_FakeInput("x", (-1,))]),
            "head",
        )
