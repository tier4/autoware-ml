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

"""Unit tests for :class:`DeploymentExport` on CPU: module selection from the deploy config,
the output layout, the ONNX files it writes with their names, dynamic axes and metadata stamp,
and the error paths that fire before any TensorRT work."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import onnx
import onnxruntime
from omegaconf import DictConfig, OmegaConf
import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.deployment.deployment_export import DeploymentExport, DeploymentExportOutputs
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.mlflow_helpers import get_git_sha
from autoware_ml.utils.onnx_meta import MODEL_DOMAIN, PRODUCER_NAME

DEPLOY_YAML = (
    Path(__file__).resolve().parents[2] / "configs" / "defaults" / "modules" / "deploy.yaml"
)
CONFIG_NAME = "detection3d/tiny/tiny_b1_1e_corpus"
NUM_POINTS = 6
OTHER_NUM_POINTS = 12
POINT_DIM = 4
FEATURE_DIM = 2
ALL_STAGES = frozenset({"onnx", "tensorrt"})


class _HeadModule(torch.nn.Module):
    """Two-input module whose first input has a dynamic point axis."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(POINT_DIM, FEATURE_DIM)

    def forward(self, points: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return torch.mul(self.linear(points), scale)


class _ExportModel(ModuleBaseModel):
    """A model exporting its head once per configured module name."""

    def __init__(
        self, module_names: Sequence[str] = ("head",), supported_stages: frozenset[str] = ALL_STAGES
    ) -> None:
        super().__init__(
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        )
        self.head = _HeadModule()
        self.module_names = tuple(module_names)
        self.supported_stages = supported_stages

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> torch.Tensor:
        point_batch = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        assert point_batch is not None
        return self.head(point_batch.points, torch.tensor(2.0))

    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del multi_task_batch_inputs
        return {"loss": multi_task_outputs.sum()}

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: torch.Tensor
    ) -> torch.Tensor:
        del multi_task_batch_inputs
        return outputs

    def build_export_specs(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> dict[str, ExportSpec]:
        point_batch = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        assert point_batch is not None
        spec = ExportSpec(
            module=self.head,
            args=(point_batch.points, torch.tensor(2.0)),
            input_param_names=["points", "scale"],
            output_names=["feat"],
            dynamic_axes={"points": {0: "num_points"}, "feat": {0: "num_points"}},
            supported_stages=self.supported_stages,
        )
        return {name: spec for name in self.module_names}


class _NoInputsModel(_ExportModel):
    """A model whose export spec carries no example inputs."""

    def build_export_specs(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> dict[str, ExportSpec]:
        return {
            "head": ExportSpec(
                module=self.head, args=(), input_param_names=[], output_names=["feat"]
            )
        }


class _PredictDatamodule:
    """The predict-split contract the export reads: prepare, set up, then one dataloader."""

    def __init__(self, batches: Sequence[ModelGTBatch]) -> None:
        self.batches = list(batches)
        self.prepared = False
        self.stage: str | None = None

    def prepare_data(self) -> None:
        self.prepared = True

    def setup(self, stage: str) -> None:
        assert self.prepared
        self.stage = stage

    def predict_dataloader(self) -> list[ModelGTBatch]:
        return self.batches


def _point_batch() -> ModelGTBatch:
    """One sample of ``NUM_POINTS`` points, the example batch the export traces with."""
    generator = torch.Generator().manual_seed(0)
    points = PointCloudGTBatch(
        points=torch.rand(NUM_POINTS, POINT_DIM, generator=generator),
        batch_indices=torch.zeros(NUM_POINTS, dtype=torch.int32),
        batch_size=1,
        timestamp_difference_dim=-1,
    )
    return ModelGTBatch(
        point_cloud_gt_batch=points,
        detection3d_gt_batch=None,
        segmentation3d_gt_batch=None,
        image_gt_batch=None,
        padding_mask=torch.zeros(1, dtype=torch.bool),
    )


def _deploy_cfg(
    modules: dict, onnx_enabled: bool = True, tensorrt_enabled: bool = False
) -> DictConfig:
    """The ``deploy`` section of the default module config with the per-module settings."""
    cfg = OmegaConf.load(DEPLOY_YAML).deploy
    cfg.onnx.enabled = onnx_enabled
    cfg.onnx.modules = modules
    cfg.tensorrt.enabled = tensorrt_enabled
    return cfg


def _exporter(
    tmp_path: Path,
    deploy_cfg: DictConfig,
    model: ModuleBaseModel | None = None,
    datamodule: _PredictDatamodule | None = None,
    release: str | None = None,
    tracker: str | None = None,
    run_id: str | None = None,
) -> DeploymentExport:
    return DeploymentExport(
        deploy_cfg=deploy_cfg,
        output_dir=tmp_path / "exports",
        datamodule=datamodule if datamodule is not None else _PredictDatamodule([_point_batch()]),
        model=model if model is not None else _ExportModel(),
        config_name=CONFIG_NAME,
        release=release,
        tracker=tracker,
        run_id=run_id,
    )


def _onnx_opset(model: onnx.ModelProto) -> int:
    return next(entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx"))


def _dim_params(value_info: onnx.ValueInfoProto) -> list[str | int]:
    return [
        dim.dim_param if dim.dim_param else dim.dim_value
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _metadata(model: onnx.ModelProto) -> dict[str, str]:
    return {prop.key: prop.value for prop in model.metadata_props}


class TestOutputLayout:
    """The exporter owns its output directory and names the files after the modules."""

    def test_output_dir_is_created_on_construction(self, tmp_path: Path) -> None:
        """
        Input: an output directory that does not exist yet.
        Expected: the constructor creates it, with parents, before any export.
        Check: the directory exists and the device is the model's.
        """
        exporter = DeploymentExport(
            deploy_cfg=_deploy_cfg({"head": {}}),
            output_dir=tmp_path / "exports" / "nested",
            datamodule=_PredictDatamodule([_point_batch()]),
            model=_ExportModel(),
            config_name=CONFIG_NAME,
            release=None,
        )

        assert (tmp_path / "exports" / "nested").is_dir()
        assert exporter.device == torch.device("cpu")

    def test_onnx_paths_follow_the_module_names_in_spec_order(self, tmp_path: Path) -> None:
        """
        Input: a model exporting ``encoder`` then ``head`` with both modules configured.
        Expected: one ``<module>.onnx`` per module under the output directory, reported in the
            order of the export specs, and no TensorRT engine while that stage is disabled.
        Check: the outputs and the files on disk.
        """
        model = _ExportModel(module_names=("encoder", "head"))
        exporter = _exporter(tmp_path, _deploy_cfg({"encoder": {}, "head": {}}), model=model)

        outputs = exporter.export()

        assert isinstance(outputs, DeploymentExportOutputs)
        assert list(outputs.onnx_exported_paths) == [
            tmp_path / "exports" / "encoder.onnx",
            tmp_path / "exports" / "head.onnx",
        ]
        assert list(outputs.tensorrt_exported_paths) == []
        assert sorted(path.name for path in (tmp_path / "exports").iterdir()) == [
            "encoder.onnx",
            "head.onnx",
        ]

    def test_disabled_stages_export_nothing(self, tmp_path: Path) -> None:
        """
        Input: both stages disabled.
        Expected: the export completes with empty outputs and an empty directory.
        Check: the outputs and the directory listing.
        """
        exporter = _exporter(tmp_path, _deploy_cfg({"head": {}}, onnx_enabled=False))

        outputs = exporter.export()

        assert list(outputs.onnx_exported_paths) == []
        assert list(outputs.tensorrt_exported_paths) == []
        assert list((tmp_path / "exports").iterdir()) == []


class TestExportSpecs:
    """The export specs come from one preprocessed predict batch."""

    def test_prepare_export_specs_reads_the_predict_split(self, tmp_path: Path) -> None:
        """
        Input: a datamodule serving one batch.
        Expected: the datamodule is prepared and set up for ``predict``, and the read-only spec
            mapping carries the model's head with the points of that batch as example input.
        Check: the datamodule state, the mapping type and the spec contents.
        """
        batch = _point_batch()
        datamodule = _PredictDatamodule([batch])
        model = _ExportModel()
        exporter = _exporter(
            tmp_path, _deploy_cfg({"head": {}}), model=model, datamodule=datamodule
        )

        specs = exporter.prepare_export_specs()

        assert datamodule.prepared
        assert datamodule.stage == "predict"
        assert isinstance(specs, MappingProxyType)
        assert list(specs) == ["head"]
        assert specs["head"].module is model.head
        assert specs["head"].input_param_names == ["points", "scale"]
        assert torch.equal(specs["head"].args[0], batch.point_cloud_gt_batch.points)


class TestOnnxExport:
    """The ONNX files carry the configured names, axes, opset and metadata."""

    def test_legacy_export_keeps_the_spec_names_and_dynamic_axes(self, tmp_path: Path) -> None:
        """
        Input: the default deploy config with ``dynamo`` disabled for the module.
        Expected: the graph inputs and outputs carry the spec names, the point axis is dynamic
            on both the input and the output, and the opset is the configured one.
        Check: the loaded ONNX graph, validated by the ONNX checker.
        """
        deploy_cfg = _deploy_cfg({"head": {"dynamo": False}})
        exporter = _exporter(tmp_path, deploy_cfg)

        outputs = exporter.export()

        onnx_model = onnx.load(str(outputs.onnx_exported_paths[0]))
        onnx.checker.check_model(onnx_model)
        assert [value.name for value in onnx_model.graph.input] == ["points", "scale"]
        assert [value.name for value in onnx_model.graph.output] == ["feat"]
        assert _dim_params(onnx_model.graph.input[0]) == ["num_points", POINT_DIM]
        assert _dim_params(onnx_model.graph.output[0]) == ["num_points", FEATURE_DIM]
        assert _onnx_opset(onnx_model) == deploy_cfg.onnx.opset_version

    def test_dynamo_export_uses_the_configured_dynamic_shapes(self, tmp_path: Path) -> None:
        """
        Input: the default deploy config (``dynamo`` enabled) with a ``dynamic_shapes`` block
            naming the point axis.
        Expected: the exported input carries that dynamic axis and the spec names.
        Check: the loaded ONNX graph.
        """
        deploy_cfg = _deploy_cfg({"head": {"dynamic_shapes": {"points": {0: "num_points"}}}})
        exporter = _exporter(tmp_path, deploy_cfg)

        outputs = exporter.export()

        onnx_model = onnx.load(str(outputs.onnx_exported_paths[0]))
        assert [value.name for value in onnx_model.graph.input] == ["points", "scale"]
        assert [value.name for value in onnx_model.graph.output] == ["feat"]
        assert _dim_params(onnx_model.graph.input[0]) == ["num_points", POINT_DIM]
        assert _onnx_opset(onnx_model) == deploy_cfg.onnx.opset_version

    def test_exported_graph_computes_the_module_on_another_point_count(
        self, tmp_path: Path
    ) -> None:
        """
        Input: the legacy export run on more points than the example batch.
        Expected: the graph honours the dynamic axis and matches the module's output.
        Check: ONNX Runtime output against the torch module.
        """
        model = _ExportModel()
        exporter = _exporter(tmp_path, _deploy_cfg({"head": {"dynamo": False}}), model=model)
        outputs = exporter.export()
        points = torch.rand(OTHER_NUM_POINTS, POINT_DIM, generator=torch.Generator().manual_seed(1))
        scale = torch.tensor(2.0)

        session = onnxruntime.InferenceSession(
            str(outputs.onnx_exported_paths[0]), providers=["CPUExecutionProvider"]
        )
        feat = session.run(["feat"], {"points": points.numpy(), "scale": scale.numpy()})[0]

        with torch.no_grad():
            expected = model.head(points, scale).numpy()
        assert feat.shape == (OTHER_NUM_POINTS, FEATURE_DIM)
        np.testing.assert_allclose(feat, expected, rtol=1e-5, atol=1e-6)

    def test_module_settings_override_the_shared_onnx_settings(self, tmp_path: Path) -> None:
        """
        Input: two modules, one overriding the shared opset version.
        Expected: each module is exported with its own merged settings.
        Check: the opset of each ONNX file.
        """
        model = _ExportModel(module_names=("encoder", "head"))
        deploy_cfg = _deploy_cfg({"encoder": {"opset_version": 17}, "head": {}})
        exporter = _exporter(tmp_path, deploy_cfg, model=model)

        outputs = exporter.export()

        encoder_path, head_path = outputs.onnx_exported_paths
        assert _onnx_opset(onnx.load(str(encoder_path))) == 17
        assert _onnx_opset(onnx.load(str(head_path))) == deploy_cfg.onnx.opset_version

    def test_fp16_precision_converts_the_weights_and_keeps_fp32_io(self, tmp_path: Path) -> None:
        """
        Input: a module configured with ``precision: fp16``.
        Expected: the initializers are stored in fp16 while the graph boundary stays fp32, and
            the metadata stamp lands on the converted file.
        Check: the initializer and I/O element types and the ``module`` property.
        """
        exporter = _exporter(
            tmp_path, _deploy_cfg({"head": {"dynamo": False, "precision": "fp16"}})
        )

        outputs = exporter.export()

        onnx_model = onnx.load(str(outputs.onnx_exported_paths[0]))
        assert {init.data_type for init in onnx_model.graph.initializer} == {
            onnx.TensorProto.FLOAT16
        }
        assert onnx_model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert onnx_model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert _metadata(onnx_model)["module"] == "head"


class TestMetadataStamp:
    """Every ONNX module carries its identity and provenance."""

    def test_release_export_stamps_identity_provenance_and_metainfo(self, tmp_path: Path) -> None:
        """
        Input: a versioned export with a tracker run and a ``metainfo`` block.
        Expected: the producer, domain, version and doc string identify the module and the
            release. The properties carry the config name, tracker, run id and every metainfo
            value as JSON.
        Check: the model header and the metadata properties.
        """
        deploy_cfg = _deploy_cfg(
            {
                "head": {
                    "dynamo": False,
                    "metainfo": {
                        "class_names": ["car", "truck"],
                        "voxel_size": [0.1, 0.1, 0.1],
                        "has_twist": True,
                        "num_proposals": 500,
                    },
                }
            }
        )
        exporter = _exporter(
            tmp_path, deploy_cfg, release="v1.2.3", tracker="mlflow", run_id="run-123"
        )

        outputs = exporter.export()

        onnx_model = onnx.load(str(outputs.onnx_exported_paths[0]))
        assert onnx_model.producer_name == PRODUCER_NAME
        assert onnx_model.producer_version == get_git_sha()
        assert onnx_model.domain == MODEL_DOMAIN
        assert onnx_model.model_version == 10203
        assert onnx_model.doc_string == "head v1.2.3"
        props = _metadata(onnx_model)
        assert props["release"] == "v1.2.3"
        assert props["module"] == "head"
        assert props["config_name"] == CONFIG_NAME
        assert props["tracker"] == "mlflow"
        assert props["run_id"] == "run-123"
        assert props["class_names"] == '["car","truck"]'
        assert props["voxel_size"] == "[0.1,0.1,0.1]"
        assert props["has_twist"] == "true"
        assert props["num_proposals"] == "500"
        assert "export_date" in props
        assert props["exported_with"].startswith("pytorch")

    def test_unversioned_export_without_tracker_stamps_only_the_identity(
        self, tmp_path: Path
    ) -> None:
        """
        Input: no release, no tracker and no metainfo.
        Expected: model version 0, an ``unversioned`` release and no tracker properties.
        Check: the model header and the metadata properties.
        """
        exporter = _exporter(tmp_path, _deploy_cfg({"head": {"dynamo": False}}))

        outputs = exporter.export()

        onnx_model = onnx.load(str(outputs.onnx_exported_paths[0]))
        assert onnx_model.model_version == 0
        assert onnx_model.doc_string == "head unversioned"
        props = _metadata(onnx_model)
        assert props["release"] == "unversioned"
        assert props["module"] == "head"
        assert props["config_name"] == CONFIG_NAME
        assert "tracker" not in props
        assert "run_id" not in props

    def test_metainfo_colliding_with_a_stamped_key_is_rejected(self, tmp_path: Path) -> None:
        """
        Input: a ``metainfo`` block redefining ``release``.
        Expected: the stamp refuses to overwrite an automatic property.
        Check: a ValueError mentioning the reserved keys is raised.
        """
        deploy_cfg = _deploy_cfg({"head": {"dynamo": False, "metainfo": {"release": "v9.9.9"}}})
        exporter = _exporter(tmp_path, deploy_cfg)

        with pytest.raises(ValueError, match="reserved"):
            exporter.export()


class TestErrorPaths:
    """Misconfigurations fail before any file is shipped."""

    def test_unknown_module_name_is_rejected(self, tmp_path: Path) -> None:
        """
        Input: a model exporting ``head`` while the deploy config only knows ``encoder``.
        Expected: the module cannot be resolved against ``deploy.onnx.modules``.
        Check: a KeyError naming the module and the available ones, and no file written.
        """
        exporter = _exporter(tmp_path, _deploy_cfg({"encoder": {}}))

        with pytest.raises(KeyError, match="'head'.*encoder"):
            exporter.export()
        assert list((tmp_path / "exports").iterdir()) == []

    def test_missing_modules_block_is_rejected(self, tmp_path: Path) -> None:
        """
        Input: the default deploy config without any ``modules`` block.
        Expected: the module cannot be resolved.
        Check: a KeyError naming the module is raised.
        """
        deploy_cfg = OmegaConf.load(DEPLOY_YAML).deploy
        deploy_cfg.tensorrt.enabled = False
        exporter = _exporter(tmp_path, deploy_cfg)

        with pytest.raises(KeyError, match="head"):
            exporter.export()

    def test_spec_without_example_inputs_is_rejected(self, tmp_path: Path) -> None:
        """
        Input: an export spec carrying no example inputs and no input names.
        Expected: the ONNX export has nothing to trace with.
        Check: a ValueError mentioning the missing parameters is raised.
        """
        exporter = _exporter(tmp_path, _deploy_cfg({"head": {}}), model=_NoInputsModel())

        with pytest.raises(ValueError, match="no parameters"):
            exporter.export()

    def test_empty_predict_split_has_no_example_batch(self, tmp_path: Path) -> None:
        """
        Input: a datamodule whose predict dataloader yields nothing.
        Expected: no example batch can be read, so the export cannot start.
        Check: the iteration failure propagates.
        """
        exporter = _exporter(tmp_path, _deploy_cfg({"head": {}}), datamodule=_PredictDatamodule([]))

        with pytest.raises(StopIteration):
            exporter.export()

    def test_module_without_onnx_support_is_rejected_when_onnx_is_enabled(
        self, tmp_path: Path
    ) -> None:
        """
        Input: a spec supporting no stage while ``deploy.onnx.enabled`` is true.
        Expected: the export refuses rather than silently skipping the module.
        Check: a RuntimeError naming the module and the stage is raised.
        """
        model = _ExportModel(supported_stages=frozenset())
        exporter = _exporter(tmp_path, _deploy_cfg({"head": {}}), model=model)

        with pytest.raises(RuntimeError, match="'head' does not support ONNX export"):
            exporter.export()

    def test_module_without_tensorrt_support_is_rejected_when_tensorrt_is_enabled(
        self, tmp_path: Path
    ) -> None:
        """
        Input: an ONNX-only spec with ``deploy.tensorrt.enabled`` true.
        Expected: the ONNX file is written, then the TensorRT stage refuses the module before
            any engine build.
        Check: a RuntimeError naming the module and the stage, with the ONNX file on disk.
        """
        model = _ExportModel(supported_stages=frozenset({"onnx"}))
        deploy_cfg = _deploy_cfg({"head": {"dynamo": False}}, tensorrt_enabled=True)
        exporter = _exporter(tmp_path, deploy_cfg, model=model)

        with pytest.raises(RuntimeError, match="'head' does not support TensorRT export"):
            exporter.export()
        assert (tmp_path / "exports" / "head.onnx").exists()

    def test_tensorrt_stage_requires_the_onnx_file(self, tmp_path: Path) -> None:
        """
        Input: ``deploy.onnx.enabled`` false and ``deploy.tensorrt.enabled`` true.
        Expected: no ONNX file exists to build an engine from.
        Check: a FileNotFoundError naming the ONNX path is raised before any engine build.
        """
        deploy_cfg = _deploy_cfg({"head": {}}, onnx_enabled=False, tensorrt_enabled=True)
        exporter = _exporter(tmp_path, deploy_cfg)

        with pytest.raises(FileNotFoundError, match="head.onnx"):
            exporter.export()

    def test_module_precision_requirement_must_match_the_config(self, tmp_path: Path) -> None:
        """
        Input: a module declaring ``required_onnx_precision = "fp16"`` under the fp32 default.
        Expected: the precision mismatch is rejected before the export.
        Check: a ValueError naming the required precision is raised and no file is written.
        """
        model = _ExportModel()
        model.head.required_onnx_precision = "fp16"
        exporter = _exporter(tmp_path, _deploy_cfg({"head": {}}), model=model)

        with pytest.raises(ValueError, match="precision='fp16'"):
            exporter.export()
        assert list((tmp_path / "exports").iterdir()) == []
