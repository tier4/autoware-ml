"""Tests for running the exported PTv3 graphs on named tensors and rebuilding model outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from autoware_ml.dataclasses.geometry.voxels import VoxelsData
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    CenterHeadOutputs,
    TransFusionHeadOutputs,
)
from autoware_ml.models.detection3d.tests.head_output_fixtures import build_transfusion_outputs
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import build_inputs
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.multi.ptv3_det3d_seg3d import PTv3Det3DSeg3DTensorRTModel
from autoware_ml.models.multi.tests.test_ptv3_det3d_seg3d_export import (
    JOINT_EXPORT_OUTPUT_NAMES,
    _build_det3d_seg3d_model,
)
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.models.segmentation3d.ptv3_base import (
    prepare_ptv3_export_inputs,
    run_export_graphs,
    split_voxels_by_sample,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.deploy import ExportSpec


class TorchGraph:
    """An export spec's module called the way a deployed engine is, on named tensors."""

    def __init__(self, spec: ExportSpec) -> None:
        self.spec = spec
        self.input_names = tuple(spec.input_names)

    def __call__(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.spec.module(*(tensors[name] for name in self.input_names))
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        return dict(zip(self.spec.output_names, outputs, strict=True))


def test_center_head_outputs_round_trip_through_export_tensors() -> None:
    names = ["heatmap", "reg", "height", "dim", "rot"]
    outputs = CenterHeadOutputs(
        heatmap=torch.rand(2, 3, 4, 4),
        reg=torch.rand(2, 2, 4, 4),
        height=torch.rand(2, 1, 4, 4),
        dim=torch.rand(2, 3, 4, 4),
        rot=torch.rand(2, 2, 4, 4),
        vel=None,
    )
    rebuilt = CenterHeadOutputs.from_export_tensors(
        dict(zip(names, outputs.export_tensors(names)))
    )
    assert rebuilt.vel is None
    for name in names:
        assert torch.equal(getattr(rebuilt, name), getattr(outputs, name))


def test_transfusion_head_outputs_round_trip_through_export_tensors() -> None:
    names = ["dense_heatmap", "query_heatmap_score", "query_labels", "heatmap", "center"]
    names += ["height", "dim", "rot"]
    outputs = build_transfusion_outputs(
        {
            name: torch.rand(1, channels, 6)
            for name, channels in (("heatmap", 2), ("center", 2), ("height", 1), ("dim", 3))
        }
        | {"rot": torch.rand(1, 2, 6)}
    )
    tensors = dict(zip(names, outputs.export_tensors(names)))
    tensors["query_labels"] = tensors["query_labels"].int()  # TensorRT may emit int32
    rebuilt = TransFusionHeadOutputs.from_export_tensors(tensors)
    assert rebuilt.export_tensors(names)[2].dtype == torch.long
    for rebuilt_tensor, tensor in zip(rebuilt.export_tensors(names), outputs.export_tensors(names)):
        assert torch.equal(rebuilt_tensor, tensor)


def _voxels(batch_indices: list[int]) -> VoxelsData:
    count = len(batch_indices)
    return VoxelsData(
        voxels=torch.arange(count, dtype=torch.float32).view(count, 1, 1),
        coords=torch.arange(3 * count, dtype=torch.int32).view(count, 3),
        num_points=torch.ones(count, dtype=torch.int32),
        batch_indices=torch.tensor(batch_indices, dtype=torch.int32),
        point_voxel_indices=torch.zeros(0, dtype=torch.long),
        num_dropped_voxels=torch.tensor(0),
    )


def test_split_voxels_by_sample_returns_the_voxels_of_every_sample() -> None:
    samples = split_voxels_by_sample(_voxels([0, 0, 1]), batch_size=2)
    assert [voxels.flatten().tolist() for voxels, _, _ in samples] == [[0.0, 1.0], [2.0]]
    assert samples[1][2].tolist() == [[6, 7, 8]]


@pytest.mark.parametrize(
    ("batch_indices", "batch_size", "message"),
    [([1, 0], 2, "not grouped"), ([0, 0], 2, "Sample 1")],
)
def test_split_voxels_by_sample_refuses_ungrouped_or_empty_samples(
    batch_indices: list[int], batch_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        split_voxels_by_sample(_voxels(batch_indices), batch_size)


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_export_graphs_on_named_tensors_rebuild_the_model_outputs() -> None:
    model = _build_det3d_seg3d_model().cuda().eval()
    batch = build_inputs(device=torch.device("cuda"), is_training=False)
    specs = model.build_export_specs(batch)
    graphs = [TorchGraph(spec) for spec in specs.values()]

    tensors = run_export_graphs(graphs, [prepare_ptv3_export_inputs(model, batch)])

    # The encoder features feed the heads, so only the head outputs are model outputs.
    head_outputs = [*specs["ptv3_seg3d_head"].output_names, *specs["ptv3_det3d_head"].output_names]
    assert sorted(tensors) == sorted(head_outputs)
    outputs = model.outputs_from_export_tensors(tensors)
    assert torch.equal(outputs.segmentation3d().logits.argmax(dim=1), tensors["pred_labels"])
    predictions = model.decode_outputs(batch, outputs)
    assert len(predictions.detection3d_predictions) == 1


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_tensorrt_model_runs_the_engines_on_every_sample(tmp_path: Path) -> None:
    model = _build_det3d_seg3d_model().cuda().eval()
    batch = build_inputs(device=torch.device("cuda"), is_training=False)
    specs = model.build_export_specs(batch)
    for name in PTv3Det3DSeg3DTensorRTModel.ENGINE_NAMES:
        (tmp_path / f"{name}.engine").touch()
    deployed = PTv3Det3DSeg3DTensorRTModel(
        engine_dir=tmp_path,
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        encoder=model.encoder,
        voxel_encoder=model.voxel_encoder,
        seg3d_head=model.seg3d_head,
        bev_neck=model.bev_neck,
        bbox_head=model.bbox_head,
        export_output_names=JOINT_EXPORT_OUTPUT_NAMES,
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
    ).eval()
    deployed.engines = {name: TorchGraph(spec) for name, spec in specs.items()}

    expected = run_export_graphs(
        list(deployed.engines.values()), [prepare_ptv3_export_inputs(model, batch)]
    )
    with torch.no_grad():
        outputs = deployed(batch)
    assert torch.equal(outputs.segmentation3d().logits.argmax(dim=1), expected["pred_labels"])
    torch.testing.assert_close(
        outputs.detection3d().transfusion_head().dense_heatmap, expected["dense_heatmap"]
    )
