"""Tests for the PTv3 joint segmentation and Sparse4D detection model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid, rebuild_point_chain
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_ptv3_encoder,
    build_seg3d_head,
)
from autoware_ml.models.detection3d.heads.sparse4d import ANCHOR_DIM, TEMPORAL_STATE_INPUT_NAMES
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.multi.ptv3_det3d_seg3d_sparse4d import PTv3Det3DSeg3DSparse4DModel
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor

requires_cuda_spconv = pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)


def _build_model(tmp_path: Path, **head_overrides: object) -> PTv3Det3DSeg3DSparse4DModel:
    """Build a small model whose head samples the stage the deepest test stage fuses into."""
    return PTv3Det3DSeg3DSparse4DModel(
        encoder=build_ptv3_encoder(),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        seg3d_head=build_seg3d_head(),
        det_fusion=PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=32),
        bbox_head=build_sparse4d_head(make_anchor_file(tmp_path), **head_overrides),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        grid_size=1.0,
        point_cloud_range=POINT_CLOUD_RANGE,
    )


def frame_inputs(device: torch.device, timestamp: float) -> ModelBatchInputs:
    """One frame of the test scene with its frame metadata."""
    batch = build_inputs(device=device)
    detections = batch.multi_task_gt_batch.detection3d_gt_batch
    gt_batch = batch.multi_task_gt_batch._replace(
        detection3d_gt_batch=detections._replace(
            gt_bboxes_num_points=torch.full_like(detections.gt_bboxes_num_points, 5)
        ),
        frame_meta_batch=FrameMetaBatch(
            ego2globals=torch.eye(4, device=device)[None],
            scene_tokens=["db/scene/0"],
            timestamps=torch.tensor([timestamp], dtype=torch.float64, device=device),
            lanelet_map_paths=[None],
        ),
    )
    return batch.replace(multi_task_gt_batch=gt_batch)


@requires_cuda_spconv
def test_temporal_model_trains_on_a_stream_and_reaches_every_parameter(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path, num_temp_instances=4, dn_query_budget=8).to(device)

    for frame_index in range(2):
        model.zero_grad()
        batch = frame_inputs(device, 100.0 + 0.1 * frame_index)
        outputs = model(batch)
        metrics = model.compute_metrics(batch, outputs)
        metrics["loss"].backward()

        assert torch.isfinite(metrics["loss"])
        assert outputs.detection3d().sparse4d_head().denoising is not None
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert missing == []

    # The second frame of the scene reads the instances the first one cached
    assert model.bbox_head.instance_bank._slot_valid.tolist() == [True]


@requires_cuda_spconv
def test_model_eval_output_and_predictions(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path).to(device).eval()
    batch = frame_inputs(device, 100.0)

    with torch.no_grad():
        outputs = model(batch)
        eval_out = model.build_eval_output(batch, outputs)
        predictions = model.decode_outputs(batch, outputs)

    assert eval_out["scene_token"] == ["db/scene/0"]
    assert predictions.detection3d_predictions[0].bboxes_3d.shape[1] == 9
    assert predictions.segmentation3d_predictions is not None


@requires_cuda_spconv
def test_temporal_model_eval_output_carries_the_tracks_of_one_stream(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path, num_temp_instances=4).to(device).eval()

    eval_outputs = []
    with torch.no_grad():
        for frame_index in range(2):
            batch = frame_inputs(device, 100.0 + 0.1 * frame_index)
            outputs = model(batch)
            eval_outputs.append(model.build_eval_output(batch, outputs))

    first, second = eval_outputs
    assert first["gt_observed"][0].all()
    assert first["stream_ids"][0] == second["stream_ids"][0]
    assert second["predictions"][0]["instance_ids"] is not None
    assert len(second["gt_instance_ids"][0]) == len(second["gt_boxes"][0])


@requires_cuda_spconv
def test_forward_needs_frame_metadata(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    with pytest.raises(ValueError, match="frame metadata"):
        model(build_inputs(device=torch.device("cuda")))


@requires_cuda_spconv
def test_split_export_decodes_like_the_model_and_carries_the_bank_state(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path, num_temp_instances=4).to(device).eval()

    specs = model.build_export_specs(frame_inputs(device, 100.0))
    spec = specs["ptv3_det3d_head"]
    stage_args, reset_state = spec.args[:4], spec.args[4:]
    with torch.no_grad():
        first = dict(zip(spec.output_names, spec.module(*spec.args)))
        # The det branch of the model on the same encoder stages
        skip_feat, deepest_feat, cluster, skip_grid_coord = stage_args
        stages = model.det_fusion(
            rebuild_point_chain([skip_feat, deepest_feat], [cluster], [skip_grid_coord])
        )
        reference = model.bbox_head.forward_temporal_export(stages, *reset_state)

    assert list(specs) == ["ptv3_lidar_encoder", "ptv3_seg3d_head", "ptv3_det3d_head"]
    assert spec.input_names[4:] == list(TEMPORAL_STATE_INPUT_NAMES)
    for name in spec.output_names:
        assert torch.allclose(first[name], reference[name], atol=1e-5), name
    assert first["temp_anchor_out"].shape == (1, 4, ANCHOR_DIM)
    assert spec.output_names[-1] == "temp_query_index_out"
    assert first["temp_query_index_out"].shape == (1, 4)

    # The next frame reads the state the first one returned
    next_state = (
        first["temp_instance_feature_out"],
        first["temp_anchor_out"],
        first["temp_confidence_out"],
        torch.ones(1, device=device),
        torch.full((1,), 0.1, device=device),
        torch.eye(4, device=device)[None],
    )
    with torch.no_grad():
        second = dict(zip(spec.output_names, spec.module(*stage_args, *next_state)))
    assert second["classification"].shape == first["classification"].shape
    assert not torch.equal(second["classification"], first["classification"])


@requires_cuda_spconv
@pytest.mark.parametrize(("keep_deepest", "feature_strides"), [(False, (1,)), (True, (1, 2))])
def test_det_head_exports_to_onnx_with_the_deploy_dynamic_axes(
    tmp_path: Path, keep_deepest: bool, feature_strides: tuple[int, ...]
) -> None:
    """
    Input: the split det head of a model sampling the fused stage, and the deepest stage too.
    Expected: the deploy export traces the sparse lookup with dynamic voxel counts, which needs a
    constant empty row, reads the deepest voxel coordinates only when it samples that stage and
    samples every decoder layer with one aggregation node.
    Check: the ONNX file passes the checker, declares the spec inputs and holds one aggregation
    node per layer with no sampling chain around it.
    """
    import onnx
    from omegaconf import OmegaConf

    from autoware_ml.utils.deploy import export_to_onnx

    device = torch.device("cuda")
    model = PTv3Det3DSeg3DSparse4DModel(
        encoder=build_ptv3_encoder(),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        seg3d_head=build_seg3d_head(),
        det_fusion=PTv3Det3DFeaturePyramid(
            stage_channels=(8, 16), out_channels=32, keep_deepest=keep_deepest
        ),
        bbox_head=build_sparse4d_head(
            make_anchor_file(tmp_path), num_temp_instances=4, feature_strides=feature_strides
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        grid_size=1.0,
        point_cloud_range=POINT_CLOUD_RANGE,
    )
    model = model.to(device).eval()
    spec = model.build_export_specs(frame_inputs(device, 100.0))["ptv3_det3d_head"]
    path = tmp_path / "ptv3_det3d_head.onnx"

    export_to_onnx(
        spec,
        OmegaConf.create({"opset_version": 17, "dynamo": False, "do_constant_folding": False}),
        path,
    )

    graph = onnx.load(str(path)).graph
    onnx.checker.check_model(onnx.load(str(path)))
    declared = {value.name for value in graph.input}
    assert declared <= set(spec.input_names)
    assert ("point_grid_coord_1" in spec.input_names) is keep_deepest
    aggregations = [node for node in graph.node if node.op_type == "SparseDeformableAggregation"]
    assert len(aggregations) == len(model.bbox_head.deformable)
    assert {(node.domain, len(node.input)) for node in aggregations} == {
        ("autoware", 2 + 2 * len(feature_strides))
    }
    assert not any(node.op_type == "Floor" for node in graph.node)
