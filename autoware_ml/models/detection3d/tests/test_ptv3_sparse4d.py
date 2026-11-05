"""Tests for the PTv3 Sparse4D detection model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.models.detection3d.heads.sparse4d import TEMPORAL_STATE_INPUT_NAMES
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid
from autoware_ml.models.detection3d.ptv3_sparse4d import PTv3Detection3DSparse4DModel
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_ptv3_encoder,
)
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.multi.tests.test_ptv3_det3d_seg3d_sparse4d import frame_inputs
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor

requires_cuda_spconv = pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)


def _build_model(
    tmp_path: Path, num_temp_instances: int = 0, dn_query_budget: int = 0
) -> PTv3Detection3DSparse4DModel:
    """Build a small detector whose head samples the stage the deepest test stage fuses into."""
    return PTv3Detection3DSparse4DModel(
        encoder=build_ptv3_encoder(),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        det_fusion=PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=32),
        bbox_head=build_sparse4d_head(
            make_anchor_file(tmp_path),
            num_temp_instances=num_temp_instances,
            dn_query_budget=dn_query_budget,
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        grid_size=1.0,
        point_cloud_range=POINT_CLOUD_RANGE,
    )


def test_pyramid_and_head_must_agree_on_the_sampled_levels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="levels"):
        PTv3Detection3DSparse4DModel(
            encoder=build_ptv3_encoder(),
            voxel_encoder=SweepSplitVoxelFeatureEncoder(),
            det_fusion=PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=32),
            bbox_head=build_sparse4d_head(make_anchor_file(tmp_path), feature_strides=(1, 2)),
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            grid_size=1.0,
            point_cloud_range=POINT_CLOUD_RANGE,
        )


def test_optimizer_groups_cover_the_encoder_and_the_detection_branch(tmp_path: Path) -> None:
    model = _build_model(tmp_path)

    groups = model.build_optimizer_groups()

    assert set(groups) == {"encoder_default", "encoder_block", "det3d_branch"}
    grouped = sum(len(parameters) for parameters in groups.values())
    assert grouped == sum(1 for parameter in model.parameters() if parameter.requires_grad)


@requires_cuda_spconv
def test_temporal_detector_trains_on_a_stream_and_reaches_every_parameter(
    tmp_path: Path,
) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path, num_temp_instances=4, dn_query_budget=8).to(device)

    for frame_index in range(2):
        model.zero_grad()
        batch = frame_inputs(device, 100.0 + 0.1 * frame_index)
        outputs = model(batch)
        metrics = model.compute_metrics(batch, outputs)
        metrics["loss"].backward()

        assert torch.isfinite(metrics["loss"])
        assert outputs.segmentation3d_head_outputs is None
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert missing == []

    # The second frame of the scene reads the instances the first one cached
    assert model.bbox_head.instance_bank._slot_valid.tolist() == [True]


@requires_cuda_spconv
def test_detector_eval_output_and_predictions_carry_boxes_only(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path, num_temp_instances=4).to(device).eval()

    eval_outputs = []
    with torch.no_grad():
        for frame_index in range(2):
            batch = frame_inputs(device, 100.0 + 0.1 * frame_index)
            outputs = model(batch)
            eval_outputs.append(model.build_eval_output(batch, outputs))
        predictions = model.decode_outputs(batch, outputs)

    first, second = eval_outputs
    assert first["scene_token"] == ["db/scene/0"]
    assert first["gt_observed"][0].all()
    assert first["stream_ids"][0] == second["stream_ids"][0]
    assert "pred_labels" not in first
    assert predictions.detection3d_predictions[0].bboxes_3d.shape[1] == 9
    assert predictions.segmentation3d_predictions is None


@requires_cuda_spconv
def test_forward_needs_frame_metadata(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    with pytest.raises(ValueError, match="frame metadata"):
        model(build_inputs(device=torch.device("cuda")))


@requires_cuda_spconv
def test_split_export_holds_the_encoder_and_the_det_head_only(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = _build_model(tmp_path, num_temp_instances=4).to(device).eval()

    specs = model.build_export_specs(frame_inputs(device, 100.0))
    spec = specs["ptv3_det3d_head"]

    assert list(specs) == ["ptv3_lidar_encoder", "ptv3_det3d_head"]
    # The fused stage, the deepest stage, their cluster and the fused grid, then the bank state
    assert spec.input_names[4:] == list(TEMPORAL_STATE_INPUT_NAMES)
    assert len(spec.args) == len(spec.input_names)
    assert spec.output_names[:3] == ["classification", "prediction", "quality"]
    assert spec.output_names[-1] == "temp_query_index_out"


def test_export_needs_the_encoder_geometry(tmp_path: Path) -> None:
    model = PTv3Detection3DSparse4DModel(
        encoder=build_ptv3_encoder(),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        det_fusion=PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=32),
        bbox_head=build_sparse4d_head(make_anchor_file(tmp_path)),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
    )
    with pytest.raises(ValueError, match="grid_size and point_cloud_range"):
        model.build_export_specs(build_inputs(device=torch.device("cpu")))
