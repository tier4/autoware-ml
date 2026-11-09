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

"""Tests for the Sparse4D detector."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.encoders.sparse.ptv3_middle_encoder import PTv3MiddleEncoder
from autoware_ml.models.detection3d.heads.sparse4d import TEMPORAL_STATE_INPUT_NAMES
from autoware_ml.models.detection3d.main_modules.sparse4d import Sparse4DDetectionModel
from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_lidar import Sparse4DLidar
from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid
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


def _build_lidar_network() -> Sparse4DLidar:
    """Build a small lidar branch whose neck returns the stage the deepest one fuses into."""
    return Sparse4DLidar(
        pts_voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        pts_middle_encoder=PTv3MiddleEncoder(
            encoder=build_ptv3_encoder(), grid_size=1.0, point_cloud_range=POINT_CLOUD_RANGE
        ),
        pts_backbone=None,
        pts_neck=PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=32),
    )


def _build_model(
    tmp_path: Path, num_temp_instances: int = 0, dn_query_budget: int = 0
) -> Sparse4DDetectionModel:
    """Build a small detector over the test lidar branch."""
    return Sparse4DDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        lidar_network=_build_lidar_network(),
        camera_network=None,
        bbox_head=build_sparse4d_head(
            make_anchor_file(tmp_path),
            num_temp_instances=num_temp_instances,
            dn_query_budget=dn_query_budget,
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
    )


def test_lidar_network_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lidar_network"):
        Sparse4DDetectionModel(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            lidar_network=None,
            camera_network=None,
            bbox_head=build_sparse4d_head(make_anchor_file(tmp_path)),
            log_dict_configs=LogDictConfigs(prog_bar=False),
        )


def test_camera_network_is_not_supported_yet(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="camera_network"):
        Sparse4DDetectionModel(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            lidar_network=_build_lidar_network(),
            camera_network=nn.Identity(),
            bbox_head=build_sparse4d_head(make_anchor_file(tmp_path)),
            log_dict_configs=LogDictConfigs(prog_bar=False),
        )


def test_lidar_branch_and_head_must_agree_on_the_sampled_levels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="levels"):
        Sparse4DDetectionModel(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            lidar_network=_build_lidar_network(),
            camera_network=None,
            bbox_head=build_sparse4d_head(make_anchor_file(tmp_path), feature_strides=(1, 2)),
            log_dict_configs=LogDictConfigs(prog_bar=False),
        )


def test_optimizer_groups_cover_the_encoder_and_the_detection_branch(tmp_path: Path) -> None:
    model = _build_model(tmp_path)

    groups = model.build_optimizer_groups()

    assert set(groups) == {"encoder_default", "encoder_block", "det3d_branch"}
    grouped = sum(len(parameters) for parameters in groups.values())
    assert grouped == sum(1 for parameter in model.parameters() if parameter.requires_grad)
    head_count = sum(1 for p in model.bbox_head.parameters() if p.requires_grad)
    neck_count = sum(1 for p in model.lidar_network.pts_neck.parameters() if p.requires_grad)
    assert len(groups["det3d_branch"]) == head_count + neck_count


def test_points_must_follow_the_layout_of_the_voxel_encoder(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    batch = frame_inputs(torch.device("cpu"), 100.0)
    gt_batch = batch.multi_task_gt_batch
    point_batch = gt_batch.point_cloud_gt_batch._replace(timestamp_difference_dim=-1)

    with pytest.raises(ValueError, match="time lag"):
        model(
            batch.replace(multi_task_gt_batch=gt_batch._replace(point_cloud_gt_batch=point_batch))
        )


def test_dropped_voxels_are_refused(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    batch = frame_inputs(torch.device("cpu"), 100.0)
    voxels_data = batch.voxels_data._replace(num_dropped_voxels=torch.tensor(3))

    with pytest.raises(RuntimeError, match="dropped 3 occupied voxels"):
        model(batch.replace(voxels_data=voxels_data))


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
