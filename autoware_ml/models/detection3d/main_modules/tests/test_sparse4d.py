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

"""Tests for the Sparse4D detector on the BEVFusion lidar network."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d import (
    TEMPORAL_STATE_INPUT_NAMES,
    Sparse4DHead,
)
from autoware_ml.models.detection3d.main_modules.sparse4d import Sparse4DDetectionModel
from autoware_ml.models.detection3d.main_modules.sparse4ds.tests.lidar_fixtures import (
    NECK_CHANNELS,
    build_lidar_network,
)
from autoware_ml.models.detection3d.necks.bev_map_pyramid import BEVMapPyramid
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
)
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.multi.tests.test_ptv3_det3d_seg3d_sparse4d import frame_inputs
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor

BEV_CHANNELS = 32


def _build_head(
    tmp_path: Path, num_temp_instances: int = 0, dn_query_budget: int = 0
) -> Sparse4DHead:
    """Build a small head sampling the neck map and its halving at the keypoints."""
    return build_sparse4d_head(
        make_anchor_file(tmp_path),
        num_temp_instances=num_temp_instances,
        dn_query_budget=dn_query_budget,
        lidar_sampling="bev",
        bev_channels=BEV_CHANNELS,
        bev=BEVMapPyramid(
            point_cloud_range=POINT_CLOUD_RANGE,
            voxel_size=[1.0, 1.0, 4.0],
            out_size_factor=1,
            in_channels=NECK_CHANNELS,
            channels=BEV_CHANNELS,
            num_downsampled=1,
        ),
    )


def _build_model(
    tmp_path: Path, num_temp_instances: int = 0, dn_query_budget: int = 0
) -> Sparse4DDetectionModel:
    """Build a small detector over the stub lidar branch."""
    torch.manual_seed(0)
    return Sparse4DDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        lidar_network=build_lidar_network(),
        camera_network=None,
        bbox_head=_build_head(tmp_path, num_temp_instances, dn_query_budget),
        log_dict_configs=LogDictConfigs(prog_bar=False),
    )


def test_lidar_network_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lidar_network"):
        Sparse4DDetectionModel(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            lidar_network=None,
            camera_network=None,
            bbox_head=_build_head(tmp_path),
            log_dict_configs=LogDictConfigs(prog_bar=False),
        )


def test_camera_network_is_not_supported_yet(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="camera_network"):
        Sparse4DDetectionModel(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            lidar_network=build_lidar_network(),
            camera_network=nn.Identity(),
            bbox_head=_build_head(tmp_path),
            log_dict_configs=LogDictConfigs(prog_bar=False),
        )


def test_temporal_detector_trains_on_a_stream_and_reaches_every_parameter(
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")
    model = _build_model(tmp_path, num_temp_instances=4, dn_query_budget=8)

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


def test_detector_eval_output_and_predictions_carry_boxes_only(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = _build_model(tmp_path, num_temp_instances=4).eval()

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


def test_forward_needs_frame_metadata(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    with pytest.raises(ValueError, match="frame metadata"):
        model(build_inputs())


def test_forward_needs_voxels(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    batch = frame_inputs(torch.device("cpu"), 100.0)
    with pytest.raises(ValueError, match="voxels"):
        model(batch.replace(voxels_data=None))


def test_export_spec_takes_the_voxels_and_the_bank_state_of_the_last_frame(
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")
    model = _build_model(tmp_path, num_temp_instances=4).eval()
    batch = frame_inputs(device, 100.0)

    specs = model.build_export_specs(batch)
    spec = specs["sparse4d_lidar"]
    with torch.no_grad():
        first = dict(zip(spec.output_names, spec.module(*spec.args), strict=True))

    assert list(specs) == ["sparse4d_lidar"]
    assert spec.input_names[:3] == ["voxels", "coors", "num_points_per_voxel"]
    assert spec.input_names[3:] == list(TEMPORAL_STATE_INPUT_NAMES)
    assert len(spec.args) == len(spec.input_names)
    # The runtime layout: single sample voxels with (z, y, x) coordinates
    assert spec.args[1].shape[1] == 3
    assert spec.output_names[:3] == ["classification", "prediction", "quality"]
    assert spec.output_names[-1] == "temp_query_index_out"
    assert first["temp_anchor_out"].shape == (1, 4, 10)
    assert first["temp_query_index_out"].shape == (1, 4)
    # The export copy is prepared apart from the trained model
    assert spec.module.lidar_network.pts_middle_encoder.exportable
    assert not model.lidar_network.pts_middle_encoder.exportable


def test_exported_graph_reads_the_state_of_the_last_frame(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = _build_model(tmp_path, num_temp_instances=4).eval()
    spec = model.build_export_specs(frame_inputs(device, 100.0))["sparse4d_lidar"]
    voxel_args = spec.args[:3]

    with torch.no_grad():
        first = dict(zip(spec.output_names, spec.module(*spec.args), strict=True))
        second = dict(
            zip(
                spec.output_names,
                spec.module(
                    voxel_args[0],
                    voxel_args[1],
                    voxel_args[2],
                    first["temp_instance_feature_out"],
                    first["temp_anchor_out"],
                    first["temp_confidence_out"],
                    torch.ones(1, device=device),
                    torch.full((1,), 0.1, device=device),
                    torch.eye(4, device=device)[None],
                ),
                strict=True,
            )
        )

    assert second["classification"].shape == first["classification"].shape
    assert not torch.equal(second["classification"], first["classification"])
