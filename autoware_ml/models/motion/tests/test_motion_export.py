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

"""Tests for the export graph of the agent motion head."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import onnxruntime
import torch
from omegaconf import OmegaConf

from autoware_ml.models.detection3d.ptv3 import PTv3Det3DFeaturePyramid
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_ptv3_encoder,
    build_seg3d_head,
)
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_recipe,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.motion.instance_queue import QueueState
from autoware_ml.models.motion.tests.motion_fixtures import build_motion_head
from autoware_ml.models.multi.ptv3_sparsedrive_motion import (
    MOTION_EXPORT_INPUT_NAMES,
    MOTION_EXPORT_OUTPUT_NAMES,
    PTv3SparseDriveMotionModel,
)
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import export_to_onnx

DIMS, CLASSES, TEMPORAL = 32, 2, 4
requires_cuda_spconv = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(),
    "PTv3 sparse-convolution tests require CUDA spconv",
)


def build_model(tmp_path: Path) -> PTv3SparseDriveMotionModel:
    """A small model with the agent motion head."""
    return PTv3SparseDriveMotionModel(
        motion_head=build_motion_head(CLASSES),
        encoder=build_ptv3_encoder(),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        seg3d_head=build_seg3d_head(),
        det_fusion=PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=DIMS),
        bbox_head=build_sparse4d_head(make_anchor_recipe(), num_temp_instances=TEMPORAL),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        grid_size=1.0,
        point_cloud_range=POINT_CLOUD_RANGE,
    )


class TestMotionExport(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    @requires_cuda_spconv
    def test_the_motion_graph_predicts_a_stream_like_the_head(self) -> None:
        """
        Input: the motion export module run over two frames of one stream, the history of the first
          frame feeding the second.
        Expected: the graph takes no encoder stage, the detection graph also returns the instance
          features, every output of both frames matches the head stepping on the same detections.
        """
        device = torch.device("cuda")
        model = build_model(self.tmp_path).to(device).eval()
        specs = model.build_export_specs(build_inputs(device=device))
        spec = specs["ptv3_motion_head"]
        self.assertEqual(spec.input_names, list(MOTION_EXPORT_INPUT_NAMES))
        self.assertEqual(spec.output_names, list(MOTION_EXPORT_OUTPUT_NAMES))
        self.assertEqual(specs["ptv3_det3d_head"].output_names[-1], "instance_feature")

        head = model.motion_head.prepare_for_export()
        state = list(spec.args[3:8])
        queries = spec.args[0].shape[1]
        for frame in range(2):
            generator = torch.Generator().manual_seed(frame)
            prediction = spec.args[1].cpu() + torch.randn(spec.args[1].shape, generator=generator)
            prediction[..., 6:8] = torch.nn.functional.normalize(prediction[..., 6:8], dim=-1)
            detections = [
                torch.randn(1, queries, DIMS, generator=generator).to(device),
                prediction.to(device),
                torch.randn(1, queries, CLASSES, generator=generator).to(device),
            ]
            stream = [
                torch.randint(0, queries, (1, TEMPORAL), generator=generator).to(device),
                torch.full((1,), float(frame > 0), device=device),
                torch.eye(4, device=device)[None],
            ]
            with torch.no_grad():
                outputs = spec.module(*detections, *state, *stream)
                reference, next_state = head.step(
                    *detections,
                    spec.module.bbox_head.anchor_encoder,
                    QueueState(*state),
                    stream[1] > 0.5,
                    stream[0],
                    stream[2],
                )
            named = dict(zip(spec.output_names, outputs, strict=True))
            torch.testing.assert_close(named["agent_trajectories"], reference.agent_trajectories)
            torch.testing.assert_close(named["agent_mode_logits"], reference.agent_mode_logits)
            for name, expected in zip(MOTION_EXPORT_OUTPUT_NAMES[2:], next_state, strict=True):
                torch.testing.assert_close(named[name], expected)
            state = list(outputs[2:])

    @requires_cuda_spconv
    def test_the_onnx_motion_graph_predicts_like_the_export_module(self) -> None:
        """
        Input: the motion graph exported with the deploy settings and run in ONNX Runtime over
          two frames, the history of the first frame feeding the second.
        Expected: the export succeeds and every output of both frames matches the export module.
        """
        device = torch.device("cuda")
        model = build_model(self.tmp_path).to(device).eval()
        spec = model.build_export_specs(build_inputs(device=device))["ptv3_motion_head"]
        onnx_cfg = OmegaConf.create(
            {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
        )
        path = self.tmp_path / "ptv3_motion_head.onnx"
        with torch.no_grad():
            export_to_onnx(spec, onnx_cfg, path)
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        state = list(spec.args[3:8])
        queries = spec.args[0].shape[1]
        for frame in range(2):
            generator = torch.Generator().manual_seed(frame)
            prediction = spec.args[1].cpu() + torch.randn(spec.args[1].shape, generator=generator)
            prediction[..., 6:8] = torch.nn.functional.normalize(prediction[..., 6:8], dim=-1)
            inputs = [
                torch.randn(1, queries, DIMS, generator=generator).to(device),
                prediction.to(device),
                torch.randn(1, queries, CLASSES, generator=generator).to(device),
                *state,
                torch.randint(0, queries, (1, TEMPORAL), generator=generator).to(device),
                torch.full((1,), float(frame > 0), device=device),
                torch.eye(4, device=device)[None],
            ]
            with torch.no_grad():
                expected = spec.module(*inputs)
            outputs = session.run(
                None,
                {
                    name: value.cpu().numpy()
                    for name, value in zip(spec.input_names, inputs, strict=True)
                },
            )
            for name, want, got in zip(spec.output_names, expected, outputs, strict=True):
                torch.testing.assert_close(
                    torch.from_numpy(got), want.cpu(), rtol=1e-4, atol=1e-4, msg=name
                )
            state = list(expected[2:])
