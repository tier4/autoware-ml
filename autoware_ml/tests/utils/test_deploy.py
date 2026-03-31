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
import torch

from autoware_ml.utils.deploy import export_to_onnx, get_export_parameter_names


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
    deploy_cfg = OmegaConf.create(
        {
            "onnx": {
                "opset_version": 17,
                "dynamo": False,
                "do_constant_folding": True,
                "input_names": ["input"],
                "output_names": ["configured_output"],
            }
        }
    )

    export_to_onnx(
        model=_SingleOutput(),
        input_sample=(torch.ones(2, 3),),
        deploy_cfg=deploy_cfg,
        input_param_names=["input"],
        output_names_override=["exported_output"],
        output_path=output_path,
    )

    assert output_path.exists()
