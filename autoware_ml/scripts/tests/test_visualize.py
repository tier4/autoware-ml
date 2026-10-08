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

"""Tests for the visualization preview entrypoint."""

from __future__ import annotations

from importlib import import_module

import pytest
from omegaconf import OmegaConf

from autoware_ml.cli import cli
from autoware_ml.scripts import visualize
from autoware_ml.utils import checkpoints

_ENTRYPOINT_CONSTANTS = (
    "TRAIN_ENTRYPOINT_MODULE",
    "DEPLOY_ENTRYPOINT_MODULE",
    "TEST_ENTRYPOINT_MODULE",
    "VISUALIZE_ENTRYPOINT_MODULE",
)


@pytest.mark.parametrize("constant_name", _ENTRYPOINT_CONSTANTS)
def test_cli_entrypoint_modules_are_importable(constant_name: str) -> None:
    """Every module the CLI dispatches to must import.

    ``autoware-ml visualize`` failed at import because the entrypoint requested
    a checkpoint helper that does not exist, and nothing exercised the module.
    """
    import_module(getattr(cli, constant_name))


def test_visualize_uses_the_shared_checkpoint_loader() -> None:
    """The entrypoint must bind the real loader, not a name that does not exist."""
    assert visualize.apply_matching_weights is checkpoints.apply_matching_weights


def test_build_preview_config_defaults_to_an_automatic_rerun_preview() -> None:
    config = visualize._build_preview_config(OmegaConf.create({}))

    assert config.mode == "auto"
    assert config.split == "test"
    assert config.sample_index == 0
    assert config.max_samples == 1
    assert config.prediction_frequency_hz == 10.0
    assert config.device == "auto"
    assert config.point_labels is False
    assert config.session.backend == "rerun"
    assert config.session.web_port == 9090
    assert config.session.grpc_port == 9876
    assert config.session.wait is True
    assert config.session.timeline == "frame"
    assert config.session.camera_frustums_visible is False


def test_build_preview_config_maps_every_visualization_override() -> None:
    config = visualize._build_preview_config(
        OmegaConf.create(
            {
                "visualization": {
                    "mode": "data",
                    "split": "val",
                    "sample_index": 4,
                    "max_samples": 8,
                    "prediction_frequency_hz": 5.0,
                    "device": "cpu",
                    "point_labels": True,
                    "backend": "noop",
                    "application_id": "preview",
                    "recording_id": "run-1",
                    "web_port": 9091,
                    "grpc_port": 9877,
                    "wait": False,
                    "server_memory_limit": "10%",
                    "timeline": "sample",
                    "camera_frustums_visible": True,
                    "sync_views": True,
                    "sync_port": 9095,
                    "segmentation3d_class_names": ["road", "car"],
                    "detection3d_class_names": ["car"],
                }
            }
        )
    )

    assert config.mode == "data"
    assert config.split == "val"
    assert config.sample_index == 4
    assert config.max_samples == 8
    assert config.prediction_frequency_hz == 5.0
    assert config.device == "cpu"
    assert config.point_labels is True
    assert config.segmentation_class_names == ("road", "car")
    assert config.detection_class_names == ("car",)
    assert config.session.backend == "noop"
    assert config.session.application_id == "preview"
    assert config.session.recording_id == "run-1"
    assert config.session.web_port == 9091
    assert config.session.grpc_port == 9877
    assert config.session.wait is False
    assert config.session.server_memory_limit == "10%"
    assert config.session.timeline == "sample"
    assert config.session.point_color_mode == "semantic"
    assert config.session.camera_frustums_visible is True
    assert config.session.sync_views is True
    assert config.session.sync_port == 9095
