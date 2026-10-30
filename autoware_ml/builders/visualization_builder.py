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

"""Builder of the visualization preview settings from the Hydra configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from autoware_ml.builders.model_builder import resolve_weights_paths
from autoware_ml.visualization.preview import VisualizationPreviewConfig
from autoware_ml.visualization.viewers.contracts import VisualizationSessionConfig


def build_preview_config(cfg: DictConfig) -> VisualizationPreviewConfig:
    """
    Build the preview settings from the ``visualization`` group of the configuration.

    Every key is optional, the CLI fills the ones it exposes. An unset class name list keeps
    the class names of the database taxonomy.

    Args:
        cfg: Hydra configuration.

    Returns:
        VisualizationPreviewConfig: Settings of the preview and of its viewer session.
    """
    visualization_cfg = cfg.get("visualization", {})
    prediction_frequency_hz = visualization_cfg.get("prediction_frequency_hz", None)
    class_names = visualization_cfg.get("class_names", None)
    segmentation_class_names = visualization_cfg.get("segmentation3d_class_names", None)
    detection_class_names = visualization_cfg.get("detection3d_class_names", None)
    return VisualizationPreviewConfig(
        mode=str(visualization_cfg.get("mode", "auto")),
        split=str(visualization_cfg.get("split", "test")),
        sample_index=int(visualization_cfg.get("sample_index", 0)),
        max_samples=int(visualization_cfg.get("max_samples", 1)),
        prediction_frequency_hz=(
            float(prediction_frequency_hz) if prediction_frequency_hz is not None else None
        ),
        device=str(visualization_cfg.get("device", "auto")),
        point_labels=bool(visualization_cfg.get("point_labels", False)),
        class_names=tuple(map(str, class_names)) if class_names else None,
        segmentation_class_names=(
            tuple(map(str, segmentation_class_names)) if segmentation_class_names else None
        ),
        detection_class_names=(
            tuple(map(str, detection_class_names)) if detection_class_names else None
        ),
        session=VisualizationSessionConfig(
            backend=str(visualization_cfg.get("backend", "rerun")),
            application_id=str(visualization_cfg.get("application_id", "autoware-ml")),
            recording_id=visualization_cfg.get("recording_id"),
            web_port=int(visualization_cfg.get("web_port", 9090)),
            grpc_port=int(visualization_cfg.get("grpc_port", 9876)),
            wait=bool(visualization_cfg.get("wait", True)),
            server_memory_limit=str(visualization_cfg.get("server_memory_limit", "25%")),
            timeline=str(visualization_cfg.get("timeline", "frame")),
            point_color_mode=str(visualization_cfg.get("point_color_mode", "semantic")),
            camera_frustums_visible=bool(visualization_cfg.get("camera_frustums_visible", False)),
            sync_views=bool(visualization_cfg.get("sync_views", False)),
            sync_port=int(visualization_cfg.get("sync_port", 9091)),
        ),
    )


def resolve_preview_weights(cfg: DictConfig, mode: str) -> Sequence[Path] | None:
    """
    Read the checkpoints of a preview, only when the preview builds a model.

    A data preview builds no model, so a configuration whose weights are mandatory (``???``),
    such as a fine-tuning one, must not fail on them there. In the automatic mode such a
    configuration without weights previews the data.

    Args:
        cfg: Hydra configuration.
        mode: Preview mode, ``auto``, ``predictions`` or ``data``.

    Returns:
        Sequence[Path] | None: The checkpoints to load, None for a preview without a model.

    Raises:
        ValueError: If a prediction preview has no weights, or they are not a list.
    """
    if mode == "data" or (mode == "auto" and OmegaConf.is_missing(cfg, "weights")):
        return None
    return resolve_weights_paths(cfg)
