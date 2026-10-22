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

"""Visualization preview entrypoint for Autoware-ML models.

This script builds the datamodule of a task config and, when weights are given, the model, and
streams consecutive samples of one split to a visualization backend.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.builders.model_builder import (
    build_data_preprocessor,
    build_model,
    resolve_weights_paths,
)
from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    get_config_path,
    log_configuration,
    set_seed,
)
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.preview import (
    VisualizationPreviewConfig,
    resolve_preview_device,
    run_visualization_preview,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()


def _class_names(visualization_cfg: DictConfig, key: str) -> tuple[str, ...] | None:
    """Read an optional class name override, None to use the database taxonomy."""
    names = visualization_cfg.get(key, None)
    return tuple(str(name) for name in names) if names else None


def _build_preview_config(cfg: DictConfig) -> VisualizationPreviewConfig:
    """Normalize optional visualization config values."""
    visualization_cfg = cfg.get("visualization", {})
    return VisualizationPreviewConfig(
        mode=str(visualization_cfg.get("mode", "auto")),
        split=str(visualization_cfg.get("split", "test")),
        sample_index=int(visualization_cfg.get("sample_index", 0)),
        max_samples=int(visualization_cfg.get("max_samples", 1)),
        device=str(visualization_cfg.get("device", "auto")),
        point_labels=bool(visualization_cfg.get("point_labels", False)),
        point_color_mode=str(visualization_cfg.get("point_color_mode", "semantic")),
        class_names=_class_names(visualization_cfg, "class_names"),
        segmentation_class_names=_class_names(visualization_cfg, "segmentation3d_class_names"),
        detection_class_names=_class_names(visualization_cfg, "detection3d_class_names"),
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


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Stream a preview of one split, with predictions when weights are given.

    Args:
        cfg: Hydra configuration.
    """
    log_configuration(cfg)
    configure_torch_runtime()
    set_seed(cfg)

    preview_config = _build_preview_config(cfg)
    weight_paths = resolve_weights_paths(cfg)
    if preview_config.mode == "predictions" and weight_paths is None:
        raise ValueError("Prediction visualization needs --weights <path> (repeatable).")

    datamodule = build_datamodule(cfg)
    model = None
    if preview_config.mode != "data" and weight_paths is not None:
        model = build_model(
            cfg,
            data_preprocessor=build_data_preprocessor(cfg),
            device=resolve_preview_device(preview_config.device),
            weights_path=weight_paths,
            resume_checkpoint_path=None,
            set_eval=True,
            enforce_full_coverage=True,
        )

    logger.info(
        f"Starting visualization preview: mode={preview_config.mode} "
        f"split={preview_config.split} sample_index={preview_config.sample_index} "
        f"max_samples={preview_config.max_samples} device={preview_config.device} "
        f"backend={preview_config.session.backend} web_port={preview_config.session.web_port} "
        f"grpc_port={preview_config.session.grpc_port} wait={preview_config.session.wait}"
    )
    visualized_count = run_visualization_preview(model, datamodule, preview_config)
    logger.info(f"Visualization preview completed for {visualized_count} frame(s).")


if __name__ == "__main__":
    main()
