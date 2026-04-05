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

"""Visualize detection2d predictions for configured models and datamodules."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from autoware_ml.utils.checkpoints import load_model_from_checkpoint
from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    get_config_path,
    log_configuration,
    resolve_work_dir,
    set_seed,
)
from autoware_ml.visualization import (
    build_label_names,
    draw_detection_preview,
    targets_to_absolute_xyxy,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, list):
        return [_move_to_device(item, device) for item in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_device(item, device) for item in batch)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    return batch


def _select_loader(datamodule: L.LightningDataModule, split: str):
    if split == "predict":
        datamodule.setup("predict")
        return datamodule.predict_dataloader(), datamodule.predict_dataset
    if split == "val":
        datamodule.setup("validate")
        return datamodule.val_dataloader(), datamodule.val_dataset
    if split == "test":
        datamodule.setup("test")
        return datamodule.test_dataloader(), datamodule.test_dataset
    raise ValueError(f"Unsupported visualization split: {split}")


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Run prediction visualization for a detection2d task config."""
    log_configuration(cfg)
    work_dir = resolve_work_dir()
    logger.info("Working directory: %s", work_dir)

    configure_torch_runtime()
    set_seed(cfg)

    visualization_cfg = cfg.get("visualization", {})
    output_dir = Path(visualization_cfg.get("out_dir", work_dir / "predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)
    score_threshold = float(visualization_cfg.get("score_threshold", 0.25))
    max_images = int(visualization_cfg.get("max_images", 32))
    draw_ground_truth = bool(visualization_cfg.get("draw_ground_truth", True))
    line_thickness = int(visualization_cfg.get("line_thickness", 2))
    split = str(visualization_cfg.get("split", "predict"))
    save_json = bool(visualization_cfg.get("save_json", True))

    logger.info("Instantiating datamodule...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    logger.info("Instantiating model...")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is not None:
        logger.info("Loading Lightning checkpoint: %s", checkpoint_path)
        load_model_from_checkpoint(model, Path(checkpoint_path), map_location="cpu")

    dataloader, dataset = _select_loader(datamodule, split)
    categories = getattr(dataset, "categories", None)
    num_classes = getattr(getattr(model, "postprocessor", None), "num_classes", None)
    label_names = build_label_names(categories, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results: list[dict[str, Any]] = []
    saved = 0
    with torch.no_grad():
        for batch_idx, batch_inputs_dict in enumerate(dataloader):
            batch_on_device = _move_to_device(batch_inputs_dict, device)
            predictions = model.predict_step(batch_on_device, batch_idx)

            for sample_idx, prediction in enumerate(predictions):
                if saved >= max_images:
                    break

                image_path = Path(batch_inputs_dict["img_paths"][sample_idx])
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    logger.warning("Skipping unreadable image: %s", image_path)
                    continue

                gt = None
                if draw_ground_truth:
                    gt = targets_to_absolute_xyxy(
                        batch_inputs_dict["targets"][sample_idx],
                        batch_inputs_dict["orig_sizes"][sample_idx],
                    )

                preview = draw_detection_preview(
                    image_bgr,
                    prediction,
                    label_names=label_names,
                    score_threshold=score_threshold,
                    line_thickness=line_thickness,
                    ground_truth=gt,
                )

                output_path = output_dir / f"{saved:04d}_{image_path.name}"
                cv2.imwrite(str(output_path), preview)

                keep = prediction["scores"].detach().cpu() >= score_threshold
                results.append(
                    {
                        "image_path": str(image_path),
                        "output_path": str(output_path),
                        "image_id": int(batch_inputs_dict["image_ids"][sample_idx]),
                        "predictions": [
                            {
                                "label": int(label),
                                "label_name": label_names.get(int(label), f"class_{int(label)}"),
                                "score": float(score),
                                "box_xyxy": [float(value) for value in box.tolist()],
                            }
                            for box, score, label in zip(
                                prediction["boxes"].detach().cpu()[keep],
                                prediction["scores"].detach().cpu()[keep],
                                prediction["labels"].detach().cpu()[keep],
                                strict=False,
                            )
                        ],
                    }
                )
                saved += 1

            if saved >= max_images:
                break

    if save_json:
        result_path = output_dir / "predictions.json"
        result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved prediction summary to: %s", result_path)

    logger.info("Saved %d visualizations to: %s", saved, output_dir)


if __name__ == "__main__":
    main()
