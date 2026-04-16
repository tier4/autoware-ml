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

"""COCO-style evaluation helpers for 2D detection tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import multiprocessing as mp
import os
import re
from typing import Any

import numpy as np
import torch
import torchvision


def _require_pycocotools() -> tuple[Any, Any]:
    """Import pycocotools lazily so the rest of the framework can still import."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for detection2d evaluation. "
            "Install the project dependencies or add pycocotools to the environment."
        ) from exc
    return COCO, COCOeval


def build_coco_api_from_dataset_dict(dataset_dict: Mapping[str, Any]) -> Any:
    """Build a ``pycocotools.coco.COCO`` object from an in-memory dataset dictionary."""
    COCO, _ = _require_pycocotools()
    coco = COCO()
    coco.dataset = copy.deepcopy(dict(dataset_dict))
    coco.createIndex()
    return coco


def build_empty_coco_results(coco_gt: Any) -> Any:
    """Build an empty COCO results object for images without predictions."""
    COCO, _ = _require_pycocotools()
    coco_dt = COCO()
    coco_dt.dataset["images"] = copy.deepcopy(coco_gt.dataset.get("images", []))
    coco_dt.dataset["categories"] = copy.deepcopy(coco_gt.dataset.get("categories", []))
    coco_dt.dataset["annotations"] = []
    coco_dt.createIndex()
    return coco_dt


def _sanitize_metric_name(value: str) -> str:
    """Convert free-form category names into stable metric key components."""
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "category"


def _extract_per_class_ap(evaluator: Any, categories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract per-category AP over COCO's default IoU sweep."""
    precision = evaluator.eval.get("precision")
    if precision is None or precision.size == 0:
        return []

    area_labels = list(getattr(evaluator.params, "areaRngLbl", []))
    max_dets = list(getattr(evaluator.params, "maxDets", []))
    area_index = area_labels.index("all") if "all" in area_labels else 0
    max_det_index = max_dets.index(100) if 100 in max_dets else len(max_dets) - 1
    category_names = {
        int(category["id"]): str(category.get("name", category["id"])) for category in categories
    }

    per_class_metrics: list[dict[str, Any]] = []
    for category_index, category_id in enumerate(getattr(evaluator.params, "catIds", [])):
        category_precision = precision[:, :, category_index, area_index, max_det_index]
        valid_precision = category_precision[category_precision > -1]
        ap = float(valid_precision.mean()) if valid_precision.size else 0.0
        category_name = category_names.get(int(category_id), str(category_id))
        per_class_metrics.append(
            {
                "category_id": int(category_id),
                "category_name": category_name,
                "metric_name": _sanitize_metric_name(category_name),
                "ap": ap,
            }
        )
    return per_class_metrics


def _evaluate_coco_results_worker(
    output_path: str,
    dataset_dict: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    iou_type: str,
    include_per_class_metrics: bool,
) -> None:
    """Run COCO evaluation in a fresh subprocess.

    Repeated COCO evaluation on large validation sets has triggered native
    crashes in the long-running trainer process on this host. Running the
    evaluator in a short-lived spawned process isolates that instability while
    preserving the reported COCO metrics.
    """
    try:
        coco_gt = build_coco_api_from_dataset_dict(dataset_dict)
        coco_dt = coco_gt.loadRes(results) if results else build_empty_coco_results(coco_gt)
        _, COCOeval = _require_pycocotools()
        evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        payload = {"stats": [float(value) for value in evaluator.stats]}
        if include_per_class_metrics:
            payload["per_class_ap"] = _extract_per_class_ap(
                evaluator,
                categories=dataset_dict.get("categories", []),
            )
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file)
        file.flush()
        os.fsync(file.fileno())
    # Exit immediately to avoid interpreter teardown crashes observed after
    # repeated COCO evaluations on this host.
    os._exit(0)


def _evaluate_coco_results_in_subprocess(
    dataset_dict: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    iou_type: str,
    include_per_class_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate COCO results in a spawned subprocess and return raw payload data."""
    import tempfile

    ctx = mp.get_context("spawn")
    with tempfile.NamedTemporaryFile(prefix="awml_coco_eval_", suffix=".json", delete=False) as file:
        output_path = file.name
    process = ctx.Process(
        target=_evaluate_coco_results_worker,
        args=(
            output_path,
            copy.deepcopy(dict(dataset_dict)),
            list(results),
            iou_type,
            include_per_class_metrics,
        ),
    )
    process.start()
    process.join()
    exit_code = process.exitcode

    try:
        if not os.path.exists(output_path):
            raise RuntimeError("COCO evaluation subprocess exited without returning metrics.")
        with open(output_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
        process.close()

    if exit_code not in (0, None) and "stats" not in payload:
        raise RuntimeError(f"COCO evaluation subprocess exited with code {exit_code}.")
    if "error" in payload:
        raise RuntimeError(f"COCO evaluation subprocess failed: {payload['error']}")
    return payload


def export_coco_predictions(
    predictions: Sequence[Mapping[str, Any]],
    label_to_category_id: Mapping[int, int],
) -> list[dict[str, Any]]:
    """Convert model predictions into COCO result dictionaries."""
    results: list[dict[str, Any]] = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        for box, score, label in zip(boxes, scores, labels, strict=False):
            x1, y1, x2, y2 = box.tolist()
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label_to_category_id[int(label)]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return results


def evaluate_coco_predictions(
    coco_gt: Any,
    predictions: Sequence[Mapping[str, Any]],
    label_to_category_id: Mapping[int, int],
    iou_type: str = "bbox",
    include_per_class_metrics: bool = False,
) -> dict[str, float]:
    """Evaluate predictions against a COCO-style ground-truth API."""
    results = export_coco_predictions(predictions, label_to_category_id)
    payload = _evaluate_coco_results_in_subprocess(
        dataset_dict=coco_gt.dataset,
        results=results,
        iou_type=iou_type,
        include_per_class_metrics=include_per_class_metrics,
    )
    stats = payload["stats"]
    metrics = {
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR@1": float(stats[6]),
        "AR@10": float(stats[7]),
        "AR@100": float(stats[8]),
    }
    if include_per_class_metrics:
        used_metric_names: set[str] = set()
        for class_metric in payload.get("per_class_ap", []):
            metric_name = str(class_metric["metric_name"])
            if metric_name in used_metric_names:
                metric_name = f"{metric_name}_{int(class_metric['category_id'])}"
            used_metric_names.add(metric_name)
            metrics[f"class_mAP/{metric_name}"] = float(class_metric["ap"])
    return metrics


def evaluate_class_agnostic_localization(
    coco_gt: Any,
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Evaluate class-agnostic localization quality for local sanity checks."""
    annotations_by_image: dict[int, list[list[float]]] = {}
    for annotation in coco_gt.dataset.get("annotations", []):
        image_id = int(annotation["image_id"])
        x, y, w, h = annotation["bbox"]
        annotations_by_image.setdefault(image_id, []).append([x, y, x + w, y + h])

    predictions_by_image: dict[int, torch.Tensor] = {}
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        boxes = prediction["boxes"]
        if isinstance(boxes, torch.Tensor):
            boxes_tensor = boxes.detach().cpu().float()
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        predictions_by_image[image_id] = boxes_tensor

    best_ious: list[torch.Tensor] = []
    for image in coco_gt.dataset.get("images", []):
        image_id = int(image["id"])
        gt_boxes = annotations_by_image.get(image_id, [])
        if not gt_boxes:
            continue

        gt_boxes_tensor = torch.as_tensor(gt_boxes, dtype=torch.float32)
        pred_boxes_tensor = predictions_by_image.get(image_id)
        if pred_boxes_tensor is None or pred_boxes_tensor.numel() == 0:
            best_ious.append(torch.zeros(len(gt_boxes_tensor), dtype=torch.float32))
            continue

        pairwise_ious = torchvision.ops.box_iou(pred_boxes_tensor, gt_boxes_tensor)
        best_ious.append(pairwise_ious.max(dim=0).values)

    if not best_ious:
        return {
            "mean_best_iou": 0.0,
            "gt_recall_0p3": 0.0,
            "gt_recall_0p5": 0.0,
        }

    all_best_ious = torch.cat(best_ious, dim=0)
    return {
        "mean_best_iou": float(all_best_ious.mean()),
        "gt_recall_0p3": float((all_best_ious >= 0.3).float().mean()),
        "gt_recall_0p5": float((all_best_ious >= 0.5).float().mean()),
    }
