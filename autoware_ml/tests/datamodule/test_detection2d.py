"""Tests for COCO-style detection datamodules."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import torch

from autoware_ml.datamodule.coco import COCODetectionDataModule
from autoware_ml.datamodule.detection2d.base import generate_scales
from autoware_ml.transforms import TransformsCompose
from autoware_ml.transforms.detection2d import (
    ConvertBoxes,
    ConvertPILImage,
    LoadDetectionImageFromFile,
    Resize,
    ToTorchVisionTensors,
)


def _write_sample_dataset(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (40, 20), color=(120, 80, 40)).save(image_path)

    ann_path = tmp_path / "instances.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "sample.jpg", "width": 40, "height": 20}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 5, 20, 10], "area": 200, "iscrowd": 0}
                ],
                "categories": [{"id": 7, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )
    return image_dir, ann_path


def test_generate_scales_matches_expected_shape() -> None:
    scales = generate_scales(640, 3)

    assert 640 in scales
    assert min(scales) >= 480
    assert max(scales) <= 800


def test_coco_detection_datamodule_collates_detection_batch(tmp_path: Path) -> None:
    image_dir, ann_path = _write_sample_dataset(tmp_path)
    transforms = TransformsCompose(
        [
            LoadDetectionImageFromFile(),
            ToTorchVisionTensors(),
            Resize(size=(32, 32)),
            ConvertPILImage(),
            ConvertBoxes(fmt="cxcywh", normalize=True),
        ]
    )

    datamodule = COCODetectionDataModule(
        data_root=str(tmp_path),
        train_ann_file=str(ann_path),
        val_ann_file=str(ann_path),
        train_img_root=str(image_dir),
        val_img_root=str(image_dir),
        train_transforms=transforms,
        val_transforms=transforms,
        train_dataloader_cfg={"batch_size": 1, "num_workers": 0},
        val_dataloader_cfg={"batch_size": 1, "num_workers": 0},
    )
    datamodule.setup(stage="fit")

    batch = datamodule.collate_fn([datamodule.train_dataset[0]])

    assert batch["images"].shape == (1, 3, 32, 32)
    assert batch["targets"][0]["boxes"].shape == (1, 4)
    assert torch.all(batch["targets"][0]["boxes"] <= 1.0)
    assert batch["image_ids"].tolist() == [1]


def test_coco_detection_datamodule_subset_filters_dataset_and_evaluator(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(image_dir / "one.jpg")
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(image_dir / "two.jpg")

    ann_path = tmp_path / "instances.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "one.jpg", "width": 32, "height": 32},
                    {"id": 2, "file_name": "two.jpg", "width": 32, "height": 32},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 7, "bbox": [4, 4, 8, 8], "area": 64, "iscrowd": 0},
                    {"id": 2, "image_id": 2, "category_id": 7, "bbox": [8, 8, 12, 12], "area": 144, "iscrowd": 0},
                ],
                "categories": [{"id": 7, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )

    datamodule = COCODetectionDataModule(
        data_root=str(tmp_path),
        train_ann_file=str(ann_path),
        val_ann_file=str(ann_path),
        train_img_root=str(image_dir),
        val_img_root=str(image_dir),
        train_dataloader_cfg={"batch_size": 1, "num_workers": 0},
        val_dataloader_cfg={"batch_size": 1, "num_workers": 0},
        max_val_samples=1,
        val_sample_ids=[2],
    )
    datamodule.setup(stage="fit")

    assert len(datamodule.val_dataset) == 1
    assert datamodule.val_dataset[0]["img_id"] == 2
    coco_api = datamodule.val_dataset.get_coco_api()
    assert sorted(coco_api.getImgIds()) == [2]
