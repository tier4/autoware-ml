"""Tests for detection2d transforms."""

from __future__ import annotations

from PIL import Image
import torch

from autoware_ml.transforms import TransformsCompose
from autoware_ml.transforms.detection2d import (
    ConvertBoxes,
    ConvertPILImage,
    Resize,
    SanitizeBoundingBoxes,
    ToTorchVisionTensors,
)


def test_sanitize_preserves_box_metadata_for_later_resize() -> None:
    image = Image.new("RGB", (40, 20), color=(120, 80, 40))
    target = {
        "boxes": torch.tensor([[10.0, 5.0, 30.0, 15.0]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
        "area": torch.tensor([200.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }
    transforms = TransformsCompose(
        [
            ToTorchVisionTensors(),
            SanitizeBoundingBoxes(min_size=1.0),
            Resize(size=(32, 32)),
            ConvertPILImage(),
            ConvertBoxes(fmt="cxcywh", normalize=True),
        ]
    )

    sample = transforms({"image": image, "target": target})
    boxes = sample["target"]["boxes"]

    assert torch.all(boxes >= 0.0)
    assert torch.all(boxes <= 1.0)
    assert torch.allclose(
        boxes,
        torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
        atol=1e-4,
    )
