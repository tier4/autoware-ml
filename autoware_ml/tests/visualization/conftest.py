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

"""Shared fixtures and doubles for the visualization test suite."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
import torch

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.models.base import BaseModel
from autoware_ml.utils.calibration import CalibrationData
from autoware_ml.visualization.events import VisualizationEvent


class RecordingBackend:
    """Capture every event and step handed to a visualization backend."""

    def __init__(self) -> None:
        """Initialize empty step and event logs."""
        self.steps: list[int] = []
        self.timestamps: list[float] = []
        self.events: list[Any] = []
        self.waited = False

    def set_step(self, step: int) -> None:
        """Record one timeline step."""
        self.steps.append(step)

    def set_timestamp(self, timestamp: float) -> None:
        """Record one sensor timestamp."""
        self.timestamps.append(timestamp)

    def log_event(self, event: VisualizationEvent) -> None:
        """Record one visualization event."""
        self.events.append(event)

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """Record multiple visualization events."""
        self.events.extend(events)

    def wait_until_interrupted(self) -> None:
        """Record that the preview asked the backend to wait."""
        self.waited = True

    def paths_of(self, event_type: type) -> list[str]:
        """Return the entity paths of every recorded event of one type."""
        return [event.path for event in self.events if isinstance(event, event_type)]


class PreviewDataset(Dataset):
    """Serve pre-built sample dictionaries to the preview pipeline."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        """Initialize the dataset from a list of samples."""
        super().__init__()
        self.samples = samples

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def get_data_info(self, index: int) -> dict[str, Any]:
        """Return the raw sample info for one index."""
        return self.samples[index]


class PreviewDataModule(DataModule):
    """Wrap ``PreviewDataset`` with an explicit collation map."""

    def __init__(self, samples: list[dict[str, Any]], collation_map: dict[str, str]) -> None:
        """Initialize the datamodule from samples and a collation map."""
        super().__init__(collation_map=collation_map)
        self.samples = samples

    def _create_dataset(self, split: str, dataset_transforms: Any = None) -> Dataset:
        """Return the preview dataset for any requested split."""
        del split, dataset_transforms
        return PreviewDataset(self.samples)


class PreviewModelBase(BaseModel):
    """Minimal Lightning model shared by the preview task doubles."""

    def forward(self, **kwargs: Any) -> Any:
        """Return nothing because preview tests only call ``predict_step``."""
        del kwargs
        return None

    def compute_metrics(
        self, batch_inputs_dict: dict[str, Any], outputs: Any
    ) -> dict[str, torch.Tensor]:
        """Return a constant loss to satisfy the base-model contract."""
        del batch_inputs_dict, outputs
        return {"loss": torch.zeros(())}


class CalibrationPreviewModel(PreviewModelBase):
    """Return fixed calibration-status probabilities."""

    def predict_step(self, batch_inputs_dict: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Return one two-class probability row."""
        del batch_inputs_dict, batch_idx
        return torch.tensor([[0.1, 0.9]], dtype=torch.float32)


class SegmentationPreviewModel(PreviewModelBase):
    """Return alternating per-point labels for the points in the batch."""

    def predict_step(
        self, batch_inputs_dict: dict[str, Any], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Return alternating labels and matching one-hot probabilities."""
        del batch_idx
        points = batch_inputs_dict["points"]
        if isinstance(points, list):
            points = points[0]
        pred_labels = torch.arange(points.shape[0], dtype=torch.long) % 2
        pred_logits = torch.nn.functional.one_hot(pred_labels, num_classes=2).float() * 8.0
        return {
            "pred_labels": pred_labels,
            "pred_probs": torch.softmax(pred_logits, dim=1),
            "pred_logits": pred_logits,
        }


class VoxelizedSegmentationPreviewModel(PreviewModelBase):
    """Return point-level labels sized from the ``inverse`` voxel mapping."""

    def predict_step(
        self, batch_inputs_dict: dict[str, Any], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Return labels aligned with the original, pre-voxelization points."""
        del batch_idx
        inverse = batch_inputs_dict["inverse"].long()
        pred_labels = torch.arange(inverse.shape[0], device=inverse.device, dtype=torch.long) % 2
        pred_logits = torch.nn.functional.one_hot(pred_labels, num_classes=2).float() * 8.0
        return {
            "pred_labels": pred_labels,
            "pred_probs": torch.softmax(pred_logits, dim=1),
            "pred_logits": pred_logits,
        }


class DetectionPreviewModel(PreviewModelBase):
    """Return one decoded 3D box in the list-of-dicts prediction format."""

    def predict_step(
        self, batch_inputs_dict: dict[str, Any], batch_idx: int
    ) -> list[dict[str, torch.Tensor]]:
        """Return a single-box detection prediction."""
        del batch_inputs_dict, batch_idx
        return [
            {
                "bboxes_3d": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.1]], dtype=torch.float32
                ),
                "scores_3d": torch.tensor([0.9], dtype=torch.float32),
                "labels_3d": torch.tensor([1], dtype=torch.long),
            }
        ]


@pytest.fixture
def recording_backend() -> RecordingBackend:
    """Return a fresh recording backend."""
    return RecordingBackend()


@pytest.fixture
def preview_calibration_data() -> CalibrationData:
    """Return calibration data with an identity-rotation extrinsic."""
    return CalibrationData(
        camera_matrix=np.array(
            [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        distortion_coefficients=np.zeros((5,), dtype=np.float32),
        lidar_to_camera_transformation=np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
