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

"""Detection ground truth of a batch and the frames it supervises, shared by the joint models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from jaxtyping import Bool

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs


def detection3d_gt_batch(batch_inputs: ModelBatchInputs) -> Detection3DGTBatch:
    """Read the detection ground truth of the batch.

    Args:
        batch_inputs: Model inputs of the batch.

    Returns:
        The detection ground truth.

    Raises:
        ValueError: If the batch carries no detection ground truth.
    """
    gt_detections = batch_inputs.multi_task_gt_batch.detection3d_gt_batch
    if gt_detections is None:
        raise ValueError("The detection loss and metrics need the 3D detection ground truth.")
    return gt_detections


def detection3d_frame_mask(gt_detections: Detection3DGTBatch) -> Bool[torch.Tensor, " batch_size"]:
    """Return the per-frame detection supervision mask.

    Supervision is carried by the annotations themselves: a frame with no ground-truth box
    contributes no detection supervision. This includes genuinely empty scenes, which therefore
    provide no pure-background signal - the deliberate price of not carrying a separate flag.

    Args:
        gt_detections: Detection ground truth of the batch.

    Returns:
        Boolean tensor of shape ``(batch_size,)``.
    """
    return gt_detections.gt_valid_bboxes > 0


def empty_unsupervised_predictions(
    predictions: Sequence[Detection3DSamplePredictions],
    det3d_mask: Bool[torch.Tensor, " batch_size"],
) -> list[Detection3DSamplePredictions]:
    """Empty the predictions of the frames without detection supervision.

    Such frames still contribute one (empty) entry: the detection metric state must grow by
    exactly one entry per frame on every rank, or torchmetrics' per-element list-state
    ``all_gather`` deadlocks under DDP when ranks see different seg/det frame mixes. Empty
    prediction + empty ground truth is metric-neutral.

    Args:
        predictions: Decoded predictions, one per frame.
        det3d_mask: Per-frame detection supervision mask.

    Returns:
        The predictions with the unsupervised frames emptied.
    """
    return [
        prediction
        if flagged
        else Detection3DSamplePredictions(
            bboxes_3d=prediction.bboxes_3d[:0],
            scores_3d=prediction.scores_3d[:0],
            labels_3d=prediction.labels_3d[:0],
            instance_ids=None if prediction.instance_ids is None else prediction.instance_ids[:0],
        )
        for prediction, flagged in zip(predictions, det3d_mask.tolist(), strict=True)
    ]
