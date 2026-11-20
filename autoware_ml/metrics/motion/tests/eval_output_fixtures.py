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

"""Builders of the typed motion eval output for the motion metric tests."""

from __future__ import annotations

import torch

from autoware_ml.metrics.eval_output import EvalOutput
from autoware_ml.metrics.motion.eval_output import MotionEvalFrame, MotionEvalOutput


def motion_output(
    frames: list[dict[str, torch.Tensor]],
    observed: list[torch.Tensor],
    scored: list[bool] | None = None,
) -> EvalOutput:
    """Typed motion frames of a batch, every frame scored unless told otherwise.

    Args:
        frames: The fields of every frame, keyed by the field names of the motion frame.
        observed: Whether the stream observed every ground truth agent, per frame.
        scored: Whether the metrics score each frame, every frame when ``None``.

    Returns:
        The eval output holding the motion frames.
    """
    return EvalOutput(
        motion=MotionEvalOutput(
            frames=[
                MotionEvalFrame(**fields, gt_observed=frame_observed, scored=frame_scored)
                for fields, frame_observed, frame_scored in zip(
                    frames, observed, scored or [True] * len(frames), strict=True
                )
            ]
        )
    )
