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

"""Builder of the small motion heads of the motion tests."""

from __future__ import annotations

import numpy as np
import torch

from autoware_ml.anchors.tests.fixtures import FixedAnchorRecipe, fill_anchors
from autoware_ml.models.sparsedrive.motion_head import MotionHead


def build_motion_head(num_classes: int, **overrides: object) -> MotionHead:
    """A small motion head over 32 channels and 4 future steps, its random anchors filled.

    Args:
        num_classes: Detection classes of the head.
        **overrides: Settings replacing the defaults of the builder.

    Returns:
        The head, built from a fixed seed.
    """
    rng = np.random.default_rng(0)
    anchors = FixedAnchorRecipe({"anchors": rng.normal(size=(num_classes, 6, 4, 2))})
    torch.manual_seed(0)
    kwargs: dict[str, object] = {
        "num_classes": num_classes,
        "embed_dims": 32,
        "anchor_recipe": anchors,
        "num_heads": 4,
        "agent_steps": 4,
        "num_agent_keys": 5,
        "queue_length": 1,
        "queue_stride": 1,
        **overrides,
    }
    return fill_anchors(MotionHead(**kwargs))
