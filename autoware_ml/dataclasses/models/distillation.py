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

"""Teacher targets a distilled model reads next to its ground truth."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch

from autoware_ml.dataclasses.models.model_outputs import ModelOutputs


class DistillationTargets(NamedTuple):
    """Outputs of a frozen teacher for the batch the student trains on.

    Attributes:
        outputs: Raw teacher outputs, structured like the student's own :class:`ModelOutputs`.
        features: Intermediate teacher tensors captured by module path, keyed by the
            ``module:`` tap that asked for them.
    """

    outputs: ModelOutputs
    features: Mapping[str, torch.Tensor]
