# Copyright 2025 TIER IV, Inc.
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

"""Learning-rate scheduler exports for Autoware-ML.

This package exposes reusable scheduler utilities and learning-rate policies
used by model configuration files.
"""

from autoware_ml.utils.schedulers.cosine_annealing import CosineAnnealingLR
from autoware_ml.utils.schedulers.cyclic_cosine_annealing import CyclicCosineAnnealingLR
from autoware_ml.utils.schedulers.cyclic_momentum import CyclicMomentumScheduler
from autoware_ml.utils.schedulers.linear_warmup_cosine_annealing import (
    LinearWarmupCosineAnnealingLR,
)

__all__ = [
    "CosineAnnealingLR",
    "CyclicCosineAnnealingLR",
    "CyclicMomentumScheduler",
    "LinearWarmupCosineAnnealingLR",
]
