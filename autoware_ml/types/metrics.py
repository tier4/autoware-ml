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

from enum import StrEnum, auto


class MotionModel(StrEnum):
    """
    Reachable-set shape of a collision participant.

    Attributes:
      WHEELED: Agent bound to a road surface, reaching along kinematically feasible arcs.
      OMNIDIRECTIONAL: Agent free to move any direction, reaching an isotropic disc.
      STATIC: Agent that never moves and keeps its own footprint.
    """

    WHEELED = auto()
    OMNIDIRECTIONAL = auto()
    STATIC = auto()


class MatchCost(StrEnum):
    """
    Cost the detection matching pairs a prediction with a ground truth box by.

    Attributes:
      CENTER: BEV center distance in meters, the nuScenes matching.
      CORNER: Mean BEV corner distance in meters, which also charges size and yaw errors.
    """

    CENTER = auto()
    CORNER = auto()
