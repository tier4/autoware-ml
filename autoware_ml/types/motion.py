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


from enum import IntEnum


class AgentRole(IntEnum):
    """
    Role of an agent inside a motion forecasting scenario.

    The values equal the Argoverse 2 ``object_category`` codes so the raw column maps onto
    this enum without a lookup table.

    Attributes:
      TRACK_FRAGMENT: Partial track that provides context only.
      UNSCORED: Full track that is not evaluated.
      SCORED: Full track that is evaluated.
      FOCAL: The single agent the scenario is centred on.
    """

    TRACK_FRAGMENT = 0
    UNSCORED = 1
    SCORED = 2
    FOCAL = 3


class PolylineType(IntEnum):
    """
    Kind of a vector map polyline.

    Attributes:
      LANE_CENTERLINE: Centerline of a lane segment.
      LANE_LEFT_BOUNDARY: Left boundary of a lane segment.
      LANE_RIGHT_BOUNDARY: Right boundary of a lane segment.
      CROSSWALK_EDGE: One of the two edges of a pedestrian crossing.
      DRIVABLE_AREA_BOUNDARY: Exterior boundary of a drivable area.
    """

    LANE_CENTERLINE = 0
    LANE_LEFT_BOUNDARY = 1
    LANE_RIGHT_BOUNDARY = 2
    CROSSWALK_EDGE = 3
    DRIVABLE_AREA_BOUNDARY = 4
