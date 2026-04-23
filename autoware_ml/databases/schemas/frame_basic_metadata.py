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

from pydantic import BaseModel, ConfigDict


class FrameBasicMetadata(BaseModel):
    """
    Basic metadata for a frame/record that can be shared by multiple datasets.

    Attributes:
      scenario_id: Scenario ID.
      sample_id: Sample ID.
      sample_index: Sample index.
      timestamp_seconds: Timestamp in seconds.
      scenario_name: Scenario name.
      location: Location.
      vehicle_type: Vehicle type.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    scenario_id: str
    sample_id: str
    sample_index: int
    timestamp_seconds: float
    scenario_name: str
    location: str | None
    vehicle_type: str | None
