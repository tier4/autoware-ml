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

"""Top-level model namespace for the Autoware-ML framework.

The package exposes only framework-level base classes. Task-specific models are
exported from their respective subpackages such as
``autoware_ml.models.calibration_status`` or
``autoware_ml.models.detection3d``.
"""

from autoware_ml.models.base import BaseModel

__all__ = ["BaseModel"]
