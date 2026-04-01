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

"""T4Dataset datamodule exports used by Autoware-ML entrypoints.

This package re-exports dataset and datamodule implementations for T4Dataset
training, evaluation, and deployment workflows.
"""

from autoware_ml.datamodule.t4dataset.calibration_status import (
    T4CalibrationDataModule,
    T4CalibrationStatusDataset,
)
from autoware_ml.datamodule.t4dataset.segmentation3d import (
    T4Segmentation3DDataModule,
    T4Segmentation3DDataset,
)

__all__ = [
    "T4CalibrationStatusDataset",
    "T4CalibrationDataModule",
    "T4Segmentation3DDataset",
    "T4Segmentation3DDataModule",
]
