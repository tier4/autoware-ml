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

from typing import Sequence

from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel


class Box3DPipeline:
    """Base class for box 3D pipelines."""

    def __call__(self, boxes3d_datamodel: Sequence[Box3DDataModel]) -> Box3DDataModel:
        """
        Process the boxes 3D.
        """
        raise NotImplementedError("Subclass must implement this method")

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return self.__class__.__name__
