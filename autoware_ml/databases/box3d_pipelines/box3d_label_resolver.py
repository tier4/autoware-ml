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

from collections.abc import Sequence

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelVocabulary


class Box3DLabelResolver(Box3DPipeline):
    """
    Pipeline to turn the raw dataset label names of the 3D bounding boxes into the fine names of
    the vocabulary. A raw name the vocabulary maps to null drops its box, and a raw name the
    vocabulary does not list raises.
    """

    def __init__(self, vocabulary: LabelVocabulary):
        """
        Initialize Box3DLabelResolver.

        Args:
          vocabulary: Vocabulary that maps the raw names of the dataset to fine names.
        """
        super().__init__()
        self.vocabulary = vocabulary

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return f"{self.__class__.__name__}(vocabulary={self.vocabulary})"

    def __call__(self, boxes3d_data_model: Sequence[Box3DDataModel]) -> Sequence[Box3DDataModel]:
        """
        Resolve the raw label names of the 3D bounding boxes into fine names.
        """
        new_boxes3d_data_model = []
        for box3d_data_model in boxes3d_data_model:
            fine_name = self.vocabulary.fine_name(box3d_data_model.box3d_label_name)
            if fine_name is None:
                continue

            new_boxes3d_data_model.append(
                box3d_data_model.create_new_data_model(box3d_label_name=fine_name)
            )

        return new_boxes3d_data_model
