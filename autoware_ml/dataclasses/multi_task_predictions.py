"""
Modules to save decoded predictions from multi-task models.
"""

from typing import Sequence

from jaxtyping import Float32
from pydantic import BaseModel, ConfigDict
import torch

from autoware_ml.dataclasses.detection3d.predictions import Detection3DSamplePredictions


class MultiTaskPredictions(BaseModel):
    """
    Dataclass to save decoded predictions from multi-task models.

    Attributes:
      detection_3d_predictions: Decoded predictions from a 3D detection task.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # Decoded predictions across samples.
    detection3d_predictions: Sequence[Detection3DSamplePredictions] | None

    # TODO (Kok Seang): Add predictions for other tasks in the future.

    def to_list(
        self,
    ) -> list[
        dict[
            str,
            Float32[torch.Tensor, "num_boxes num_bbox_params"]
            | Float32[torch.Tensor, " num_boxes"],
        ]
    ]:
        """
        Convert the MultiTaskPredictions to a list of dictionaries.
        Note that this method is primarily for compatibility with the existing detection metric,
        which expects a list of dictionaries.
        Returns:
            A list of dictionaries representing the predictions for each sample.
        """
        predictions_list = []
        if self.detection3d_predictions is None:
            return []

        for detection_pred in self.detection3d_predictions:
            predictions_list.append(
                {
                    "bboxes_3d": detection_pred.bboxes_3d,
                    "scores_3d": detection_pred.scores_3d,
                    "labels_3d": detection_pred.labels_3d,
                }
            )
        return predictions_list
