"""
Modules to save the batched inputs to multi-task models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, InstanceOf

from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.range_view import RangeViewData
from autoware_ml.dataclasses.geometry.voxels import VoxelsData


class ModelBatchInputs(BaseModel):
    """Data class to represent the gt batch and data features for inputs to a multi-task model."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # Every field below is a NamedTuple. InstanceOf keeps pydantic from recursing into their
    # fields, whose jaxtyping annotations use symbolic axes (e.g. "batch_size*num_points") that
    # can only be resolved inside a @jaxtyped scope, and it also keeps the validated payload the
    # very object that was passed in instead of a rebuilt copy.
    multi_task_gt_batch: InstanceOf[ModelGTBatch]

    voxels_data: InstanceOf[VoxelsData] | None

    # Image data
    image_data: InstanceOf[ImageGTBatch] | None

    # Point cloud projected into range images
    range_view_data: InstanceOf[RangeViewData] | None

    # TODO(Kok Seang): Add input features for 2D detection/segmentation model.

    @classmethod
    def from_gt_batch(cls, batch: ModelGTBatch) -> ModelBatchInputs:
        """
        Start the model inputs of a collated batch, before any preprocessing.

        Args:
            batch: Collated batch on the target device.

        Returns:
            The batch with the images it carries and no preprocessed features.
        """
        return cls(
            multi_task_gt_batch=batch,
            voxels_data=None,
            image_data=batch.image_gt_batch,
            range_view_data=None,
        )

    def replace(self, **fields: Any) -> ModelBatchInputs:
        """
        Copy the inputs with some fields replaced.

        Args:
            **fields: New values of the replaced fields.

        Returns:
            The validated copy.
        """
        current = {name: getattr(self, name) for name in type(self).model_fields}
        return type(self)(**(current | fields))

    def batch_size(self) -> int:
        """
        Give the number of samples of the batch.

        Returns:
            The sample count of the collated batch.
        """
        return int(self.multi_task_gt_batch.infer_batch_size())
