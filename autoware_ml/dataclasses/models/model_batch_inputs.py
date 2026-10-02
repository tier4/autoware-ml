"""
Modules to save the batched inputs to multi-task models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, InstanceOf

from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
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

    # TODO(Kok Seang): Add input features for 3D segmentation model.

    # TODO(Kok Seang): Add input features for 2D detection/segmentation model.
