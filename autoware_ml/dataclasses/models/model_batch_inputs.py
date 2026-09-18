from __future__ import annotations

from pydantic import BaseModel, ConfigDict, InstanceOf

from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.ops.voxelization.voxelization import VoxelsData


class ModelBatchInputs(BaseModel):
    """Data class to represent the gt batch and data features for inputs to a multi-task model."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # InstanceOf keeps pydantic from recursing into the NamedTuple's fields, whose jaxtyping
    # annotations use symbolic axes (e.g. "batch_size*num_points") that can only be resolved
    # inside a @jaxtyped scope.
    multi_task_gt_batch: InstanceOf[ModelGTBatch]

    voxels_data: VoxelsData | None

    # Image data
    image_data: ImageGTBatch | None

    # TODO(Kok Seang): Add input features for 3D segmentation model.

    # TODO(Kok Seang): Add input features for 2D detection/segmentation model.
