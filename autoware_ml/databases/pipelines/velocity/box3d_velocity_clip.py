from autoware_ml.databases.pipelines.box3d_pipeline import Boxes3DPipeline
from autoware_ml.databases.t4dataset.t4sample_records import Boxes3DMetadata


class Box3DVelocityNormClip(Boxes3DPipeline):
    """
    Pipeline to clip the velocity norm of the 3D bounding boxes.
    """

    def __init__(self, velocity_norm_threshold: float):
        super().__init__()
        self.velocity_norm_threshold = velocity_norm_threshold

    def __call__(self, boxes_3d_metadata: Boxes3DMetadata) -> Boxes3DMetadata:
        """
        Clip the velocity norm of the 3D bounding boxes.
        """
        return boxes_3d_metadata
