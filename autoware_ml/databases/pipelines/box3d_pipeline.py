"""
This
"""

from autoware_ml.databases.t4dataset.t4sample_records import Boxes3DMetadata


class Boxes3DPipeline:
    """
    Base class for box 3D pipelines.
    """

    def __call__(self, boxes_3d_metadata: Boxes3DMetadata) -> Boxes3DMetadata:
        """
        Process the boxes 3D.
        """
        raise NotImplementedError("Subclass must implement this method")
