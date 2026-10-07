from typing import Sequence

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy


class Box3DLabelResolver(Box3DPipeline):
    """
    Pipeline to turn the raw dataset label names of the 3D bounding boxes into the fine names of
    the vocabulary. A raw name the vocabulary maps to null drops its box, and a raw name the
    vocabulary does not list raises.
    """

    def __init__(self, taxonomy: LabelTaxonomy):
        """
        Initialize Box3DLabelResolver.

        Args:
          taxonomy: Taxonomy whose vocabulary maps raw names to fine names and whose levels give
            the class index.
        """
        super().__init__()
        self.taxonomy = taxonomy

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return f"{self.__class__.__name__}(taxonomy={self.taxonomy})"

    def __call__(self, boxes3d_data_model: Sequence[Box3DDataModel]) -> Sequence[Box3DDataModel]:
        """
        Resolve the raw label names of the 3D bounding boxes into fine names.
        """
        new_boxes3d_data_model = []
        for box3d_data_model in boxes3d_data_model:
            fine_name = self.taxonomy.vocabulary.fine_name(box3d_data_model.box3d_label_name)
            if fine_name is None:
                continue

            new_boxes3d_data_model.append(
                box3d_data_model.create_new_data_model(
                    box3d_label_name=fine_name,
                    box3d_label_index=self.taxonomy.class_index(fine_name),
                )
            )

        return new_boxes3d_data_model
