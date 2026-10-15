from typing import Sequence

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy


class Box3DLabelRemapper(Box3DPipeline):
    """
    Remap the class index of the 3D boxes from their fine label name through the class_mapping
    of the taxonomy level, for example after a merger renamed boxes. The fine label name is kept
    as it is. A fine name the taxonomy level drops gets the ignore index.
    """

    def __init__(self, taxonomy: LabelTaxonomy):
        """
        Initialize Box3DLabelRemapper.

        Args:
          taxonomy: Taxonomy whose class_mapping folds a fine label name onto its class index.
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
        Remap the class index of the 3D bounding boxes from their current label name.

        The current label name is used, not the dataset label name, so that the pass builds on
        the label name a previous pipeline assigned, for example the target label of a merger.

        Raises:
          KeyError: If a label name is not a fine name of the vocabulary.
        """
        new_boxes3d_data_model = []
        for box3d_data_model in boxes3d_data_model:
            new_boxes3d_data_model.append(
                box3d_data_model.create_new_data_model(
                    box3d_label_index=self.taxonomy.class_index(box3d_data_model.box3d_label_name),
                )
            )

        return new_boxes3d_data_model
