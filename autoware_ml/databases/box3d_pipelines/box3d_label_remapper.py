from typing import Mapping, Sequence

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy


class Box3DLabelRemapper(Box3DPipeline):
    """
    Rename the fine label names of the 3D boxes, for example the names left by a merger. A name
    mapped to null drops its box. A name missing from the mapping keeps its label. A name the
    taxonomy level drops gets the ignore index.
    """

    def __init__(self, taxonomy: LabelTaxonomy, label_remapper: Mapping[str, str | None]):
        """
        Initialize Box3DLabelRemapper.

        Args:
          taxonomy: Taxonomy that maps a label name to its class index.
          label_remapper: Fine label name to the fine name it folds into, null for a name to
            drop.
        """
        super().__init__()
        self.taxonomy = taxonomy
        self.label_remapper = dict(label_remapper)

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return f"{self.__class__.__name__}(label_remapper={self.label_remapper})"

    def __call__(self, boxes3d_data_model: Sequence[Box3DDataModel]) -> Sequence[Box3DDataModel]:
        """
        Remap the label names of the 3D bounding boxes to another label name.

        The current label name is remapped, not the dataset label name, so that a pipeline can
        run several remappers in a row: every pass builds on the label name the previous pass
        assigned, and a box no pass remaps keeps the dataset label name it started with.
        """
        new_boxes3d_data_model = []
        for box3d_data_model in boxes3d_data_model:
            new_box3d_label_name = self.label_remapper.get(
                box3d_data_model.box3d_label_name, box3d_data_model.box3d_label_name
            )
            if new_box3d_label_name is None:
                continue

            new_boxes3d_data_model.append(
                box3d_data_model.create_new_data_model(
                    box3d_label_name=new_box3d_label_name,
                    box3d_label_index=self.taxonomy.class_index(new_box3d_label_name),
                )
            )

        return new_boxes3d_data_model
