from types import MappingProxyType
from typing import Sequence

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schema import Box3DDataModel


class Box3DLabelRemapper(Box3DPipeline):
    """
    Pipeline to remap the label names and indices of the 3D bounding boxes to another label name.
    If the new label name for the box3d is not in the target class names, it will map to the ignore label index.
    """

    def __init__(
        self,
        label_remapper: MappingProxyType[str, str],
        class_names: Sequence[str],
        ignore_label_index: int,
    ):
        """
        Initialize Box3DLabelRemapper.

        Args:
          label_remapper: Mapping to remap label names.
          class_names: List of class names in the database, used for category mapping.
          ignore_label_index: Index to use for ignored labels.
        """
        super().__init__()
        self.label_remapper = label_remapper
        self.class_names = class_names
        self.ignore_label_index = ignore_label_index
        self.label_index_remapper = {
            class_name: index for index, class_name in enumerate(class_names)
        }

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return f"{self.__class__.__name__}(label_remapper={self.label_remapper},\
            class_names={self.class_names}, ignore_label_index={self.ignore_label_index})"

    def __call__(self, boxes3d_datamodel: Sequence[Box3DDataModel]) -> Sequence[Box3DDataModel]:
        """
        Remap the label names of the 3D bounding boxes to another label name.
        """
        new_boxes3d_datamodel = []
        for box3d_datamodel in boxes3d_datamodel:
            if box3d_datamodel.box3d_dataset_label_name in self.label_remapper:
                new_box3d_label_name = self.label_remapper[box3d_datamodel.box3d_label_name]
            else:
                new_box3d_label_name = box3d_datamodel.box3d_label_name

            # Map the new label name to the new label index,
            # if the new label name is not in the target class names, map to the ignore label index
            if new_box3d_label_name in self.label_index_remapper:
                new_box3d_label_index = self.label_index_remapper[new_box3d_label_name]
            else:
                new_box3d_label_index = self.ignore_label_index

            new_box3d = box3d_datamodel.create_new_datamodel(
                box3d_label_name=new_box3d_label_name,
                box3d_label_index=new_box3d_label_index,
            )
            new_boxes3d_datamodel.append(new_box3d)

        return new_boxes3d_datamodel
