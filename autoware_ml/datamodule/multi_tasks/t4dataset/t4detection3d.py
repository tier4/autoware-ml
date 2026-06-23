import polars as pl

from autoware_ml.datamodule.multi_task.multi_task_dataset_interface import MultiTaskDatasetInterface
from autoware_ml.transforms.base import TransformsCompose


class T4Detection3DDataset(MultiTaskDatasetInterface):
    """
    A dataset class that supports 3D detection task.
    It extends the MultiTaskDatasetInterface class and allows to use it in the
    multi-task pipeline.
    """

    def __init__(
        self, dataset_records_dataframe: pl.DataFrame, transforms: TransformsCompose
    ) -> None:
        """
        Initialize the T4Detection3DDataset class.
        Args:

        """
        super().__init__(dataset_records_dataframe=dataset_records_dataframe, transforms=transforms)
