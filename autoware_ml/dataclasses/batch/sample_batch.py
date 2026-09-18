from __future__ import annotations

from typing import Sequence, NamedTuple

import torch

from autoware_ml.dataclasses.batch.detection3d import (
    Detection3DGTBatch,
)
from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch, FrameMetaSample
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTSample
from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.dataclasses.geometry.images import ImageGTBatch, ImageSample
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch, LiDARPointCloudSample


class ModelGTSample(NamedTuple):
    """
    Named tuple to represent a single row/sample of multi-task data when inputting to the
    multi-task model.
    """

    # Can be multi-sweep LiDAR point cloud data, which is a list of LiDAR point cloud data rows for each sweep.
    lidar_point_cloud_samples: Sequence[LiDARPointCloudSample] | None
    # Sequence of image data, which is a list of image data row for each sample.
    image_samples: Sequence[ImageSample] | None

    # (number of point clouds, number of features for each point), can be None
    # if it doesn't need to be loaded
    point_cloud_data: BasePoints | None

    # (num_cameras, num_channels, height, width), can be None if it doesn't need to be loaded
    camera_image_data: BaseImages | None

    detection3d_gt_bboxes_3d: BaseBBoxes3D | None
    segmentation3d_gt_sample: Segmentation3DGTSample | None

    # Information about lidar transformation
    lidar_transformation_sample: LiDARTransformationSample | None = None

    # Per-frame evaluation metadata (map pose and scene identifier) read by the metrics.
    # None when the dataset does not provide it.
    frame_meta: FrameMetaSample | None = None

    # Temporary per-sample flag telling whether traffic cones and barriers are annotated in the
    # frame, so 3D detection can treat missing cone/barrier boxes as unlabeled rather than
    # negatives. None when the dataset does not carry the flag.
    detection3d_traffic_cone_barrier_bbox_status: bool | None = None

    # Seconds spent loading this sample and running it through the transform pipeline.
    # Assigned by the dataset once the pipeline has finished.
    io_processing_time: float = 0.0


class ModelGTBatch(NamedTuple):
    """
    Named tuple to represent a batch of multi-task data after collating from sequence of
    ModelGTSample when inputting to the multi-task model.
    """

    # 3D branch
    point_cloud_gt_batch: PointCloudGTBatch | None
    detection3d_gt_batch: Detection3DGTBatch | None

    # TODO (Kok Seang): 3D segmentation

    # Images
    image_gt_batch: ImageGTBatch | None

    # Summed io_processing_time of every sample collated into this batch.
    io_processing_time: float = 0.0

    # Per-frame evaluation metadata, None when the samples carry none.
    frame_meta_batch: FrameMetaBatch | None = None

    def to_device(self, device: torch.device) -> ModelGTBatch:
        """
        Move the ModelGTBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          ModelGTBatch: The batch moved to the specified device.
        """
        return ModelGTBatch(
            point_cloud_gt_batch=self.point_cloud_gt_batch.to_device(device)
            if self.point_cloud_gt_batch is not None
            else None,
            detection3d_gt_batch=self.detection3d_gt_batch.to_device(device)
            if self.detection3d_gt_batch is not None
            else None,
            image_gt_batch=self.image_gt_batch.to_device(device)
            if self.image_gt_batch is not None
            else None,
            io_processing_time=self.io_processing_time,
            frame_meta_batch=self.frame_meta_batch.to_device(device)
            if self.frame_meta_batch is not None
            else None,
        )

    def infer_batch_size(self) -> int:
        """
        Infer the batch size from the collated multi-task GT batch.

        Returns:
            Batch size if it can be inferred, otherwise raises ValueError.
        """
        if self.point_cloud_gt_batch is not None:
            return self.point_cloud_gt_batch.batch_size
        elif self.detection3d_gt_batch is not None:
            return self.detection3d_gt_batch.gt_bboxes_3d.shape[0]
        elif self.image_gt_batch is not None:
            return self.image_gt_batch.images.shape[0]
        else:
            raise ValueError("Cannot infer batch size from an empty ModelGTBatch.")

    @staticmethod
    def collate_pointcloud_gt_samples(
        gt_samples: Sequence[ModelGTSample],
    ) -> PointCloudGTBatch | None:
        """
        Collate a sequence of point cloud GT samples into a PointCloudGTBatch.

        Args:
          gt_samples: Sequence of ModelGTSample to be collated.

        Returns:
          PointCloudGTBatch: Collated point cloud GT batch.
        """
        if len(gt_samples) == 0:
            return None

        pointcloud_samples = []
        for sample in gt_samples:
            if sample.point_cloud_data is None:
                raise ValueError("All samples must have point_cloud_data for collating.")
            pointcloud_samples.append(sample.point_cloud_data)

        point_cloud_gt_batch = PointCloudGTBatch.collate_gt_samples(pointcloud_samples)
        return point_cloud_gt_batch

    @staticmethod
    def collate_detection3d_gt_samples(
        gt_samples: Sequence[ModelGTSample], max_num_3d_gt_bboxes: int
    ) -> Detection3DGTBatch | None:
        """
        Collate a sequence of detection3d GT samples into a Detection3DGTBatch.

        Args:
          gt_samples: Sequence of ModelGTSample to be collated.
          max_num_3d_gt_bboxes: The maximum number of 3D ground truth bounding boxes
            for each sample in the batch.

        Returns:
          Detection3DGTBatch: Collated detection3d GT batch.
        """
        if len(gt_samples) == 0:
            return None

        detection3d_gt_bboxes_3d = []
        detection3d_traffic_cone_barrier_bbox_status = []
        for sample in gt_samples:
            if sample.detection3d_gt_bboxes_3d is None:
                raise ValueError("All samples must have detection3d_gt_bboxes_3d for collating.")

            detection3d_gt_bboxes_3d.append(sample.detection3d_gt_bboxes_3d)
            detection3d_traffic_cone_barrier_bbox_status.append(
                sample.detection3d_traffic_cone_barrier_bbox_status
            )

        detection3d_gt_batch = Detection3DGTBatch.collate_gt_samples(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            max_num_3d_gt_bboxes=max_num_3d_gt_bboxes,
            detection3d_traffic_cone_barrier_bbox_status=detection3d_traffic_cone_barrier_bbox_status,
        )
        return detection3d_gt_batch

    @staticmethod
    def collate_image_gt_samples(gt_samples: Sequence[ModelGTSample]) -> ImageGTBatch | None:
        """
        Collate sequence of ModelGTSample into a ImagesGtBatch

        Args:
          gt_samples: Sequence of ModelGTSample to be collated.
          max_num_3d_gt_bboxes: The maximum number of 3D ground truth bounding boxes
            for each sample in the batch.

        Returns:
          ImageGTBatch: Collated images GT batch.
        """
        if len(gt_samples) == 0:
            return None

        image_gt_samples = []
        for sample in gt_samples:
            if sample.camera_image_data is None:
                raise ValueError("All samples must have camera_image_data for collating.")

            image_gt_samples.append(sample.camera_image_data)

        image_gt_batch = ImageGTBatch.collate_gt_samples(images_gt_samples=image_gt_samples)
        return image_gt_batch

    @staticmethod
    def collate_frame_meta_samples(gt_samples: Sequence[ModelGTSample]) -> FrameMetaBatch | None:
        """
        Collate the frame metadata of a sequence of ModelGTSample into a FrameMetaBatch.

        Args:
          gt_samples: Sequence of ModelGTSample to be collated.

        Returns:
          FrameMetaBatch: Collated frame metadata, None when the samples carry none.

        Raises:
          ValueError: If only some of the samples carry frame metadata.
        """
        if len(gt_samples) == 0:
            return None

        frame_meta_samples = []
        for sample in gt_samples:
            if sample.frame_meta is None:
                raise ValueError("All samples must have frame_meta for collating.")
            frame_meta_samples.append(sample.frame_meta)

        return FrameMetaBatch.collate_gt_samples(
            frame_meta_samples=frame_meta_samples,
            lidar_transformation_samples=[
                sample.lidar_transformation_sample for sample in gt_samples
            ],
        )

    @staticmethod
    def collate_gt_samples(
        gt_samples: Sequence[ModelGTSample], max_num_3d_gt_bboxes: int
    ) -> ModelGTBatch:
        """
        Collate a sequence of ModelGTSample into a ModelGTBatch.

        Args:
          gt_samples: Sequence of ModelGTSample to be collated.

        Returns:
          ModelGTBatch: Collated multi-task GT batch.
        """
        # Collate point cloud GT batch
        point_cloud_gt_batch = ModelGTBatch.collate_pointcloud_gt_samples(gt_samples)

        # Collate detection3d GT batch
        detection3d_gt_batch = ModelGTBatch.collate_detection3d_gt_samples(
            gt_samples=gt_samples, max_num_3d_gt_bboxes=max_num_3d_gt_bboxes
        )

        # Collate image gt batch
        image_gt_batch = ModelGTBatch.collate_image_gt_samples(gt_samples=gt_samples)

        # Collate the per-frame evaluation metadata
        frame_meta_batch = ModelGTBatch.collate_frame_meta_samples(gt_samples=gt_samples)

        return ModelGTBatch(
            point_cloud_gt_batch=point_cloud_gt_batch,
            detection3d_gt_batch=detection3d_gt_batch,
            image_gt_batch=image_gt_batch,
            io_processing_time=sum(sample.io_processing_time for sample in gt_samples),
            frame_meta_batch=frame_meta_batch,
        )
