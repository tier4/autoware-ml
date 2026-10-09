from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from jaxtyping import Bool, Int32, Int64
from torch import Tensor


class Segmentation3DGTSample(NamedTuple):
    """
    Named tuple to represent a single sample of 3D segmentation GT data. The mask holds the
    training label of every point of the cloud, so a label and its point share a position.
    """

    # (N, ), training label of every point of the point cloud
    gt_semantic_mask: Int64[Tensor, " num_points"]
    # Label of a point the taxonomy does not name, dropped by the loss and the metrics
    ignore_index: int

    def remove_labels(self, valid_mask: Bool[Tensor, " num_points"]) -> Segmentation3DGTSample:
        """
        Keep the labels of the masked points, so the mask follows a filtered point cloud.

        Args:
          valid_mask: Mask of the points to keep.

        Returns:
          Segmentation3DGTSample: Sample holding the labels of the kept points.
        """
        return self._replace(gt_semantic_mask=self.gt_semantic_mask[valid_mask])

    def reorder_labels(self, indices: Int32[Tensor, " num_points"]) -> Segmentation3DGTSample:
        """
        Reorder the labels, so the mask follows a permuted point cloud.

        Args:
          indices: Permutation applied to the points.

        Returns:
          Segmentation3DGTSample: Sample holding the reordered labels.
        """
        return self._replace(gt_semantic_mask=self.gt_semantic_mask[indices])

    def append_ignored_labels(self, num_points: int) -> Segmentation3DGTSample:
        """
        Append the ignore index once per appended point, so points a sweep brings in shape the
        geometry without reaching the loss or the metrics.

        Args:
          num_points: Number of points appended to the point cloud.

        Returns:
          Segmentation3DGTSample: Sample covering the appended points as well.
        """
        appended = torch.full(
            (num_points,),
            self.ignore_index,
            dtype=self.gt_semantic_mask.dtype,
            device=self.gt_semantic_mask.device,
        )
        return self._replace(gt_semantic_mask=torch.cat([self.gt_semantic_mask, appended], dim=0))


class Segmentation3DGTBatch(NamedTuple):
    """
    Named tuple to represent the semantic labels of a batch with their batch indices. The
    labels are concatenated the same way the points are, so a label and its point share a
    position in both tensors.
    """

    # (B*P, ), semantic label of every point in the batch
    gt_semantic_masks: Int64[Tensor, " batch_size*num_points"]
    # (B*P, ), batch indices for each label
    batch_indices: Int32[Tensor, " batch_size*num_points"]

    @staticmethod
    def collate_gt_samples(
        segmentation3d_gt_samples: Sequence[Segmentation3DGTSample],
    ) -> Segmentation3DGTBatch | None:
        """
        Collate a sequence of 3D segmentation GT samples into a single Segmentation3DGTBatch.

        Args:
          segmentation3d_gt_samples: Sequence of Segmentation3DGTSample to be collated.

        Returns:
          Segmentation3DGTBatch: Collated 3D segmentation GT batch.
        """
        if len(segmentation3d_gt_samples) == 0:
            return None

        # Concatenate the masks of every sample, one label per point
        gt_semantic_masks = torch.cat(
            [sample.gt_semantic_mask for sample in segmentation3d_gt_samples], dim=0
        )

        # Convert it to (0, 0, 0, 1, 1, 1, 2, 2, 2, ...) for each label in the batch
        batch_indices = torch.cat(
            [
                torch.full(
                    (sample.gt_semantic_mask.shape[0],),
                    index,
                    dtype=torch.int32,
                    device=sample.gt_semantic_mask.device,
                )
                for index, sample in enumerate(segmentation3d_gt_samples)
            ],
            dim=0,
        )

        if gt_semantic_masks.shape[0] != batch_indices.shape[0]:
            raise ValueError(
                "Mismatch between number of semantic labels and batch indices. "
                f"Labels shape: {gt_semantic_masks.shape}, "
                f"Batch indices shape: {batch_indices.shape}"
            )

        return Segmentation3DGTBatch(
            gt_semantic_masks=gt_semantic_masks,
            batch_indices=batch_indices,
        )

    def to_device(self, device: torch.device) -> Segmentation3DGTBatch:
        """
        Move the Segmentation3DGTBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          Segmentation3DGTBatch: The batch moved to the specified device.
        """
        return Segmentation3DGTBatch(
            gt_semantic_masks=self.gt_semantic_masks.to(device),
            batch_indices=self.batch_indices.to(device),
        )
