# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""K-means anchors of the Sparse4D head from the training boxes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float32
import polars as pl
import torch

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.anchors.kmeans import cluster_means, kmeans
from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.datamodule.common.detection3d import flatten_box_fields
from autoware_ml.datamodule.training_split import TrainingSplit
from autoware_ml.geometry.bbox_3d.physical import physical_shape_mask
from autoware_ml.types.geometry import ANCHOR_DIM, AnchorFieldIndex, Box3DFieldIndex
from autoware_ml.types.tasks import TaskType


def load_gt_boxes(
    records: pl.DataFrame, taxonomy: LabelTaxonomy, point_cloud_range: Sequence[float]
) -> Float32[np.ndarray, "num_boxes 7"]:
    """Load the valid, physical and in range ground truth boxes ``(N, 7)`` of some records.

    Args:
        records: Dataset records holding the boxes.
        taxonomy: Detection taxonomy, boxes of a fine label it trains are kept.
        point_cloud_range: Range ``[x_min, y_min, z_min, x_max, y_max, z_max]`` of the
            sampled voxel grid, boxes centered outside it are dropped.

    Returns:
        Array of ``[cx, cy, cz, l, w, h, yaw]`` rows.
    """
    trained = [fine for fine, name in taxonomy.class_mapping.items() if name is not None]
    valid = Box3DDatasetSchema.BOX3D_VALID.name
    label = Box3DDatasetSchema.BOX3D_LABEL_NAME.name
    params = Box3DDatasetSchema.BOX3D_PARAMS.name
    _, box_fields = flatten_box_fields(records, [valid, label, params])
    trained_boxes = box_fields.filter(pl.col(valid) & pl.col(label).is_in(trained))
    boxes = trained_boxes[params].to_numpy().astype(np.float32, copy=False)
    boxes = boxes.reshape(-1, len(Box3DFieldIndex))[:, : Box3DFieldIndex.YAW + 1]
    # The anchors take the log of the sizes, the shape rule of the box filters applies
    centers = boxes[:, Box3DFieldIndex.X : Box3DFieldIndex.Z + 1]
    physical = physical_shape_mask(torch.from_numpy(boxes)).numpy()
    range_min = np.asarray(point_cloud_range[:3], dtype=np.float32)
    range_max = np.asarray(point_cloud_range[3:], dtype=np.float32)
    in_range = ((centers >= range_min) & (centers < range_max)).all(axis=1)
    return boxes[physical & in_range]


def cluster_boxes(
    boxes: Float32[np.ndarray, "num_boxes 7"], num_anchors: int, iterations: int, seed: int
) -> Float32[np.ndarray, "num_anchors anchor_dim"]:
    """Cluster box centers into anchors in the Sparse4D anchor layout, on the CPU.

    Args:
        boxes: Ground truth boxes ``(N, 7)`` as ``[cx, cy, cz, l, w, h, yaw]``.
        num_anchors: Number of anchors.
        iterations: Number of k-means iterations.
        seed: Seed of the initial centroid draw.

    Returns:
        Anchors ``(num_anchors, 10)`` as
        ``[cx, cy, cz, ln l, ln w, ln h, sin yaw, cos yaw, vx, vy]``. The size of an anchor
        is the mean log size of its cluster, or of every box for an empty cluster.

    Raises:
        ValueError: If there are fewer boxes than anchors.
    """
    if boxes.shape[0] < num_anchors:
        raise ValueError(f"{boxes.shape[0]} boxes cannot seed {num_anchors} anchors.")
    boxes_tensor = torch.from_numpy(boxes)
    centers = boxes_tensor[:, Box3DFieldIndex.X : Box3DFieldIndex.Z + 1]
    centroids, assignments = kmeans(centers, num_anchors, iterations, seed)

    log_dims = boxes_tensor[:, Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT + 1].log()
    cluster_log_dims, counts = cluster_means(log_dims, assignments, num_anchors)
    anchors = torch.zeros((num_anchors, ANCHOR_DIM))
    anchors[:, AnchorFieldIndex.X : AnchorFieldIndex.Z + 1] = centroids
    anchors[:, AnchorFieldIndex.LOG_LENGTH : AnchorFieldIndex.LOG_HEIGHT + 1] = torch.where(
        counts[:, None] > 0, cluster_log_dims, log_dims.mean(dim=0)
    )
    # Identity heading
    anchors[:, AnchorFieldIndex.COS_YAW] = 1.0
    return anchors.numpy()


class BoxAnchorRecipe(AnchorRecipe):
    """K-means anchors of the box centers in range, with the mean log size of every cluster."""

    name = "sparse4d_boxes"
    task_types = (TaskType.DETECTION3D,)

    def __init__(
        self,
        *,
        cache_path: str,
        taxonomy: LabelTaxonomy,
        num_anchors: int,
        point_cloud_range: Sequence[float],
        max_samples: int,
        iterations: int,
        seed: int,
    ) -> None:
        """Initialize the recipe.

        Args:
            cache_path: Directory of the cached anchors.
            taxonomy: Detection taxonomy, boxes of a fine label it trains seed the anchors.
            num_anchors: Number of anchors.
            point_cloud_range: Range of the sampled voxel grid, boxes centered outside it are
                dropped.
            max_samples: Most boxes the clustering reads, a seeded draw of them.
            iterations: Number of k-means iterations.
            seed: Seed of the box draw and of the initial centroids.

        Raises:
            ValueError: If the range is malformed.
        """
        if len(point_cloud_range) != 6:
            raise ValueError(
                f"point_cloud_range must have 6 values, got {list(point_cloud_range)}."
            )
        super().__init__(cache_path)
        self.taxonomy = taxonomy
        self.num_anchors = num_anchors
        self.point_cloud_range = [float(value) for value in point_cloud_range]
        self.max_samples = max_samples
        self.iterations = iterations
        self.seed = seed

    def hash_repr(self) -> str:
        """Canonical text of the recipe settings."""
        return (
            f"{type(self).__name__}(taxonomy={self.taxonomy}, num_anchors={self.num_anchors}, "
            f"point_cloud_range={self.point_cloud_range}, max_samples={self.max_samples}, "
            f"iterations={self.iterations}, seed={self.seed})"
        )

    def build(self, split: TrainingSplit) -> dict[str, Float32[np.ndarray, "..."]]:
        """Cluster the training boxes in range into anchors.

        Args:
            split: Training sources and their records.

        Returns:
            ``anchors`` ``(num_anchors, 10)``.

        Raises:
            ValueError: If no training box lies in range.
        """
        # Only the datasets whose boxes the model learns from seed the anchors
        boxes = np.concatenate(
            [
                load_gt_boxes(
                    training.dataset.dataset_records_dataframe,
                    self.taxonomy,
                    self.point_cloud_range,
                )
                for training in split.datasets(self.task_types)
                if training.dataset.det3d_supervised
            ]
        )
        if boxes.shape[0] == 0:
            raise ValueError(f"No training box lies inside {self.point_cloud_range}.")
        rng = np.random.default_rng(self.seed)
        if boxes.shape[0] > self.max_samples:
            boxes = boxes[rng.choice(boxes.shape[0], self.max_samples, replace=False)]
        anchors = cluster_boxes(boxes, self.num_anchors, self.iterations, self.seed)
        return {"anchors": anchors}
