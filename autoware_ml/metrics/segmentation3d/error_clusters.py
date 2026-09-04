"""Misclassification rate and error clusters.

An error is simply a point whose predicted class disagrees with the ground
truth. The *area* that makes an error dangerous (the road) is not this metric's
concern: attach a lanelet ``RegionFilter`` and the same metric reports errors on
the road, in the corridor, or whole-scene.

Besides the point rate, nearby error points are merged into spatial clusters
and each cluster counts as one phantom regardless of size: one stray point
and a ten-point blob both cause a single emergency stop, so they carry the same
penalty.

Reported whole-scene and per class (over points whose ground truth is that
class). Run in a grouped suite the per-class values become per-group with
intra-group confusion counted as correct, e.g. error clusters on
``grouped_flat_surface`` are phantom obstacles on the drivable area.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.segmentation3d.point_cloud import (
    PointCloudSegState,
    class_token,
    valid_point_mask,
)
from autoware_ml.metrics.segmentation3d.spatial import cluster_sizes


class ErrorClusters(Metric[PointCloudSegState]):
    """Misclassified-point rate plus the count of error clusters."""

    def __init__(
        self,
        cluster_radius: float = 0.5,
        min_cluster_points: int = 1,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Store the clustering parameters.

        Args:
            cluster_radius: Neighbour distance in meters merging error points into one cluster.
            min_cluster_points: Smallest error cluster that counts.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.cluster_radius = float(cluster_radius)
        # Default 1: every error cluster counts once regardless of size. Raise it
        # only to deliberately ignore very small phantoms.
        self.min_cluster_points = int(min_cluster_points)

    def evaluate(self, state: PointCloudSegState, stage: EvalStage) -> dict[str, float]:
        """Aggregate the error rate and cluster counts, whole-scene and per class.

        Args:
            state: Point-cache state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        num_classes = state.num_classes
        error_total = valid_total = cluster_total = num_frames = 0
        per_error = [0] * num_classes
        per_valid = [0] * num_classes
        per_cluster = [0] * num_classes
        for frame in state.frames:
            valid = valid_point_mask(frame, num_classes, state.ignore_index)
            if not valid.any():
                continue
            num_frames += 1
            coord, pred, target = frame.coord[valid], frame.pred[valid], frame.target[valid]
            wrong = pred != target
            valid_total += int(valid.sum())
            error_total += int(wrong.sum())
            cluster_total += self._clusters(coord[wrong])
            for class_index in range(num_classes):
                is_class = target == class_index
                per_valid[class_index] += int(is_class.sum())
                wrong_class = wrong & is_class
                per_error[class_index] += int(wrong_class.sum())
                per_cluster[class_index] += self._clusters(coord[wrong_class])

        report = {
            "error_rate": (error_total / valid_total) if valid_total else float("nan"),
            "error_cluster_count": float(cluster_total),
        }
        if num_frames:
            report["error_clusters_per_frame"] = cluster_total / num_frames
        for class_index in range(num_classes):
            name = class_token(class_index, state.class_names)
            report[f"error_rate_{name}"] = (
                per_error[class_index] / per_valid[class_index]
                if per_valid[class_index]
                else float("nan")
            )
            report[f"error_cluster_count_{name}"] = float(per_cluster[class_index])
        return report

    def _clusters(self, coord: np.ndarray) -> int:
        sizes = cluster_sizes(coord, self.cluster_radius)
        return int(np.sum(sizes >= self.min_cluster_points))
