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
    """Misclassified-point rate plus the count of error clusters.

    The linking radius is fixed, so the cluster count is range dependent: the same
    phantom splits into more clusters the further out it sits, where the points that
    form it are further apart. Compare counts within a range bucket, not across them.
    """

    def __init__(
        self,
        cluster_radius: float = 0.5,
        min_cluster_points: int = 1,
        stages: tuple[str, ...] | list[str] = ("test",),
        filter=None,
    ) -> None:
        """Validate the clustering parameters.

        Args:
            cluster_radius: Neighbour distance in meters merging error points into one
                cluster. Fixed, so see the class docstring on comparing across ranges.
            min_cluster_points: Smallest error cluster that counts.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.cluster_radius = float(cluster_radius)
        # Default 1: every error cluster counts once regardless of size. Raise it
        # only to deliberately ignore very small phantoms.
        self.min_cluster_points = int(min_cluster_points)
        if self.cluster_radius <= 0.0:
            raise ValueError("cluster_radius must be > 0.")
        if self.min_cluster_points < 1:
            raise ValueError("min_cluster_points must be >= 1.")

    def evaluate(self, state: PointCloudSegState, stage: EvalStage) -> dict[str, float]:
        """Aggregate the error rate and cluster counts, whole-scene and per class.

        Args:
            state: Point-cache state for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """
        # Scored points are the ones the filter and the ignore index kept, a point is
        # wrong when its predicted class differs from its target.
        num_classes = state.num_classes
        wrong_points = scored_points = clusters = frames_seen = 0
        wrong_per_class = [0] * num_classes
        scored_per_class = [0] * num_classes
        clusters_per_class = [0] * num_classes
        for frame in state.frames:
            valid = valid_point_mask(frame, num_classes, state.ignore_index)
            if not valid.any():
                continue
            frames_seen += 1
            coord, pred, target = frame.coord[valid], frame.pred[valid], frame.target[valid]
            wrong = pred != target
            scored_points += int(valid.sum())
            wrong_points += int(wrong.sum())
            clusters += self._clusters(coord[wrong])
            for class_index in range(num_classes):
                is_class = target == class_index
                scored_per_class[class_index] += int(is_class.sum())
                wrong_class = wrong & is_class
                wrong_per_class[class_index] += int(wrong_class.sum())
                clusters_per_class[class_index] += self._clusters(coord[wrong_class])

        report = {
            "error_rate": (wrong_points / scored_points) if scored_points else float("nan"),
            "error_cluster_count": float(clusters),
            "error_clusters_per_frame": clusters / frames_seen if frames_seen else float("nan"),
        }
        for class_index in range(num_classes):
            name = class_token(class_index, state.class_names)
            report[f"error_rate_{name}"] = (
                wrong_per_class[class_index] / scored_per_class[class_index]
                if scored_per_class[class_index]
                else float("nan")
            )
            report[f"error_cluster_count_{name}"] = float(clusters_per_class[class_index])
        return report

    def _clusters(self, coord: np.ndarray) -> int:
        sizes = cluster_sizes(coord, self.cluster_radius)
        return int(np.sum(sizes >= self.min_cluster_points))
