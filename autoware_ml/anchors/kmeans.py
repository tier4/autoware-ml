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


"""Lloyd k-means of the anchor recipes."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Float32, Int64
import torch


def cluster_means(
    values: Float[torch.Tensor, "num_points dims"],
    assignments: Int64[torch.Tensor, " num_points"],
    num_clusters: int,
) -> tuple[Float[torch.Tensor, "num_clusters dims"], Int64[torch.Tensor, " num_clusters"]]:
    """Average the values of every cluster.

    Args:
        values: Values ``(N, D)``.
        assignments: Cluster of every value ``(N,)``.
        num_clusters: Number of clusters.

    Returns:
        Tuple of the cluster means ``(num_clusters, D)``, zero for empty clusters, and the
        cluster sizes ``(num_clusters,)``.
    """
    sums = values.new_zeros((num_clusters, values.shape[1])).index_add_(0, assignments, values)
    counts = torch.bincount(assignments, minlength=num_clusters)
    return sums / counts.clamp(min=1)[:, None].to(values.dtype), counts


# Points per distance matrix of one k-means assignment
_CHUNK = 16384


def kmeans(
    points: Float[torch.Tensor, "num_points dims"], num_clusters: int, iterations: int, seed: int
) -> tuple[Float[torch.Tensor, "num_clusters dims"], Int64[torch.Tensor, " num_points"]]:
    """Run Lloyd k-means and return the centroids and the point assignments.

    An empty cluster keeps its previous centroid.

    Args:
        points: Input points ``(N, D)``.
        num_clusters: Number of centroids.
        iterations: Number of Lloyd iterations.
        seed: Seed of the initial centroid draw.

    Returns:
        Tuple of centroids ``(num_clusters, D)`` and assignments ``(N,)``.
    """
    generator = torch.Generator(device=points.device).manual_seed(seed)
    initial = torch.randperm(points.shape[0], generator=generator, device=points.device)
    centroids = points[initial[:num_clusters]].clone()
    assignments = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    for _ in range(iterations):
        # Chunks of points bound the distance matrix, the assignments are the same
        assignments = torch.cat(
            [torch.cdist(chunk, centroids).argmin(dim=1) for chunk in points.split(_CHUNK)]
        )
        means, counts = cluster_means(points, assignments, num_clusters)
        centroids = torch.where(counts[:, None] > 0, means, centroids)
    return centroids, assignments


def cluster_modes(
    samples: list[Float[np.ndarray, "..."]],
    num_modes: int,
    iterations: int,
    seed: int,
    name: str,
) -> Float32[np.ndarray, "num_modes ..."]:
    """Cluster arrays of one shape into modes of the same shape.

    Args:
        samples: Trajectories of one shape.
        num_modes: Number of modes.
        iterations: Number of k-means iterations.
        seed: Seed of the initial centroid draw.
        name: What the arrays describe, for the error message.

    Returns:
        Modes ``(num_modes, *shape)``.

    Raises:
        ValueError: If there are fewer arrays than modes.
    """
    if len(samples) < num_modes:
        raise ValueError(f"{name} has {len(samples)} samples, {num_modes} modes need more.")
    points = torch.from_numpy(np.stack(samples).reshape(len(samples), -1)).float()
    centroids, _ = kmeans(points, num_modes, iterations, seed)
    return centroids.reshape(num_modes, *samples[0].shape).numpy()
