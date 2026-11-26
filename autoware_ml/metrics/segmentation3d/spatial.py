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

"""Fixed-radius neighbourhood helpers for the point-level segmentation metrics.

The neighbourhood-tolerant error rate and the drivable-area cluster count need
spatial neighbour queries on a frame's points. These are thin, tested wrappers
over ``scipy.spatial.cKDTree`` so the metric components stay focused on their own
math.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def tolerant_error_mask(
    coord: np.ndarray, pred: np.ndarray, target: np.ndarray, radius: float
) -> np.ndarray:
    """Misclassified points that have no correct-class neighbour within ``radius``.

    A point ``p`` counts as an error only when it is wrong (``pred != target``)
    *and* no point ``q`` within ``radius`` was predicted as ``p``'s true class
    (``pred[q] == target[p]``). This forgives boundary jitter and annotation noise:
    a flipped point beside a correctly predicted one does not count. At
    ``radius <= 0`` it reduces to the plain misclassification mask. Empty input
    returns an empty mask.

    Evaluated per true class with counting queries: a wrong point of class ``c``
    is rescued exactly when at least one point *predicted* ``c`` lies within
    ``radius``, so each class is one ``query_ball_point(..., return_length=True)``
    against the tree of its candidates. No neighbour lists are ever
    materialized, keeping memory bounded on dense, badly-predicted frames.

    Args:
        coord: Point coordinates ``(N, 3)``.
        pred: Predicted labels ``(N,)``.
        target: True labels ``(N,)``.
        radius: Rescue radius in meters.

    Returns:
        Boolean error mask ``(N,)``.
    """
    n = coord.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    wrong = pred != target
    if not wrong.any() or radius <= 0.0:
        return wrong.copy()
    mask = wrong.copy()
    for true_class in np.unique(target[wrong]):
        candidates = pred == true_class
        if not candidates.any():
            continue  # nothing predicted this class anywhere, no rescue possible
        queries = np.flatnonzero(wrong & (target == true_class))
        tree = cKDTree(coord[candidates])
        counts = tree.query_ball_point(coord[queries], r=radius, return_length=True)
        mask[queries[counts > 0]] = False
    return mask


def cluster_sizes(coord: np.ndarray, radius: float) -> np.ndarray:
    """Sizes of the connected components of the radius-neighbour graph.

    Two points are linked when they lie within ``radius`` and a cluster is a
    connected component. Empty input returns an empty array.

    Args:
        coord: Point coordinates ``(N, 3)``.
        radius: Linking radius in meters.

    Returns:
        Component sizes, one entry per cluster.
    """
    n = coord.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    pairs = cKDTree(coord).query_pairs(r=radius, output_type="ndarray")
    return np.bincount(_component_labels(n, pairs)).astype(np.int64)


def cluster_counts(
    coord: np.ndarray, labels: np.ndarray, num_classes: int, radius: float, min_points: int = 1
) -> tuple[int, np.ndarray]:
    """Clusters among all the points, and per class among that class's own points.

    The per-class count links only points of the same label, as clustering each
    class's points on their own would. Both counts read one neighbour search: the
    pairs of a class are the pairs of the whole set whose two points carry that
    label.

    Args:
        coord: Point coordinates ``(N, 3)``.
        labels: Class index per point ``(N,)``, in ``[0, num_classes)``.
        num_classes: Number of classes the per-class count covers.
        radius: Linking radius in meters.
        min_points: Smallest cluster that counts.

    Returns:
        The number of clusters over all points, and ``(num_classes,)`` clusters per
        class.
    """
    n = coord.shape[0]
    if n == 0:
        return 0, np.zeros((num_classes,), dtype=np.int64)
    pairs = cKDTree(coord).query_pairs(r=radius, output_type="ndarray")
    whole_sizes = np.bincount(_component_labels(n, pairs))
    same_class = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    component = _component_labels(n, pairs[same_class])
    sizes = np.bincount(component)
    # every point of a component carries the same label, so any member names its class
    component_class = np.zeros(sizes.shape[0], dtype=np.int64)
    component_class[component] = labels
    per_class = np.bincount(component_class[sizes >= min_points], minlength=num_classes)
    return int(np.sum(whole_sizes >= min_points)), per_class.astype(np.int64)


def _component_labels(num_points: int, pairs: np.ndarray) -> np.ndarray:
    """Connected-component label of every point, given the linked index pairs.

    Args:
        num_points: Number of points, the nodes of the graph.
        pairs: Linked pairs ``(P, 2)``, every pair once.

    Returns:
        Component label per point ``(num_points,)``.
    """
    # float64 is the dtype csgraph works on, so it takes the graph as it is.
    graph = coo_matrix(
        (np.ones(pairs.shape[0], dtype=np.float64), (pairs[:, 0], pairs[:, 1])),
        shape=(num_points, num_points),
    )
    # The pairs hold no duplicate, and the traversal does not need them ordered, so
    # the conversion can skip sorting every row, which costs more than the search.
    graph.has_canonical_format = True
    _, labels = connected_components(graph.tocsr(), directed=False)
    return labels
