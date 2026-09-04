"""Fixed-radius neighbourhood helpers for the point-level segmentation metrics.

The neighbourhood-tolerant error rate and the drivable-area cluster count need
spatial neighbour queries on a frame's points. These are thin, tested wrappers
over ``scipy.spatial.cKDTree`` so the metric components stay focused on their own
math.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
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
        # Single-threaded on purpose: scipy spawns a fresh thread pool per call,
        # and these per-class query batches are small enough that the spawn
        # costs far more than the query itself.
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
    tree = cKDTree(coord)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if pairs.shape[0] == 0:
        return np.ones((n,), dtype=np.int64)  # every point is its own cluster
    data = np.ones(pairs.shape[0], dtype=np.int8)
    graph = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return np.bincount(labels).astype(np.int64)
