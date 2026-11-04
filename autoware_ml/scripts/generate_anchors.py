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

"""Generate the k-means anchors of the Sparse4D head of a task into the cache directory.

Usage: ``python -m autoware_ml.scripts.generate_anchors --config-name <task config>``, the
task config of the bundled or the external config root like the CLI names it.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.models.detection3d.heads.sparse4d import ANCHOR_DIM
from autoware_ml.types.dataset import SplitType
from autoware_ml.utils.cli.helpers import resolve_config_reference

logger = logging.getLogger(__name__)
CONFIG_NAME_PREFIX = "tasks"


def load_gt_boxes(
    records: pl.DataFrame, class_names: Sequence[str], point_cloud_range: Sequence[float]
) -> np.ndarray:
    """Load the valid in range ground truth boxes ``(N, 7)`` of some dataset records.

    Args:
        records: Dataset records holding the boxes.
        class_names: Detection classes whose boxes are kept.
        point_cloud_range: Range ``[x_min, y_min, z_min, x_max, y_max, z_max]`` of the
            sampled voxel grid, boxes centered outside it are dropped.

    Returns:
        Array of ``[cx, cy, cz, l, w, h, yaw]`` rows.

    Raises:
        ValueError: If the range is malformed.
    """
    if len(point_cloud_range) != 6:
        raise ValueError(f"point_cloud_range must have 6 values, got {list(point_cloud_range)}.")
    boxes_rows = (
        records.select(pl.col(DatasetTableSchema.BOXES_3D.name).explode())
        .drop_nulls()
        .unnest(DatasetTableSchema.BOXES_3D.name)
        .filter(
            pl.col(Box3DDatasetSchema.BOX3D_VALID.name)
            & pl.col(Box3DDatasetSchema.BOX3D_LABEL_NAME.name).is_in(list(class_names))
        )
    )
    boxes = np.asarray(
        boxes_rows[Box3DDatasetSchema.BOX3D_PARAMS.name].to_list(), dtype=np.float32
    ).reshape(-1, 10)[:, :7]
    range_min = np.asarray(point_cloud_range[:3], dtype=np.float32)
    range_max = np.asarray(point_cloud_range[3:], dtype=np.float32)
    in_range = ((boxes[:, :3] >= range_min) & (boxes[:, :3] < range_max)).all(axis=1)
    return boxes[in_range]


def _cluster_means(
    values: torch.Tensor, assignments: torch.Tensor, num_clusters: int
) -> tuple[torch.Tensor, torch.Tensor]:
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


def kmeans(
    points: torch.Tensor, num_clusters: int, iterations: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
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
        assignments = torch.cdist(points, centroids).argmin(dim=1)
        means, counts = _cluster_means(points, assignments, num_clusters)
        centroids = torch.where(counts[:, None] > 0, means, centroids)
    return centroids, assignments


def build_anchors(
    boxes: np.ndarray, num_anchors: int, iterations: int, seed: int, device: str
) -> np.ndarray:
    """Cluster box centers into anchors in the Sparse4D anchor layout.

    Args:
        boxes: Ground truth boxes ``(N, 7)`` as ``[cx, cy, cz, l, w, h, yaw]``.
        num_anchors: Number of anchors.
        iterations: Number of k-means iterations.
        seed: Seed of the initial centroid draw.
        device: Device the clustering runs on.

    Returns:
        Anchors ``(num_anchors, 10)`` as
        ``[cx, cy, cz, ln l, ln w, ln h, sin yaw, cos yaw, vx, vy]``. The size of an anchor
        is the mean log size of its cluster, or of every box for an empty cluster.

    Raises:
        ValueError: If there are fewer boxes than anchors.
    """
    if boxes.shape[0] < num_anchors:
        raise ValueError(f"{boxes.shape[0]} boxes cannot seed {num_anchors} anchors.")
    boxes_tensor = torch.from_numpy(boxes).to(device)
    centroids, assignments = kmeans(boxes_tensor[:, :3], num_anchors, iterations, seed)

    log_dims = boxes_tensor[:, 3:6].log()
    cluster_log_dims, counts = _cluster_means(log_dims, assignments, num_anchors)
    anchors = torch.zeros((num_anchors, ANCHOR_DIM), device=device)
    anchors[:, :3] = centroids
    anchors[:, 3:6] = torch.where(counts[:, None] > 0, cluster_log_dims, log_dims.mean(dim=0))
    anchors[:, 7] = 1.0  # cos yaw, identity heading
    return anchors.cpu().numpy()


def generate_anchors(cfg: DictConfig) -> None:
    """Generate the anchors of the Sparse4D head of the configured task.

    The ``anchors`` group of the config sets the sample cap, the k-means iterations and the seed.

    Args:
        cfg: Hydra configuration of the task.

    Raises:
        ValueError: If the training split has no box supervised source or no box in range.
    """
    head_cfg = cfg.model.bbox_head
    datamodule: DataModule = instantiate(cfg.datamodule)
    sources = [
        source
        for source in datamodule.split_sources(SplitType.TRAIN).values()
        if source.train_dataset.det3d_supervised
    ]
    if not sources:
        raise ValueError("The training split of the task has no box supervised source.")

    boxes = []
    for source in sources:
        database = source.database
        database.process_scenario_records()
        records = datamodule.splitter.split_by_polars_dataframe(
            dataset_records_dataframe=database.load_polars_scenario_dataframe(),
            scenarios=database.scenarios,
        )[SplitType.TRAIN]
        source_boxes = load_gt_boxes(
            records, cfg.taxonomy.detection3d.class_names, head_cfg.point_cloud_range
        )
        logger.info(f"{database.version}: {source_boxes.shape[0]:,} training boxes in range")
        boxes.append(source_boxes)
    all_boxes = np.concatenate(boxes)
    if all_boxes.shape[0] == 0:
        raise ValueError(f"No training box lies inside {list(head_cfg.point_cloud_range)}.")

    rng = np.random.default_rng(cfg.anchors.seed)
    if all_boxes.shape[0] > cfg.anchors.max_samples:
        all_boxes = all_boxes[
            rng.choice(all_boxes.shape[0], cfg.anchors.max_samples, replace=False)
        ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anchors = build_anchors(
        all_boxes, head_cfg.num_anchor, cfg.anchors.iterations, cfg.anchors.seed, device
    )

    output = Path(head_cfg.anchor_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, anchors)
    logger.info(
        f"Saved {head_cfg.num_anchor} anchors from {all_boxes.shape[0]:,} boxes to {output}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Compose the task config of either config root and generate its anchors.

    Args:
        argv: Command line arguments, ``sys.argv[1:]`` when ``None``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--config-name",
        required=True,
        help="Task config name or the path of a task config inside a config root",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args(argv)

    config_path, config_name, searchpath_overrides = resolve_config_reference(
        args.config_name, CONFIG_NAME_PREFIX
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name=config_name, overrides=[*searchpath_overrides, *args.overrides])
    logging.basicConfig(level=logging.INFO)
    generate_anchors(cfg)


if __name__ == "__main__":
    main()
