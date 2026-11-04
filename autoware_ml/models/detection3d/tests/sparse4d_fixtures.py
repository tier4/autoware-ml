"""Fixtures building small Sparse4D heads for the head and denoising tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from autoware_ml.models.detection3d.heads.sparse4d import ANCHOR_DIM, Sparse4DHead
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import POINT_CLOUD_RANGE


def make_anchor_file(tmp_path: Path, num_anchor: int = 8) -> str:
    """Write a synthetic Sparse4D anchor file covering the 8 m test range."""
    rng = np.random.default_rng(0)
    anchors = np.zeros((num_anchor, ANCHOR_DIM), dtype=np.float32)
    anchors[:, :2] = rng.uniform(0.0, 8.0, size=(num_anchor, 2))
    anchors[:, 2] = rng.uniform(-1.0, 1.0, size=num_anchor)
    anchors[:, 3:6] = np.log(rng.uniform(0.5, 4.0, size=(num_anchor, 3)))
    anchors[:, 7] = 1.0
    anchor_file = tmp_path / "sparse4d_anchors.npy"
    np.save(anchor_file, anchors)
    return str(anchor_file)


def build_sparse4d_head(
    anchor_file: str, head_type: type[Sparse4DHead] = Sparse4DHead, **overrides: object
) -> Sparse4DHead:
    """Return a small Sparse4D head matching the 8 m test geometry."""
    kwargs: dict[str, object] = {
        "num_classes": 2,
        "embed_dims": 32,
        "anchor_file": anchor_file,
        "num_anchor": 8,
        "point_cloud_range": POINT_CLOUD_RANGE,
        "grid_size": 1.0,
        "feature_strides": (1,),
        "num_decoder": 2,
        "num_heads": 4,
        "num_groups": 4,
        "feedforward_channels": 64,
        "num_learnable_pts": 2,
        "num_output": 8,
        "min_observed_points": 1,
        "track_birth_score": 0.0,
    }
    kwargs.update(overrides)
    return head_type(**kwargs)


def random_stage(
    grid_dims: tuple[int, int, int], num_voxels: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One sample of random voxels inside a stage grid, with features that track gradients."""
    generator = torch.Generator().manual_seed(num_voxels)
    grid_coord = torch.stack(
        [torch.randint(0, size, (num_voxels,), generator=generator) for size in grid_dims], dim=1
    )
    feat = torch.randn((num_voxels, 32), generator=generator).requires_grad_(True)
    return feat, grid_coord, torch.tensor([num_voxels])
