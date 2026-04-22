"""Tests for segmentation3d datasets and datamodules."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import torch

from autoware_ml.datamodule.nuscenes.segmentation3d import NuscenesSegmentation3DDataset
from autoware_ml.datamodule.t4dataset.segmentation3d import T4Segmentation3DDataset
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.point_cloud.loading import LoadPointsFromFile
from autoware_ml.transforms.segmentation3d.loading import LoadSegAnnotations3D


def test_nuscenes_segmentation_dataset_resolves_lidar_top_samples(tmp_path: Path) -> None:
    data_root = tmp_path / "nuscenes"
    lidar_dir = data_root / "samples" / "LIDAR_TOP"
    lidarseg_dir = data_root / "lidarseg" / "v1.0-trainval"
    lidar_dir.mkdir(parents=True)
    lidarseg_dir.mkdir(parents=True)

    lidar_path = lidar_dir / "sample.bin"
    torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.0]], dtype=torch.float32).numpy().tofile(lidar_path)
    mask_path = lidarseg_dir / "sample_lidarseg.bin"
    torch.tensor([1], dtype=torch.uint8).numpy().tofile(mask_path)

    ann_file = tmp_path / "infos.pkl"
    ann_file.write_bytes(
        pickle.dumps(
            {
                "data_list": [
                    {
                        "token": "sample-token",
                        "lidar_points": {"lidar_path": "sample.bin"},
                        "pts_semantic_mask_path": "sample_lidarseg.bin",
                    }
                ]
            }
        )
    )

    dataset = NuscenesSegmentation3DDataset(
        data_root=str(data_root),
        ann_file=str(ann_file),
        lidarseg_dir=str(lidarseg_dir),
        dataset_transforms=TransformsCompose([LoadPointsFromFile(), LoadSegAnnotations3D()]),
    )
    sample = dataset[0]

    assert sample["lidar_path"] == str(lidar_path)
    assert sample["points"].shape == (1, 4)
    assert torch.equal(torch.as_tensor(sample["pts_semantic_mask"]), torch.tensor([1]))


def test_nuscenes_segmentation_dataset_returns_name_key(tmp_path: Path) -> None:
    """The unified dataset should expose 'name' for sample identification."""
    data_root = tmp_path / "nuscenes"
    (data_root / "samples" / "LIDAR_TOP").mkdir(parents=True)
    ann_file = tmp_path / "infos.pkl"
    ann_file.write_bytes(
        pickle.dumps(
            {
                "data_list": [
                    {
                        "token": "tok-123",
                        "lidar_points": {"lidar_path": "sample.bin"},
                        "pts_semantic_mask_path": "mask.bin",
                    }
                ]
            }
        )
    )

    dataset = NuscenesSegmentation3DDataset(
        data_root=str(data_root),
        ann_file=str(ann_file),
    )

    info = dataset.get_data_info(0)
    assert info["name"] == "tok-123"


def test_nuscenes_segmentation_dataset_accepts_pre_prefixed_lidar_path(tmp_path: Path) -> None:
    data_root = tmp_path / "nuscenes"
    lidar_dir = data_root / "samples" / "LIDAR_TOP"
    lidarseg_dir = data_root / "lidarseg" / "v1.0-trainval"
    lidar_dir.mkdir(parents=True)
    lidarseg_dir.mkdir(parents=True)

    lidar_path = lidar_dir / "sample.bin"
    torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.0]], dtype=torch.float32).numpy().tofile(lidar_path)
    mask_path = lidarseg_dir / "sample_lidarseg.bin"
    torch.tensor([1], dtype=torch.uint8).numpy().tofile(mask_path)

    ann_file = tmp_path / "infos.pkl"
    ann_file.write_bytes(
        pickle.dumps(
            {
                "data_list": [
                    {
                        "token": "sample-token",
                        "lidar_points": {"lidar_path": "samples/LIDAR_TOP/sample.bin"},
                        "pts_semantic_mask_path": "sample_lidarseg.bin",
                    }
                ]
            }
        )
    )

    dataset = NuscenesSegmentation3DDataset(
        data_root=str(data_root),
        ann_file=str(ann_file),
        lidarseg_dir=str(lidarseg_dir),
    )

    info = dataset.get_data_info(0)
    assert info["lidar_path"] == str(lidar_path)


def test_t4_segmentation_dataset_warns_for_empty_source(caplog, tmp_path: Path) -> None:
    data_root = tmp_path / "t4dataset"
    data_root.mkdir(parents=True)
    ann_file = tmp_path / "infos.pkl"
    ann_file.write_bytes(
        pickle.dumps(
            {
                "data_list": [
                    {
                        "lidar_points": {"lidar_path": "sample.bin", "num_pts_feats": 5},
                        "lidar_sources": {
                            "LIDAR_FRONT_UPPER": {
                                "sensor_token": "sensor-1",
                                "translation": [0.0, 0.0, 0.0],
                                "rotation": [1.0, 0.0, 0.0, 0.0],
                            }
                        },
                        "lidar_sources_info": {
                            "sources": [
                                {
                                    "sensor_token": "sensor-1",
                                    "idx_begin": 0,
                                    "length": 0,
                                }
                            ]
                        },
                        "pts_semantic_mask_categories": {"car": 0},
                        "pts_semantic_mask_path": "labels.bin",
                    }
                ]
            }
        )
    )

    dataset = T4Segmentation3DDataset(
        data_root=str(data_root),
        ann_file=str(ann_file),
        lidar_sources=["LIDAR_FRONT_UPPER"],
    )

    with caplog.at_level(logging.WARNING):
        sample = dataset.get_data_info(0)

    assert sample["length"] == 0
    assert "has no points" in caplog.text


def test_t4_segmentation_dataset_without_lidar_sources(tmp_path: Path) -> None:
    """When lidar_sources is omitted the dataset returns one item per annotation."""
    data_root = tmp_path / "t4dataset"
    data_root.mkdir(parents=True)
    ann_file = tmp_path / "infos.pkl"
    ann_file.write_bytes(
        pickle.dumps(
            {
                "data_list": [
                    {
                        "token": "tok-abc",
                        "lidar_points": {"lidar_path": "sample.bin", "num_pts_feats": 4},
                        "pts_semantic_mask_categories": {"car": 0},
                        "pts_semantic_mask_path": "labels.bin",
                    },
                    {
                        "token": "tok-def",
                        "lidar_points": {"lidar_path": "sample2.bin", "num_pts_feats": 4},
                        "pts_semantic_mask_categories": {"car": 0},
                        "pts_semantic_mask_path": "labels2.bin",
                    },
                ]
            }
        )
    )

    dataset = T4Segmentation3DDataset(
        data_root=str(data_root),
        ann_file=str(ann_file),
    )

    assert len(dataset) == 2
    info = dataset.get_data_info(0)
    assert info["name"] == "tok-abc"
    assert "idx_begin" not in info
    assert "length" not in info
