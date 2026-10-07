"""Unit tests for multi-source segdet annotation loading and supervision flags."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoware_ml.datamodule.common.sources import (
    AnnotationSource,
    coerce_annotation_sources,
)
from autoware_ml.datamodule.t4dataset.segdet import T4SegmentationDetection3DDataset


def _make_frame(token: str, *, with_instances: bool = True) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "token": token,
        "lidar_points": {
            "lidar_path": f"db/uuid/0/{token}.bin",
            "num_pts_feats": 5,
            "lidar2ego": np.eye(4),
        },
        "pts_semantic_mask_path": f"seg/{token}.bin",
        "pts_semantic_mask_categories": {"car": 0, "vegetation": 1},
        "ego2global": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "timestamp": 12.3,
        "scene_token": "scene-1",
        "images": {
            "CAM_FRONT": {
                "img_path": f"images/{token}.jpg",
                "cam2img": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "lidar2cam": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
    }
    if with_instances:
        frame["instances"] = [
            {
                "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                "velocity": [0.0, 0.0],
                "bbox_label_3d": 0,
                "gt_nusc_name": "car",
                "num_lidar_pts": 10,
            }
        ]
    return frame


def _write_pkl(path: Path, frames: list[dict[str, Any]], *, with_metainfo: bool = True) -> str:
    payload: dict[str, Any] = {"data_list": frames}
    if with_metainfo:
        payload["metainfo"] = {"classes": ["car"]}
    with open(path, "wb") as file:
        pickle.dump(payload, file)
    return str(path)


def _build_dataset(
    sources: list[AnnotationSource],
    *,
    data_root: str = "/data",
) -> T4SegmentationDetection3DDataset:
    return T4SegmentationDetection3DDataset(
        data_root=data_root,
        ann_sources=sources,
        class_names=["car"],
        name_mapping={"car": "car"},
    )


def _build_numbered_intermediate_dataset(
    tmp_path: Path,
    *,
    lidar_frames: set[int],
    camera_frames: set[int],
) -> T4SegmentationDetection3DDataset:
    """Build adjacent 1 Hz keyframes backed by selected 10 Hz source files."""
    first = _make_frame("keyframe-0")
    following = _make_frame("keyframe-1")
    first["timestamp"] = 100.0
    following["timestamp"] = 101.0
    first["lidar_points"]["lidar_path"] = "scene/data/LIDAR_CONCAT/00000.pcd.bin"
    following["lidar_points"]["lidar_path"] = "scene/data/LIDAR_CONCAT/00010.pcd.bin"
    first["images"]["CAM_FRONT"]["img_path"] = "scene/data/CAM_FRONT/00000.jpg"
    following["images"]["CAM_FRONT"]["img_path"] = "scene/data/CAM_FRONT/00010.jpg"

    lidar_dir = tmp_path / "scene/data/LIDAR_CONCAT"
    camera_dir = tmp_path / "scene/data/CAM_FRONT"
    lidar_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)
    for frame_number in lidar_frames:
        (lidar_dir / f"{frame_number:05d}.pcd.bin").touch()
    for frame_number in camera_frames:
        (camera_dir / f"{frame_number:05d}.jpg").touch()

    annotation_path = _write_pkl(tmp_path / "frames.pkl", [first, following])
    return _build_dataset(
        [AnnotationSource(path=annotation_path, det3d=True, seg3d=True, repeat=1)],
        data_root=str(tmp_path),
    )


def test_coerce_annotation_sources_single_path_keeps_full_supervision(
    tmp_path: Path,
) -> None:
    sources = coerce_annotation_sources("info/train.pkl", str(tmp_path))

    assert sources == [
        AnnotationSource(path=str(tmp_path / "info/train.pkl"), det3d=True, seg3d=True, repeat=1)
    ]


def test_coerce_annotation_sources_requires_exact_spec_keys() -> None:
    with pytest.raises(ValueError, match="missing \\['repeat'\\]"):
        coerce_annotation_sources([{"path": "a.pkl", "det3d": True, "seg3d": True}], "/data")
    with pytest.raises(ValueError, match="unknown \\['oversample'\\]"):
        coerce_annotation_sources(
            [
                {
                    "path": "a.pkl",
                    "det3d": True,
                    "seg3d": True,
                    "repeat": 1,
                    "oversample": 2,
                }
            ],
            "/data",
        )
    with pytest.raises(ValueError, match="repeat must be >= 1"):
        coerce_annotation_sources(
            [{"path": "a.pkl", "det3d": True, "seg3d": True, "repeat": 0}], "/data"
        )
    with pytest.raises(ValueError, match="at least one entry"):
        coerce_annotation_sources([], "/data")
    with pytest.raises(TypeError, match="path or a sequence"):
        coerce_annotation_sources(42, "/data")


def test_dataset_mixes_sources_with_flags_and_repeat(tmp_path: Path) -> None:
    det_seg_pkl = _write_pkl(tmp_path / "det_seg.pkl", [_make_frame("a1"), _make_frame("a2")])
    seg_only_pkl = _write_pkl(
        tmp_path / "seg_only.pkl",
        [_make_frame("b1", with_instances=False)],
        with_metainfo=False,
    )
    dataset = _build_dataset(
        [
            AnnotationSource(path=det_seg_pkl, det3d=True, seg3d=True, repeat=1),
            AnnotationSource(path=seg_only_pkl, det3d=False, seg3d=True, repeat=3),
        ]
    )

    assert len(dataset) == 5
    # Detection supervision is carried by the instances themselves: seg-only
    # source frames have them dropped.
    instance_counts = [
        len(dataset.get_data_info(index)["instances"]) for index in range(len(dataset))
    ]
    assert instance_counts == [1, 1, 0, 0, 0]

    det_info = dataset.get_data_info(0)
    assert len(det_info["instances"]) == 1
    assert det_info["pts_semantic_mask_categories"] == {"car": 0, "vegetation": 1}
    assert det_info["label_to_category"] == {0: "car"}
    assert det_info["has_detection_ground_truth"] is True
    assert det_info["has_segmentation_ground_truth"] is True
    assert det_info["timestamp"] == 12.3
    assert det_info["scene_token"] == "db/uuid/0"
    assert det_info["images"]["CAM_FRONT"]["img_path"] == "/data/images/a1.jpg"

    seg_info = dataset.get_data_info(2)
    assert seg_info["instances"] == []
    assert seg_info["label_to_category"] == {}
    assert seg_info["pts_semantic_mask_categories"] == {"car": 0, "vegetation": 1}
    assert seg_info["has_detection_ground_truth"] is False
    assert seg_info["has_segmentation_ground_truth"] is True


def test_dataset_empties_seg_categories_when_seg3d_disabled(tmp_path: Path) -> None:
    pkl = _write_pkl(tmp_path / "det_only_supervision.pkl", [_make_frame("a1")])
    dataset = _build_dataset([AnnotationSource(path=pkl, det3d=True, seg3d=False, repeat=1)])

    info = dataset.get_data_info(0)
    assert info["pts_semantic_mask_categories"] == {}
    assert len(info["instances"]) == 1
    assert info["has_detection_ground_truth"] is True
    assert info["has_segmentation_ground_truth"] is False


def test_dataset_rejects_det_source_without_instances(tmp_path: Path) -> None:
    pkl = _write_pkl(tmp_path / "broken.pkl", [_make_frame("a1", with_instances=False)])

    with pytest.raises(ValueError, match="declares det3d supervision"):
        _build_dataset([AnnotationSource(path=pkl, det3d=True, seg3d=True, repeat=1)])


def test_dataset_rejects_frames_without_mask_path(tmp_path: Path) -> None:
    frame = _make_frame("a1")
    del frame["pts_semantic_mask_path"]
    pkl = _write_pkl(tmp_path / "no_mask.pkl", [frame])

    with pytest.raises(ValueError, match="pts_semantic_mask_path"):
        _build_dataset([AnnotationSource(path=pkl, det3d=True, seg3d=True, repeat=1)])


def test_dataset_builds_unlabeled_10hz_frames_and_prunes_sweeps(tmp_path: Path) -> None:
    first = _make_frame("keyframe-0")
    following = _make_frame("keyframe-1")
    first["timestamp"] = 100.0
    following["timestamp"] = 101.0
    first["lidar_points"]["lidar_path"] = "scene/data/LIDAR_CONCAT/00000.pcd.bin"
    following["lidar_points"]["lidar_path"] = "scene/data/LIDAR_CONCAT/00010.pcd.bin"
    first["images"]["CAM_FRONT"]["img_path"] = "scene/data/CAM_FRONT/00000.jpg"
    following["images"]["CAM_FRONT"]["img_path"] = "scene/data/CAM_FRONT/00010.jpg"
    first["images"]["CAM_FRONT"]["timestamp"] = 100.02
    following["images"]["CAM_FRONT"]["timestamp"] = 101.02
    first["sweeps"] = [
        {
            "lidar_path": "scene/data/LIDAR_CONCAT/previous.pcd.bin",
            "timestamp": 99.9,
        }
    ]

    lidar_dir = tmp_path / "scene/data/LIDAR_CONCAT"
    camera_dir = tmp_path / "scene/data/CAM_FRONT"
    lidar_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)
    for frame_number in range(11):
        (lidar_dir / f"{frame_number:05d}.pcd.bin").touch()
        (camera_dir / f"{frame_number:05d}.jpg").touch()

    annotation_path = _write_pkl(tmp_path / "frames.pkl", [first, following])
    dataset = _build_dataset(
        [AnnotationSource(path=annotation_path, det3d=True, seg3d=True, repeat=1)],
        data_root=str(tmp_path),
    )

    intermediate = dataset.get_intermediate_prediction_infos(0, 10.0)

    assert len(intermediate) == 9
    assert intermediate[0]["lidar_path"] == str(lidar_dir / "00001.pcd.bin")
    assert intermediate[-1]["lidar_path"] == str(lidar_dir / "00009.pcd.bin")
    assert intermediate[0]["timestamp"] == pytest.approx(100.1)
    assert intermediate[0]["images"]["CAM_FRONT"]["img_path"] == str(camera_dir / "00001.jpg")
    assert intermediate[0]["images"]["CAM_FRONT"]["timestamp"] == pytest.approx(100.12)
    assert all(frame["sweeps"] == [] for frame in intermediate)
    assert all(frame["has_detection_ground_truth"] is False for frame in intermediate)
    assert all(frame["has_segmentation_ground_truth"] is False for frame in intermediate)
    assert all("pts_semantic_mask_path" not in frame for frame in intermediate)


def test_intermediate_reconstruction_raises_on_missing_lidar_frame(
    tmp_path: Path,
) -> None:
    dataset = _build_numbered_intermediate_dataset(
        tmp_path,
        lidar_frames=set(range(11)) - {1},
        camera_frames=set(range(11)),
    )

    with pytest.raises(FileNotFoundError, match="LiDAR.*offset 1.*00001"):
        dataset.get_intermediate_prediction_infos(0, 10.0)


def test_intermediate_reconstruction_raises_on_missing_camera_frame(
    tmp_path: Path,
) -> None:
    dataset = _build_numbered_intermediate_dataset(
        tmp_path,
        lidar_frames=set(range(11)),
        camera_frames=set(range(11)) - {1},
    )

    with pytest.raises(FileNotFoundError, match="CAM_FRONT.*offset 1.*00001"):
        dataset.get_intermediate_prediction_infos(0, 10.0)


def test_intermediate_reconstruction_requires_matching_camera_sets(
    tmp_path: Path,
) -> None:
    first = _make_frame("00000")
    following = _make_frame("00010")
    first["timestamp"] = 100.0
    following["timestamp"] = 101.0
    following["images"] = {}
    annotation_path = _write_pkl(tmp_path / "frames.pkl", [first, following])
    dataset = _build_dataset(
        [AnnotationSource(path=annotation_path, det3d=True, seg3d=True, repeat=1)],
        data_root=str(tmp_path),
    )

    with pytest.raises(ValueError, match="Camera sets differ"):
        dataset.get_intermediate_prediction_infos(0, 10.0)


def test_dataset_rejects_camera_metadata_without_an_image_path(tmp_path: Path) -> None:
    frame = _make_frame("broken-camera")
    del frame["images"]["CAM_FRONT"]["img_path"]
    annotation_path = _write_pkl(tmp_path / "frames.pkl", [frame])
    dataset = _build_dataset(
        [AnnotationSource(path=annotation_path, det3d=True, seg3d=True, repeat=1)],
        data_root=str(tmp_path),
    )

    with pytest.raises(ValueError, match="CAM_FRONT.*missing img_path"):
        dataset.get_data_info(0)
