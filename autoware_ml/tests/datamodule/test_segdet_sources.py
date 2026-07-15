"""Unit tests for multi-source segdet annotation loading and supervision flags."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest

from autoware_ml.datamodule.common.sources import AnnotationSource, coerce_annotation_sources
from autoware_ml.datamodule.t4dataset.segdet import T4SegmentationDetection3DDataset


def _make_frame(token: str, *, with_instances: bool = True) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "token": token,
        "lidar_points": {"lidar_path": f"lidar/{token}.bin", "num_pts_feats": 5},
        "pts_semantic_mask_path": f"seg/{token}.bin",
        "pts_semantic_mask_categories": {"car": 0, "vegetation": 1},
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


def _build_dataset(sources: list[AnnotationSource]) -> T4SegmentationDetection3DDataset:
    return T4SegmentationDetection3DDataset(
        data_root="/data",
        ann_sources=sources,
        class_names=["car"],
        name_mapping={"car": "car"},
    )


def test_coerce_annotation_sources_single_path_keeps_full_supervision(tmp_path: Path) -> None:
    sources = coerce_annotation_sources("info/train.pkl", str(tmp_path))

    assert sources == [
        AnnotationSource(path=str(tmp_path / "info/train.pkl"), det3d=True, seg3d=True, repeat=1)
    ]


def test_coerce_annotation_sources_requires_exact_spec_keys() -> None:
    with pytest.raises(ValueError, match="missing \\['repeat'\\]"):
        coerce_annotation_sources([{"path": "a.pkl", "det3d": True, "seg3d": True}], "/data")
    with pytest.raises(ValueError, match="unknown \\['oversample'\\]"):
        coerce_annotation_sources(
            [{"path": "a.pkl", "det3d": True, "seg3d": True, "repeat": 1, "oversample": 2}],
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

    seg_info = dataset.get_data_info(2)
    assert seg_info["instances"] == []
    assert seg_info["label_to_category"] == {}
    assert seg_info["pts_semantic_mask_categories"] == {"car": 0, "vegetation": 1}


def test_dataset_empties_seg_categories_when_seg3d_disabled(tmp_path: Path) -> None:
    pkl = _write_pkl(tmp_path / "det_only_supervision.pkl", [_make_frame("a1")])
    dataset = _build_dataset([AnnotationSource(path=pkl, det3d=True, seg3d=False, repeat=1)])

    info = dataset.get_data_info(0)
    assert info["pts_semantic_mask_categories"] == {}
    assert len(info["instances"]) == 1


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
