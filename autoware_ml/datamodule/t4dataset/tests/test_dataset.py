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

"""Tests for the image and lidar samples the T4 multi task dataset serves."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import numpy as np
import pytest
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.image_frames import ImageFrameDataModel
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.t4dataset import dataset as dataset_module
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.metrics.geometry.lanelet import LaneletMapProvider, T4LaneletMapResolver

_ROOT = "/data/t4dataset"
_SCENE = "db_j6gen2_v1/0a1b2c3d-0000-4000-8000-000000000000/0"


def camera_frame(channel: str) -> dict:
    """
    Record table row of one camera keyframe, with the calibration a projection needs.

    Args:
      channel: Camera channel name.

    Returns:
      dict: Image frame as the record table stores it.
    """
    return ImageFrameDataModel(
        image_frame_id=f"{channel}-token",
        image_keyframe=True,
        image_sensor_id=f"{channel}-sensor",
        image_sensor_channel_name=channel,
        image_timestamp_seconds=0.0,
        image_relative_path=f"{_SCENE}/data/{channel}/00000.jpg",
        image_height=1080,
        image_width=1920,
        cam2img=np.eye(3),
        image_distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
        image_distortion_model="plumb_bob",
        image_sensor_to_ego_pose_matrix=np.eye(4),
        image_frame_ego_pose_to_global_matrix=np.eye(4),
        lidar2cam=np.eye(4),
        lidar2img=np.eye(4),
    ).to_dictionary()


def build_dataset(
    image_frames: list[list[list[dict]]],
    camera_names: list[str] | None = None,
    camera_sources: list[str] | None = None,
    map_provider: object | None = None,
) -> T4Dataset:
    """
    Dataset over records that carry the given camera frames.

    Args:
      image_frames: Camera frames of every record, as the record table stores them. The
        frames of every camera at the sample time come first, then one list per sweep.
      camera_names: Cameras the dataset serves, None for every camera of a record.
      camera_sources: Cameras the dataset serves one by one.
      map_provider: Map provider of the database, None when it ships no maps.

    Returns:
      T4Dataset: Dataset serving those records, with no task datasets attached.
    """
    records = pl.DataFrame(
        {DatasetTableSchema.IMAGE_FRAMES.name: image_frames},
        schema={DatasetTableSchema.IMAGE_FRAMES.name: DatasetTableSchema.IMAGE_FRAMES.dtype},
    )
    return T4Dataset(
        lidar_intensity_scale=255.0,
        database_root_path=_ROOT,
        max_num_3d_gt_bboxes=0,
        dataset_records_dataframe=records,
        transforms=None,
        dataset_tasks=MappingProxyType({}),
        det3d_supervised=True,
        seg3d_supervised=True,
        camera_names=camera_names,
        camera_sources=camera_sources,
        map_provider=map_provider,
    )


def test_every_camera_at_the_sample_time_is_served_and_the_sweeps_are_not() -> None:
    dataset = build_dataset(
        [[[camera_frame("CAM_FRONT"), camera_frame("CAM_BACK")], [camera_frame("CAM_FRONT")]]]
    )

    samples = dataset.get_image_data_samples(0, 0)

    assert samples is not None
    assert [sample.camera_name for sample in samples] == ["CAM_FRONT", "CAM_BACK"]


def test_the_declared_cameras_are_served_in_their_order() -> None:
    cameras = ["CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT"]
    dataset = build_dataset(
        [[[camera_frame(camera) for camera in cameras]]], camera_names=["CAM_BACK", "CAM_FRONT"]
    )

    samples = dataset.get_image_data_samples(0, 0)

    assert samples is not None
    assert [sample.camera_name for sample in samples] == ["CAM_BACK", "CAM_FRONT"]


def test_records_missing_a_declared_camera_are_left_out() -> None:
    # On a corpus where every lidar frame is a record, a camera can have no frame at the
    # lidar timestamp, so the record cannot serve the model and is dropped.
    dataset = build_dataset(
        [
            [[camera_frame("CAM_FRONT")]],
            [[camera_frame("CAM_FRONT"), camera_frame("CAM_BACK")]],
            [[]],
            [],
        ],
        camera_names=["CAM_FRONT", "CAM_BACK"],
    )

    assert len(dataset) == 1
    samples = dataset.get_image_data_samples(0, 0)
    assert samples is not None
    assert [sample.camera_name for sample in samples] == ["CAM_FRONT", "CAM_BACK"]


def test_rejects_cameras_no_record_holds() -> None:
    with pytest.raises(ValueError, match="CAM_BACK"):
        build_dataset([[[camera_frame("CAM_FRONT")]]], camera_names=["CAM_BACK"])


def test_rejects_repeated_camera_names() -> None:
    with pytest.raises(ValueError, match="distinct"):
        build_dataset([[[camera_frame("CAM_FRONT")]]], camera_names=["CAM_FRONT", "CAM_FRONT"])


def test_camera_sources_serve_each_camera_as_its_own_sample() -> None:
    cameras = ["CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT"]
    records = [[[camera_frame(camera) for camera in cameras]] for _ in range(2)]
    dataset = build_dataset(records, camera_sources=["CAM_BACK", "CAM_FRONT"])

    assert len(dataset) == 4
    served = [dataset.get_image_data_samples(0, source_index) for source_index in range(2)]
    assert [[sample.camera_name for sample in samples] for samples in served] == [
        ["CAM_BACK"],
        ["CAM_FRONT"],
    ]


def test_camera_sources_leave_out_records_missing_one_of_them() -> None:
    dataset = build_dataset(
        [[[camera_frame("CAM_FRONT")]], [[camera_frame("CAM_FRONT"), camera_frame("CAM_BACK")]]],
        camera_sources=["CAM_FRONT", "CAM_BACK"],
    )

    assert len(dataset) == 2


@pytest.mark.parametrize(
    "selection",
    [
        {"camera_names": ["CAM_FRONT"], "camera_sources": ["CAM_FRONT"]},
        {"camera_sources": ["CAM_FRONT"], "lidar_sources": ["LIDAR_TOP"]},
    ],
)
def test_camera_sources_reject_other_selections(selection: dict) -> None:
    records = pl.DataFrame(
        {DatasetTableSchema.IMAGE_FRAMES.name: [[[camera_frame("CAM_FRONT")]]]},
        schema={DatasetTableSchema.IMAGE_FRAMES.name: DatasetTableSchema.IMAGE_FRAMES.dtype},
    )

    with pytest.raises(ValueError):
        T4Dataset(
            lidar_intensity_scale=255.0,
            database_root_path=_ROOT,
            max_num_3d_gt_bboxes=0,
            dataset_records_dataframe=records,
            transforms=None,
            dataset_tasks=MappingProxyType({}),
            det3d_supervised=True,
            seg3d_supervised=True,
            **selection,
        )


def test_a_record_without_a_camera_at_the_sample_time_carries_no_image() -> None:
    dataset = build_dataset([[[], [camera_frame("CAM_FRONT")]], []])

    assert dataset.get_image_data_samples(0, 0) is None
    assert dataset.get_image_data_samples(1, 0) is None


def test_lidar_sources_multiply_the_samples_of_every_record() -> None:
    records = pl.DataFrame({DatasetTableSchema.LIDAR_SOURCES.name: [None, None]})
    dataset = T4Dataset(
        lidar_intensity_scale=255.0,
        database_root_path=_ROOT,
        max_num_3d_gt_bboxes=0,
        dataset_records_dataframe=records,
        transforms=None,
        dataset_tasks=MappingProxyType({}),
        det3d_supervised=True,
        seg3d_supervised=True,
        lidar_sources=["LIDAR_FRONT_LOWER", "LIDAR_REAR_LOWER"],
    )

    assert len(dataset) == 4


def test_a_record_without_lidar_sources_cannot_serve_one() -> None:
    records = pl.DataFrame({DatasetTableSchema.LIDAR_SOURCES.name: [None]})
    dataset = T4Dataset(
        lidar_intensity_scale=255.0,
        database_root_path=_ROOT,
        max_num_3d_gt_bboxes=0,
        dataset_records_dataframe=records,
        transforms=None,
        dataset_tasks=MappingProxyType({}),
        det3d_supervised=True,
        seg3d_supervised=True,
        lidar_sources=["LIDAR_FRONT_LOWER"],
    )

    with pytest.raises(ValueError, match="lists no lidar sources"):
        dataset.get_lidar_source_view(0, "LIDAR_FRONT_LOWER")


def test_lidar_sources_reject_detection3d() -> None:
    with pytest.raises(ValueError, match="cannot run on single lidar sources"):
        T4Dataset(
            lidar_intensity_scale=255.0,
            database_root_path=_ROOT,
            max_num_3d_gt_bboxes=0,
            dataset_records_dataframe=pl.DataFrame(),
            transforms=None,
            dataset_tasks=MappingProxyType({"detection3d": lambda **kwargs: None}),
            det3d_supervised=True,
            seg3d_supervised=True,
            lidar_sources=["LIDAR_FRONT_LOWER"],
        )


class _Provider:
    """Stand-in map provider, the dataset only keeps it for the evaluation filters."""


def test_the_dataset_keeps_the_map_provider_of_its_database_through_binding() -> None:
    provider = _Provider()
    dataset = build_dataset([[[camera_frame("CAM_FRONT")]]], map_provider=provider)

    bound = dataset.bind(dataset.dataset_records_dataframe)

    assert dataset.map_provider is provider
    assert bound.map_provider is provider


def test_a_database_without_maps_leaves_the_dataset_without_a_provider() -> None:
    dataset = build_dataset([[[camera_frame("CAM_FRONT")]]])

    assert dataset.map_provider is None


def _lidar_frame(scene: str) -> dict:
    """The record table entry of a keyframe lidar frame stored under the scene directory."""
    return {
        "lidar_frame_id": "frame-0",
        "lidar_keyframe": True,
        "lidar_sensor_id": "lidar",
        "lidar_sensor_channel_name": "LIDAR_TOP",
        "lidar_timestamp_seconds": 0.0,
        "lidar_pointcloud_relative_path": f"{scene}/data/LIDAR_TOP/0.pcd.bin",
        "lidar_pointcloud_source_relative_path": None,
        "lidar_pointcloud_num_features": 5,
        "lidar_sensor_to_ego_pose_matrix": np.eye(4).tolist(),
        "lidar_frame_ego_pose_to_global_matrix": np.eye(4).tolist(),
        "lidar_sensor_to_lidar_sweep_matrix": np.eye(4).tolist(),
        "lidar_pointcloud_semantic_mask_relative_path": None,
    }


def _dataset_over(root: Path, records: pl.DataFrame, map_provider: object | None) -> T4Dataset:
    return T4Dataset(
        lidar_intensity_scale=255.0,
        database_root_path=str(root),
        max_num_3d_gt_bboxes=0,
        dataset_records_dataframe=records,
        transforms=None,
        dataset_tasks=MappingProxyType({}),
        det3d_supervised=True,
        seg3d_supervised=True,
        map_provider=map_provider,
    )


def test_frame_meta_carries_the_map_resolved_against_the_datasets_own_root(tmp_path: Path) -> None:
    """
    Input: a source over the pseudo-labelled corpus under ``t4dataset/pseudo_large`` whose
    record carries a map and a sample time, with its maps resolved against its own root, against
    the main corpus root and without a map provider.
    Expected: a source over another corpus resolves the maps of its frames against its own root;
    looked up against the main corpus root, as the metrics used to do for every source, the same
    frames have no map and drop out of the map-backed slices silently. The frame meta also
    carries the sample time of the record.
    Check: the frame meta of the own-root lookup holds the time, the scene token and the map
    path, and the other two lookups have no map.
    """
    main_root = tmp_path / "t4dataset"
    pseudo_root = main_root / "pseudo_large"
    scene = "db_pseudo/0a1b2c3d-0000-4000-8000-000000000000/0"
    map_path = pseudo_root / scene / "map" / "lanelet2_map.osm"
    map_path.parent.mkdir(parents=True)
    map_path.write_text("<osm></osm>")
    # The frame meta also reads the sample time of the record, so the table carries it.
    records = pl.DataFrame(
        {
            DatasetTableSchema.LIDAR_FRAMES.name: [[_lidar_frame(scene)]],
            DatasetTableSchema.TIMESTAMP_SECONDS.name: [12.5],
        },
        schema={
            DatasetTableSchema.LIDAR_FRAMES.name: DatasetTableSchema.LIDAR_FRAMES.dtype,
            DatasetTableSchema.TIMESTAMP_SECONDS.name: DatasetTableSchema.TIMESTAMP_SECONDS.dtype,
        },
    )

    own_corpus = _dataset_over(
        pseudo_root, records, LaneletMapProvider(T4LaneletMapResolver(str(pseudo_root)))
    )
    sample = own_corpus.get_frame_meta_sample(0, None)
    assert sample.timestamp == 12.5

    assert sample.scene_token == scene
    assert sample.lanelet_map_path == str(map_path)

    main_corpus_lookup = _dataset_over(
        pseudo_root, records, LaneletMapProvider(T4LaneletMapResolver(str(main_root)))
    )
    assert main_corpus_lookup.get_frame_meta_sample(0, None).lanelet_map_path is None
    assert (
        _dataset_over(pseudo_root, records, None).get_frame_meta_sample(0, None).lanelet_map_path
        is None
    )


class _LoggingTask(BaseDatasetTask):
    """A task recording every call to log_dataset_info."""

    calls: list[int] = []

    def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
        return dataset_records_dataframe

    def get_data_sample(self, idx: int) -> ModelGTSample:
        raise AssertionError("not read in this test")

    def log_dataset_info(self) -> None:
        _LoggingTask.calls.append(len(self.dataset_records_dataframe))


@pytest.mark.parametrize("rank,expected_calls", [(0, [1]), (1, [])])
def test_binding_summarises_the_tasks_on_rank_zero_only(
    rank: int, expected_calls: list[int]
) -> None:
    """
    Input: a dataset bound to one record on rank 0 and on rank 1.
    Expected: every rank binds the records of every stage, so only rank zero pays for and
        prints the task summaries.
    Check: the summary calls the task recorded.
    """
    _LoggingTask.calls = []
    records = pl.DataFrame(
        {DatasetTableSchema.IMAGE_FRAMES.name: [[]]},
        schema={DatasetTableSchema.IMAGE_FRAMES.name: DatasetTableSchema.IMAGE_FRAMES.dtype},
    )

    with patch.object(dataset_module.rank_zero_only, "rank", rank):
        T4Dataset(
            lidar_intensity_scale=255.0,
            database_root_path=_ROOT,
            max_num_3d_gt_bboxes=0,
            dataset_records_dataframe=records,
            transforms=None,
            dataset_tasks=MappingProxyType({"detection3d": _LoggingTask}),
            det3d_supervised=True,
            seg3d_supervised=True,
        )

    assert _LoggingTask.calls == expected_calls
