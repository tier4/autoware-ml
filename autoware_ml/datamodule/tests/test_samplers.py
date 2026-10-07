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

"""Tests for the repeat factor sampling of the training frames."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import polars as pl
import pytest

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.base_dataset import ConcatDataset
from autoware_ml.datamodule.samplers import (
    LOW_PEDESTRIAN_CATEGORY,
    DistributedWeightedRandomSampler,
    FrameSamplingConfig,
    compute_frame_sampling_weights,
)
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.types.geometry import Box3DFieldIndex

_ROOT = "/data/t4dataset"
_CLASSES = ["car", "pedestrian", "bicycle"]
_RULES = [["bicycle", "vehicle_state.parked"]]
_IGNORE_INDEX = -1


def config(**overrides) -> FrameSamplingConfig:
    """Repeat factor settings of the tests, with fields overridden."""
    settings = dict(
        repeat_sampling_factor=0.5,
        object_bev_range=[-100.0, -100.0, 100.0, 100.0],
        pedestrian_class_name="pedestrian",
        low_pedestrian_height_threshold=1.5,
        low_pedestrian_bev_range=[-50.0, -50.0, 50.0, 50.0],
        class_names=_CLASSES,
        ignore_index=_IGNORE_INDEX,
        filter_attributes=_RULES,
        seed=0,
    )
    settings.update(overrides)
    return FrameSamplingConfig(**settings)


def box(
    label_name: str,
    x: float = 0.0,
    height: float = 1.8,
    points: int = 8,
    valid: bool = True,
    attributes: list[str] | None = None,
) -> dict:
    """Record table entry of one box."""
    params = np.zeros(len(Box3DFieldIndex), dtype=np.float64)
    params[Box3DFieldIndex.X] = x
    params[Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT] = 1.0
    params[Box3DFieldIndex.HEIGHT] = height
    return {
        "box3d_params": params.tolist(),
        "box3d_instance_id": f"{label_name}-{x}",
        "box3d_dataset_label_name": label_name,
        "box3d_label_name": label_name,
        "box3d_label_index": (
            _CLASSES.index(label_name) if label_name in _CLASSES else _IGNORE_INDEX
        ),
        "box3d_num_lidar_points": points,
        "box3d_num_radar_points": 0,
        "box3d_valid": valid,
        "box3d_attributes": attributes or [],
        "box3d_coordinate": "gravity_center",
    }


def dataset(
    records: list[list[dict]],
    det3d_supervised: bool = True,
    lidar_sources: list[str] | None = None,
) -> T4Dataset:
    """Dataset over records carrying the given boxes, with no task datasets attached."""
    schema = DatasetTableSchema.to_polars_schema()
    frame = pl.DataFrame(
        {DatasetTableSchema.BOXES_3D.name: records},
        schema={DatasetTableSchema.BOXES_3D.name: schema[DatasetTableSchema.BOXES_3D.name]},
    )
    return T4Dataset(
        lidar_intensity_scale=255.0,
        database_root_path=_ROOT,
        max_num_3d_gt_bboxes=8,
        dataset_records_dataframe=frame,
        transforms=None,
        dataset_tasks=MappingProxyType({}),
        det3d_supervised=det3d_supervised,
        seg3d_supervised=False,
        lidar_sources=lidar_sources,
    )


def test_frames_with_a_rare_class_weigh_more() -> None:
    # Cars are in every frame, the bicycle in one of four, so only that frame is lifted.
    split = ConcatDataset(
        [dataset([[box("car")], [box("car")], [box("car")], [box("car"), box("bicycle")]])], [1]
    )

    weights = compute_frame_sampling_weights(split, config())

    assert weights[:3] == [1.0, 1.0, 1.0]
    assert weights[3] > 1.0


def test_a_frame_weighs_as_much_as_its_rarest_category() -> None:
    split = ConcatDataset(
        [dataset([[box("car"), box("bicycle")], [box("bicycle")], [box("car")], [box("car")]])],
        [1],
    )

    weights = compute_frame_sampling_weights(split, config())

    # Both bicycle frames get the bicycle factor, the car beside one of them changes nothing.
    assert weights[0] == weights[1] > 1.0
    assert weights[2:] == [1.0, 1.0]


def test_a_box_counts_as_the_class_its_label_index_names() -> None:
    # A police car keeps its annotated name and trains as a car, so a frame of police cars
    # weighs like a frame of cars and not like a frame without a counted box.
    police_car = {**box("car"), "box3d_label_name": "police_car"}
    split = ConcatDataset(
        [dataset([[police_car], [box("car")], [box("car")], [box("bicycle")]])], [1]
    )

    weights = compute_frame_sampling_weights(split, config())

    assert weights[0] == weights[1] == weights[2] == 1.0
    assert weights[3] > 1.0


def test_boxes_that_are_not_targets_do_not_count() -> None:
    frames = [
        [box("car")],
        [box("car")],
        [box("car"), box("bicycle", attributes=["vehicle_state.parked"])],
        [box("car"), box("bicycle", points=0)],
        [box("car"), box("bicycle", valid=False)],
        [box("car"), box("bicycle", x=150.0)],
        [box("car"), box("animal")],
    ]

    weights = compute_frame_sampling_weights(ConcatDataset([dataset(frames)], [1]), config())

    assert weights == [1.0] * len(frames)


def test_a_short_pedestrian_close_by_is_its_own_category() -> None:
    frames = [[box("pedestrian")] for _ in range(3)] + [[box("pedestrian", height=1.0, x=10.0)]]

    weights = compute_frame_sampling_weights(ConcatDataset([dataset(frames)], [1]), config())

    assert weights[:3] == [1.0, 1.0, 1.0]
    assert weights[3] > 1.0


def test_a_short_pedestrian_far_away_is_a_pedestrian() -> None:
    frames = [[box("pedestrian")] for _ in range(3)] + [[box("pedestrian", height=1.0, x=80.0)]]

    weights = compute_frame_sampling_weights(ConcatDataset([dataset(frames)], [1]), config())

    assert weights == [1.0] * 4


def test_an_unsupervised_source_weighs_one_and_repeats_expand_the_weights() -> None:
    supervised = dataset([[box("car")], [box("car"), box("bicycle")]])
    seg3d_only = dataset([[box("bicycle")]], det3d_supervised=False)
    split = ConcatDataset([supervised, seg3d_only], [1, 2])

    weights = compute_frame_sampling_weights(split, config())

    assert len(weights) == 4
    assert weights[0] == 1.0 and weights[1] > 1.0
    assert weights[2:] == [1.0, 1.0]


def test_every_sample_of_a_record_carries_the_categories_of_its_record() -> None:
    records = [[box("car")], [box("car"), box("bicycle")]]
    split = ConcatDataset([dataset(records, lidar_sources=["LIDAR_LEFT", "LIDAR_RIGHT"])], [1])

    weights = compute_frame_sampling_weights(split, config())

    # Two samples per record, the record order kept
    assert len(weights) == 4
    assert weights[0] == weights[1] == 1.0
    assert weights[2] == weights[3] > 1.0


def test_an_unsupervised_source_leaves_the_factors_unchanged() -> None:
    records = [[box("car")], [box("car"), box("bicycle")]]
    alone = ConcatDataset([dataset(records)], [1])
    mixed = ConcatDataset(
        [dataset(records), dataset([[box("bicycle")]] * 4, det3d_supervised=False)], [1, 6]
    )

    assert compute_frame_sampling_weights(mixed, config())[:2] == (
        compute_frame_sampling_weights(alone, config())
    )


def test_a_repeated_source_weighs_like_its_records_listed_that_many_times() -> None:
    records = [[box("car")], [box("car")], [box("car"), box("bicycle")]]
    single = compute_frame_sampling_weights(ConcatDataset([dataset(records)], [1]), config())
    repeated = compute_frame_sampling_weights(ConcatDataset([dataset(records)], [3]), config())
    listed = compute_frame_sampling_weights(ConcatDataset([dataset(records * 3)], [1]), config())

    assert single[2] > 1.0
    assert repeated == listed == single * 3


def test_rejects_a_box_label_index_outside_the_classes() -> None:
    # Only the ignore index marks a box of no trained class, any other index is a corrupt table
    stray = {**box("car"), "box3d_label_index": len(_CLASSES)}

    with pytest.raises(ValueError, match="outside the 3 trained classes"):
        compute_frame_sampling_weights(ConcatDataset([dataset([[stray]])], [1]), config())


def test_rejects_a_split_without_boxes() -> None:
    with pytest.raises(ValueError, match="at least one box"):
        compute_frame_sampling_weights(ConcatDataset([dataset([[], []])], [1]), config())


@pytest.mark.parametrize(
    "overrides",
    [
        {"repeat_sampling_factor": 0.0},
        {"object_bev_range": [0.0, 0.0, 0.0, 0.0]},
        {"pedestrian_class_name": "person"},
        {"class_names": [*_CLASSES, LOW_PEDESTRIAN_CATEGORY]},
        {"filter_attributes": [["bicycle"]]},
    ],
)
def test_rejects_settings_that_cannot_weigh_a_frame(overrides: dict) -> None:
    with pytest.raises(ValueError):
        config(**overrides)


def test_the_ranks_share_one_epoch_without_overlap() -> None:
    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
    split = ConcatDataset([dataset([[box("car")] for _ in weights])], [1])
    full = list(DistributedWeightedRandomSampler(split, weights, seed=3))

    ranks = []
    for rank in range(2):
        sampler = DistributedWeightedRandomSampler(split, weights, seed=3)
        sampler.num_replicas, sampler.rank = 2, rank
        sampler.num_samples, sampler.total_size = len(weights) // 2, len(weights)
        ranks.append(list(sampler))

    # Every rank keeps every second draw of the same epoch, so together they cover it once
    assert len(full) == len(weights)
    assert ranks == [full[0::2], full[1::2]]


def test_locate_maps_every_index_to_its_source_and_sample() -> None:
    # Two records repeated twice, then three records once
    split = ConcatDataset(
        [dataset([[box("car")], [box("car")]]), dataset([[box("car")]] * 3)], [2, 1]
    )

    assert len(split) == 7
    assert [split.locate(index) for index in range(len(split))] == [
        (0, 0),
        (0, 1),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    for index in (-1, len(split)):
        with pytest.raises(IndexError):
            split.locate(index)


def test_rejects_weights_that_do_not_match_the_dataset() -> None:
    split = ConcatDataset([dataset([[box("car")], [box("car")]])], [1])

    with pytest.raises(ValueError, match="weights"):
        DistributedWeightedRandomSampler(split, [1.0], seed=0)
    with pytest.raises(ValueError, match="positive"):
        DistributedWeightedRandomSampler(split, [1.0, 0.0], seed=0)
