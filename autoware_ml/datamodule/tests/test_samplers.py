"""Tests for the repeat factor sampling and the scene streaming sampler."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import polars as pl
import pytest
from torch.utils.data.distributed import DistributedSampler

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.base_dataset import ConcatDataset, PaddingSlot
from autoware_ml.datamodule.samplers import (
    LOW_PEDESTRIAN_CATEGORY,
    DistributedWeightedRandomSampler,
    FrameSamplingConfig,
    GroupStreamingSampler,
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
    with pytest.raises(ValueError, match="non-negative"):
        DistributedWeightedRandomSampler(split, [1.0, -1.0], seed=0)
    with pytest.raises(ValueError, match="positive"):
        DistributedWeightedRandomSampler(split, [0.0, 0.0], seed=0)


class _FakeScenesDataset:
    """Minimal dataset stub exposing scene_index_groups()."""

    def __init__(self, scene_lengths: list[int]) -> None:
        self._groups = []
        start = 0
        for length in scene_lengths:
            self._groups.append(list(range(start, start + length)))
            start += length
        self.scene_starts = {group[0] for group in self._groups}
        self.total = start

    def scene_index_groups(self) -> list[list[int]]:
        return [list(group) for group in self._groups]

    def __len__(self) -> int:
        return sum(len(group) for group in self._groups)

    def __getitem__(self, index: int) -> int:
        return index


def test_lanes_stay_scene_contiguous_across_batches() -> None:
    """
    Input: scenes of 10, 5 and 5 frames served in order through two lanes on one rank.
    Expected: the joined 20 frames are cut in half, lane 0 holding scene 0 and lane 1 scenes
    1 and 2, and batch position k walks lane k frame by frame across consecutive batches.
    Check: the indices alternate between the two lanes, each lane in frame order.
    """
    sampler = GroupStreamingSampler(
        _FakeScenesDataset([10, 5, 5]), batch_size=2, shuffle=False, seed=0
    )
    expected = [index for pair in zip(range(0, 10), range(10, 20)) for index in pair]
    assert list(sampler) == expected


def test_set_epoch_reshuffles_deterministically() -> None:
    dataset = _FakeScenesDataset([4] * 8)

    def epoch_indices(epoch: int) -> list[int]:
        sampler = GroupStreamingSampler(dataset, batch_size=2, shuffle=True, seed=0)
        sampler.set_epoch(epoch)
        return list(sampler)

    assert epoch_indices(0) == epoch_indices(0)
    assert epoch_indices(0) != epoch_indices(1)


def test_raises_when_lanes_cannot_be_filled() -> None:
    """
    Input: one single-frame scene for two lanes.
    Expected: a lane without frames would leave its batch position empty, so the sampler
    refuses the layout.
    Check: building the sampler raises a ValueError naming the lanes it cannot fill.
    """
    with pytest.raises(ValueError, match="cannot fill"):
        GroupStreamingSampler(_FakeScenesDataset([1]), batch_size=2, shuffle=False, seed=0)


def test_is_a_distributed_sampler_so_lightning_does_not_wrap_it() -> None:
    """
    Input: a sampler built outside a distributed run.
    Expected: it is a ``DistributedSampler``, which Lightning leaves alone instead of wrapping
    it into one that keeps every ``num_replicas``-th index, and without a process group it
    serves the whole split as the only rank.
    Check: its type, its number of replicas and its rank.
    """
    sampler = GroupStreamingSampler(_FakeScenesDataset([4, 4]), batch_size=2, shuffle=False, seed=0)
    assert isinstance(sampler, DistributedSampler)
    assert (sampler.num_replicas, sampler.rank) == (1, 0)


def _ranks(
    dataset: _FakeScenesDataset, world_size: int, batch_size: int, shuffle: bool, seed: int = 0
):
    """The sampler of every rank of a ``world_size``-rank job, built as every rank would."""
    samplers = []
    for rank in range(world_size):
        sampler = GroupStreamingSampler(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)
        sampler.num_replicas = world_size
        sampler.rank = rank
        samplers.append(sampler)
    return samplers


def _lanes(sampler: GroupStreamingSampler) -> list[list[int]]:
    indices = list(iter(sampler))
    return [indices[lane :: sampler.batch_size] for lane in range(sampler.batch_size)]


class TestFixedLengthTrainingEpochs:
    """With ``shuffle=True`` every epoch has the same length (Lightning assumes it)."""

    def test_length_is_constant_across_epochs(self) -> None:
        """
        Input: scenes of 7, 5, 18, 3, 8 and 9 frames shuffled through two lanes over six
        epochs.
        Expected: Lightning derives the epoch length once, so whatever the scene order every
        epoch serves ``rounds_per_epoch`` rounds of two frames, at least every frame.
        Check: the length and the number of indices yielded every epoch.
        """
        dataset = _FakeScenesDataset([7, 5, 18, 3, 8, 9])
        sampler = GroupStreamingSampler(dataset, batch_size=2, shuffle=True, seed=0)

        lengths = []
        for epoch in range(6):
            sampler.set_epoch(epoch)
            lengths.append(len(sampler))
            assert len(list(iter(sampler))) == lengths[-1]

        assert len(set(lengths)) == 1
        assert lengths[0] == sampler.rounds_per_epoch * 2
        assert lengths[0] >= dataset.total

    def test_lanes_only_advance_within_a_scene_or_start_a_new_stream(self) -> None:
        """
        Input: scenes of 7, 5, 18, 3, 8 and 9 frames shuffled through three lanes over four
        epochs.
        Expected: a lane moves on to the next frame of its scene or starts a new stream at the
        first frame of a scene, and a padded lane then repeats its last frame.
        Check: every pair of consecutive frames of every lane.
        """
        dataset = _FakeScenesDataset([7, 5, 18, 3, 8, 9])
        sampler = GroupStreamingSampler(dataset, batch_size=3, shuffle=True, seed=0)

        for epoch in range(4):
            sampler.set_epoch(epoch)
            for lane in _lanes(sampler):
                for previous, current in zip(lane, lane[1:]):
                    assert (
                        current == previous + 1
                        or current in dataset.scene_starts
                        or (isinstance(current, PaddingSlot) and current == previous)
                    )


class TestEvaluationServesEveryFrame:
    """With ``shuffle=False`` the lanes keep the record order and serve every frame once."""

    def test_single_lane_serves_every_frame_exactly_once_in_file_order(self) -> None:
        """
        Input: scenes of 7, 5 and 18 frames served in order through one lane.
        Expected: one lane on one rank needs no padding, so it serves every frame once, the
        scenes in the order of the record table.
        Check: the indices are every frame, in order.
        """
        dataset = _FakeScenesDataset([7, 5, 18])
        sampler = GroupStreamingSampler(dataset, batch_size=1, shuffle=False, seed=0)

        assert list(iter(sampler)) == list(range(dataset.total))

    def test_a_lane_cut_inside_a_scene_continues_that_scene(self) -> None:
        """
        Input: one scene of nine frames served in order through two lanes.
        Expected: the cut falls inside the scene, so lane 1 starts at frame 4 and walks on to
        the end of the scene, while lane 0, a frame shorter, ends with a padding slot
        repeating frame 3.
        Check: the frames of each lane and which of them is a padding slot.
        """
        sampler = GroupStreamingSampler(
            _FakeScenesDataset([9]), batch_size=2, shuffle=False, seed=0
        )

        lanes = _lanes(sampler)
        assert lanes == [[0, 1, 2, 3, 3], [4, 5, 6, 7, 8]]
        assert [isinstance(index, PaddingSlot) for index in lanes[0]] == [False] * 4 + [True]


# (world_size, batch_size, shuffle) of the lane layouts every property below holds for.
LANE_LAYOUTS = [
    (1, 1, False),
    (1, 3, True),
    (2, 1, False),
    (2, 2, False),
    (2, 3, True),
    (3, 2, False),
    (4, 3, True),
    (8, 2, False),
]
UNEVEN_SCENES = [17, 4, 9, 23, 1, 12, 6, 30, 2, 11, 8]


@pytest.mark.parametrize(("world_size", "batch_size", "shuffle"), LANE_LAYOUTS)
def test_every_frame_is_served_once_across_ranks(
    world_size: int, batch_size: int, shuffle: bool
) -> None:
    """
    Input: eleven uneven scenes of 123 frames over three epochs, for several world sizes and
    lane counts, shuffled and in order.
    Expected: whatever the layout, every rank serves the length it reports, rounds_per_epoch
    rounds, the real samples of all ranks are every frame of the split exactly once, and fewer
    slots than there are lanes are padding.
    Check: the indices of every rank against its length, the sorted real indices of all
    ranks, and the number of padding slots.
    """
    dataset = _FakeScenesDataset(UNEVEN_SCENES)
    samplers = _ranks(dataset, world_size, batch_size, shuffle=shuffle)

    for epoch in range(3):
        served = []
        for sampler in samplers:
            sampler.set_epoch(epoch)
            indices = list(sampler)
            assert len(indices) == len(sampler) == sampler.rounds_per_epoch * batch_size
            served.extend(indices)
        real = [index for index in served if not isinstance(index, PaddingSlot)]
        assert sorted(real) == list(range(dataset.total))
        assert len(served) - len(real) < world_size * batch_size


@pytest.mark.parametrize(("world_size", "batch_size", "shuffle"), LANE_LAYOUTS)
def test_a_lane_ends_with_at_most_one_padding_slot_holding_its_last_frame(
    world_size: int, batch_size: int, shuffle: bool
) -> None:
    """
    Input: eleven uneven scenes of 123 frames over three epochs, for several world sizes and
    lane counts, shuffled and in order.
    Expected: the lanes differ by at most one frame, so a lane is padded with at most one
    slot, at its end, holding the index of its last frame.
    Check: the padding slots of every lane of every rank: their number, their position and
    their value.
    """
    samplers = _ranks(_FakeScenesDataset(UNEVEN_SCENES), world_size, batch_size, shuffle=shuffle)

    for epoch in range(3):
        for sampler in samplers:
            sampler.set_epoch(epoch)
            for lane in _lanes(sampler):
                padded = [isinstance(index, PaddingSlot) for index in lane]
                assert sum(padded) <= 1
                assert not any(padded[:-1])
                if padded[-1]:
                    assert lane[-1] == lane[-2]
