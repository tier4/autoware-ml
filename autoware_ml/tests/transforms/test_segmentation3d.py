"""Tests for segmentation3d data transforms."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from autoware_ml.datamodule.pipeline_context import PipelineContext
from autoware_ml.transforms.base import BaseTransform, TransformsCompose
from autoware_ml.transforms.segmentation3d.formatting import PreparePointSegInput
from autoware_ml.transforms.segmentation3d.loading import LoadSegAnnotations3D
from autoware_ml.transforms.segmentation3d.mixing import FrustumMix, InstanceCopy
from autoware_ml.transforms.segmentation3d.range_view import RangeInterpolation


class _MixDataset:
    def __init__(self, sample: dict[str, npt.NDArray]) -> None:
        self._sample = sample

    def __len__(self) -> int:
        return 1

    def get_data_info(self, index: int) -> dict[str, npt.NDArray]:
        del index
        return {
            "points": self._sample["points"].copy(),
            "pts_semantic_mask": self._sample["pts_semantic_mask"].copy(),
        }

    def apply_transforms(
        self,
        input_dict: dict[str, npt.NDArray],
        dataset_transforms: TransformsCompose | None,
        context: PipelineContext,
    ) -> dict[str, npt.NDArray]:
        del context
        if dataset_transforms is None:
            return input_dict
        return dataset_transforms(input_dict)


def test_prepare_point_seg_input_scales_intensity() -> None:
    transform = PreparePointSegInput()
    output = transform(
        {
            "points": np.array([[1.0, 2.0, 3.0, 255.0], [4.0, 5.0, 6.0, 0.0]], dtype=np.float32),
            "pts_semantic_mask": np.array([7, 8], dtype=np.int64),
        }
    )

    assert np.allclose(
        output["coord"], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    )
    assert np.allclose(output["strength"], np.array([[1.0], [0.0]], dtype=np.float32))
    assert np.array_equal(output["segment"], np.array([7, 8], dtype=np.int64))


def test_frustum_mix_combines_points_from_source_and_mix_sample() -> None:
    np.random.seed(0)
    sample = {
        "points": np.array([[5.0, 0.0, 0.0, 1.0], [4.0, 1.0, 0.0, 2.0]], dtype=np.float32),
        "pts_semantic_mask": np.array([1, 2], dtype=np.int64),
    }
    mix_sample = {
        "points": np.array([[3.0, -1.0, 0.0, 3.0], [2.0, 0.0, 0.0, 4.0]], dtype=np.float32),
        "pts_semantic_mask": np.array([9, 8], dtype=np.int64),
    }

    dataset = _MixDataset(mix_sample)
    output = FrustumMix(height=8, width=16, fov_up=10.0, fov_down=-30.0, num_areas=[2], prob=1.0)(
        sample,
        context=PipelineContext(dataset=dataset, index=0),
    )

    assert output["points"].shape[1] == 4
    assert output["points"].shape[0] == output["pts_semantic_mask"].shape[0]
    assert set(output["pts_semantic_mask"].tolist()).issubset({1, 2, 8, 9})
    assert any(label in {8, 9} for label in output["pts_semantic_mask"].tolist())


def test_instance_copy_appends_requested_semantic_classes() -> None:
    sample = {
        "points": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "pts_semantic_mask": np.array([1], dtype=np.int64),
    }
    dataset = _MixDataset(
        {
            "points": np.array([[1.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 3.0]], dtype=np.float32),
            "pts_semantic_mask": np.array([4, 5], dtype=np.int64),
        }
    )

    output = InstanceCopy(instance_classes=[5], prob=1.0)(
        sample,
        context=PipelineContext(dataset=dataset, index=0),
    )

    assert output["points"].shape == (2, 4)
    assert np.array_equal(output["pts_semantic_mask"], np.array([1, 5], dtype=np.int64))


def test_frustum_mix_respects_zero_probability() -> None:
    sample = {
        "points": np.array([[5.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "pts_semantic_mask": np.array([1], dtype=np.int64),
    }

    output = FrustumMix(
        height=8,
        width=16,
        fov_up=10.0,
        fov_down=-30.0,
        num_areas=[2],
        prob=0.0,
    )(sample.copy())

    assert np.array_equal(output["points"], sample["points"])
    assert np.array_equal(output["pts_semantic_mask"], sample["pts_semantic_mask"])


def test_instance_copy_respects_zero_probability() -> None:
    sample = {
        "points": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "pts_semantic_mask": np.array([1], dtype=np.int64),
    }

    output = InstanceCopy(instance_classes=[5], prob=0.0)(sample.copy())

    assert np.array_equal(output["points"], sample["points"])
    assert np.array_equal(output["pts_semantic_mask"], sample["pts_semantic_mask"])


def test_mix_transforms_apply_pre_transform_to_secondary_sample() -> None:
    class _AppendClassNinetyNine(BaseTransform):
        def transform(self, input_dict: dict[str, npt.NDArray]) -> dict[str, npt.NDArray]:
            input_dict["pts_semantic_mask"] = np.full_like(input_dict["pts_semantic_mask"], 99)
            return input_dict

    sample = {
        "points": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "pts_semantic_mask": np.array([1], dtype=np.int64),
    }
    dataset = _MixDataset(
        {
            "points": np.array([[1.0, 0.0, 0.0, 2.0]], dtype=np.float32),
            "pts_semantic_mask": np.array([5], dtype=np.int64),
        }
    )

    output = InstanceCopy(
        instance_classes=[99],
        pre_transform=TransformsCompose(pipeline=[_AppendClassNinetyNine()]),
        prob=1.0,
    )(sample, context=PipelineContext(dataset=dataset, index=0))

    assert np.array_equal(output["pts_semantic_mask"], np.array([1, 99], dtype=np.int64))


def test_pipeline_context_reuses_current_sample_for_single_item_dataset(caplog) -> None:
    sample = {
        "points": np.array([[1.0, 0.0, 0.0, 2.0]], dtype=np.float32),
        "pts_semantic_mask": np.array([5], dtype=np.int64),
    }
    context = PipelineContext(dataset=_MixDataset(sample), index=0)

    with caplog.at_level(logging.WARNING):
        secondary = context.sample_secondary()

    assert secondary["points"].shape == sample["points"].shape
    assert np.array_equal(secondary["pts_semantic_mask"], sample["pts_semantic_mask"])
    assert "Dataset contains only one sample" in caplog.text


def test_range_interpolation_adds_midpoint_and_boundary_label() -> None:
    sample = {
        "points": np.array([[1.0, 1.0, 0.0, 0.5], [-1.0, -1.0, 0.0, 1.5]], dtype=np.float32),
        "pts_semantic_mask": np.array([3, 7], dtype=np.int64),
    }

    output = RangeInterpolation(height=1, width=4, fov_up=10.0, fov_down=-10.0, ignore_index=255)(
        sample
    )

    assert output["num_points"] == 2
    assert output["points"].shape[0] == 3
    assert output["pts_semantic_mask"].shape[0] == 3
    assert output["pts_semantic_mask"][-1] == 255


def test_load_seg_annotations_maps_unlisted_labels_to_ignore(tmp_path) -> None:
    path = tmp_path / "labels.bin"
    np.array([0, 1, 4, 7], dtype=np.uint8).tofile(path)

    output = LoadSegAnnotations3D(
        label_mapping={0: 3, 4: 9},
        max_label=7,
        ignore_index=255,
    )({"pts_semantic_mask_path": str(path)})

    assert np.array_equal(output["pts_semantic_mask"], np.array([3, 255, 9, 255], dtype=np.int64))


def test_load_seg_annotations_class_mapping_path(tmp_path) -> None:
    """LoadSegAnnotations3D should resolve labels via class_mapping and per-sample categories."""
    path = tmp_path / "labels.bin"
    np.array([2, 5, 9], dtype=np.uint8).tofile(path)

    output = LoadSegAnnotations3D(
        class_mapping={"car": 0, "pedestrian": 1},
        ignore_index=255,
    )(
        {
            "pts_semantic_mask_path": str(path),
            "pts_semantic_mask_categories": {"car": 2, "pedestrian": 5},
        }
    )

    assert np.array_equal(output["pts_semantic_mask"], np.array([0, 1, 255], dtype=np.int64))


def test_load_seg_annotations_respects_idx_begin_and_length(tmp_path) -> None:
    """LoadSegAnnotations3D should slice to [idx_begin : idx_begin + length]."""
    import numpy as np

    path = tmp_path / "labels.bin"
    np.array([0, 1, 2, 3, 4], dtype=np.uint8).tofile(path)

    output = LoadSegAnnotations3D(
        label_mapping={1: 10, 2: 20},
        max_label=4,
        ignore_index=255,
    )({"pts_semantic_mask_path": str(path), "idx_begin": 1, "length": 2})

    assert np.array_equal(output["pts_semantic_mask"], np.array([10, 20], dtype=np.int64))


def test_range_interpolation_no_interpolatable_points() -> None:
    """A single point has no neighbors; the point array must remain unchanged."""
    sample = {
        "points": np.array([[1.0, 0.0, 0.0, 0.5]], dtype=np.float32),
        "pts_semantic_mask": np.array([3], dtype=np.int64),
    }

    output = RangeInterpolation(height=4, width=8, fov_up=10.0, fov_down=-10.0, ignore_index=255)(
        sample
    )

    assert output["points"].shape == (1, 4)
    assert output["pts_semantic_mask"].shape == (1,)
    assert output["num_points"] == 1
