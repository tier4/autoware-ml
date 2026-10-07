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

"""T4Dataset datamodule for combined PTv3 segmentation+detection evaluation.

Each split accepts one annotation file or a list of explicit
:class:`AnnotationSource` specs, so detection+segmentation sets can be mixed
with segmentation-only sources. Sources declared ``det3d: false`` have their
instances dropped, so their frames carry no detection supervision (models
treat box-less frames as detection-unsupervised).
"""

from __future__ import annotations

from copy import deepcopy
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.datamodule.common.detection3d import (
    build_detection_dataloader,
    build_label_to_category,
    normalize_detection_sample,
    resolve_data_path,
    resolve_sweep_paths,
)
from autoware_ml.datamodule.common.serialization import SerializedSampleList
from autoware_ml.datamodule.common.sources import AnnotationSource, coerce_annotation_sources
from autoware_ml.datamodule.t4dataset.frame_meta import scene_dir_fragment
from autoware_ml.datamodule.t4dataset.detection3d import (
    FrameSamplingConfig,
    coerce_frame_sampling,
    compute_frame_sampling_weights,
)
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.boxes3d.annotations import normalize_filter_attributes


def _interpolate_numbered_path(
    current_path: str,
    next_path: str,
    fraction: float,
) -> Path | None:
    """Interpolate between matching paths whose filenames contain frame numbers."""
    current = Path(current_path)
    following = Path(next_path)
    if current.parent != following.parent:
        return None

    current_matches = list(re.finditer(r"\d+", current.name))
    next_matches = list(re.finditer(r"\d+", following.name))
    if not current_matches or not next_matches:
        return None
    current_match = current_matches[-1]
    next_match = next_matches[-1]
    if (
        current.name[: current_match.start()] != following.name[: next_match.start()]
        or current.name[current_match.end() :] != following.name[next_match.end() :]
    ):
        return None

    current_number = int(current_match.group())
    next_number = int(next_match.group())
    target_number = round(current_number + (next_number - current_number) * fraction)
    lower, upper = sorted((current_number, next_number))
    if not lower < target_number < upper:
        return None

    width = max(len(current_match.group()), len(next_match.group()))
    filename = (
        current.name[: current_match.start()]
        + f"{target_number:0{width}d}"
        + current.name[current_match.end() :]
    )
    return current.parent / filename


class T4SegmentationDetection3DDataset(Dataset):
    """T4 dataset for combined PTv3 segmentation+detection from annotation sources."""

    def __init__(
        self,
        data_root: str,
        ann_sources: Sequence[AnnotationSource],
        class_names: list[str],
        name_mapping: Mapping[str, str],
        filter_attributes: list[list[str]] | None = None,
        frame_sampling: FrameSamplingConfig | None = None,
        dataset_transforms: TransformsCompose | None = None,
    ) -> None:
        """Initialize the combined T4 segmentation+detection dataset.

        Args:
            data_root: Dataset root directory.
            ann_sources: Annotation sources merged into one sample list.
            class_names: Detector class names in label order.
            name_mapping: Raw category to detector class mapping.
            filter_attributes: Class and attribute name pairs excluded from detection.
            frame_sampling: Repeat-factor settings, or ``None`` to disable weighting.
            dataset_transforms: Optional dataset transform pipeline.
        """
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = name_mapping
        self.filter_attributes = normalize_filter_attributes(filter_attributes)
        self.frame_sampling = frame_sampling

        data_infos: list[dict[str, Any]] = []
        for source in ann_sources:
            data_infos.extend(self._load_source(source))

        self.frame_weights = compute_frame_sampling_weights(
            data_infos,
            self.class_names,
            self.name_mapping,
            self.frame_sampling,
            self.filter_attributes,
        )
        # Serialize last: frame_weights above must run on the live list.
        self.data_infos = SerializedSampleList(data_infos)

    def _load_source(self, source: AnnotationSource) -> list[dict[str, Any]]:
        """Load one annotation source and apply its supervision declaration."""
        with open(source.path, "rb") as file:
            data = pickle.load(file)
        label_to_category = (
            build_label_to_category(data.get("metainfo", {})) if source.det3d else {}
        )

        source_infos: list[dict[str, Any]] = []
        for raw_sample in data["data_list"]:
            token = raw_sample.get("token", "<unknown>")
            if source.det3d and "instances" not in raw_sample:
                raise ValueError(
                    f"Record with token '{token}' in '{source.path}' has no 'instances' but the "
                    f"source declares det3d supervision. Declare 'det3d: false' for "
                    f"segmentation-only sources."
                )
            if "pts_semantic_mask_path" not in raw_sample:
                raise ValueError(
                    f"Record with token '{token}' in '{source.path}' is missing "
                    f"'pts_semantic_mask_path'. Segdet sources must provide a mask file per "
                    f"frame even when seg3d supervision is disabled (its labels are ignored)."
                )
            sample = normalize_detection_sample(raw_sample)
            if not source.det3d:
                sample["instances"] = []
            sample["has_detection_ground_truth"] = source.det3d
            sample["has_segmentation_ground_truth"] = source.seg3d
            sample["label_to_category"] = label_to_category
            sample["pts_semantic_mask_path"] = raw_sample["pts_semantic_mask_path"]
            sample["pts_semantic_mask_categories"] = (
                raw_sample["pts_semantic_mask_categories"] if source.seg3d else {}
            )
            source_infos.append(sample)
        return source_infos * source.repeat

    def __len__(self) -> int:
        """Return the number of T4 segdet samples."""
        return len(self.data_infos)

    def get_data_info(self, index: int) -> dict[str, Any]:
        """Build one combined metadata record consumed by the transform pipeline.

        Args:
            index: Dataset index.

        Returns:
            Metadata dictionary with detection and segmentation fields.
        """
        sample = self.data_infos[index]
        info = {
            "instances": sample.get("instances", []),
            "class_names": self.class_names,
            "name_mapping": self.name_mapping,
            "label_to_category": sample["label_to_category"],
            "sample_token": sample["token"],
            "has_detection_ground_truth": sample["has_detection_ground_truth"],
            "has_segmentation_ground_truth": sample["has_segmentation_ground_truth"],
            "lidar_path": resolve_data_path(self.data_root, sample["lidar_path"]),
            "num_pts_feats": int(sample["lidar_points"].get("num_pts_feats", 5)),
            "sweeps": resolve_sweep_paths(sample, self.data_root),
            "pts_semantic_mask_categories": sample["pts_semantic_mask_categories"],
            "pts_semantic_mask_path": resolve_data_path(
                self.data_root, sample["pts_semantic_mask_path"]
            ),
            "timestamp": float(sample["timestamp"]),
            "ego2global": np.asarray(sample["ego2global"], dtype=np.float64),
            "scene_token": scene_dir_fragment(sample["lidar_path"], self.data_root),
        }
        images = sample.get("images")
        if images is not None:
            if not isinstance(images, Mapping):
                raise TypeError(f"Record {sample['token']!r} images must be a mapping")
            resolved_images = {}
            for camera_name, camera_info in images.items():
                if not isinstance(camera_info, Mapping):
                    raise TypeError(
                        f"Record {sample['token']!r} camera {camera_name!r} must be a mapping"
                    )
                image_path = camera_info.get("img_path")
                if not image_path:
                    raise ValueError(
                        f"Record {sample['token']!r} camera {camera_name!r} is missing img_path"
                    )
                resolved_images[camera_name] = {
                    **camera_info,
                    "img_path": resolve_data_path(self.data_root, image_path),
                }
            info["images"] = resolved_images
        return info

    def get_intermediate_prediction_infos(
        self,
        index: int,
        prediction_frequency_hz: float,
    ) -> list[dict[str, Any]]:
        """Build unlabeled current-sweep frames between adjacent 1 Hz GT records.

        T4 segdet annotation pickles contain ground-truth keyframes, while the
        source folders retain the intermediate LiDAR and camera files. This
        method reconstructs those prediction-only records by interpolating the
        numbered filenames and timestamps. Historical ``sweeps`` are always
        emptied so each prediction consumes only its current 10 Hz frame.
        """
        if prediction_frequency_hz <= 0:
            raise ValueError("prediction_frequency_hz must be greater than zero")
        if index + 1 >= len(self):
            return []

        current = self.get_data_info(index)
        following = self.get_data_info(index + 1)
        current_scene = current.get("scene_token")
        following_scene = following.get("scene_token")
        if current_scene is None or following_scene is None:
            raise ValueError(
                "Intermediate prediction reconstruction requires scene_token on both keyframes"
            )
        if current_scene != following_scene:
            return []

        if current.get("timestamp") is None or following.get("timestamp") is None:
            raise ValueError(
                "Intermediate prediction reconstruction requires timestamps on both keyframes"
            )
        current_timestamp = float(current["timestamp"])
        next_timestamp = float(following["timestamp"])
        duration = next_timestamp - current_timestamp
        if duration <= 0:
            raise ValueError(
                "Following keyframe timestamp must be later than the current timestamp"
            )
        interval_count = int(round(duration * prediction_frequency_hz))
        if interval_count <= 1:
            return []

        current_images = current.get("images")
        next_images = following.get("images")
        if current_images is None and next_images is None:
            current_images = {}
            next_images = {}
        elif not isinstance(current_images, Mapping) or not isinstance(next_images, Mapping):
            raise ValueError(
                "Camera metadata must be present on both keyframes when reconstructing images"
            )
        if set(current_images) != set(next_images):
            raise ValueError(
                "Camera sets differ between adjacent keyframes: "
                f"{sorted(current_images)} != {sorted(next_images)}"
            )

        intermediate_infos: list[dict[str, Any]] = []
        used_lidar_paths: set[Path] = set()
        for offset in range(1, interval_count):
            fraction = offset / interval_count
            lidar_path = _interpolate_numbered_path(
                current["lidar_path"], following["lidar_path"], fraction
            )
            if lidar_path is None:
                raise ValueError(
                    "Could not interpolate an intermediate LiDAR path between "
                    f"{current['lidar_path']!r} and {following['lidar_path']!r}"
                )
            if lidar_path in used_lidar_paths:
                raise ValueError(
                    f"Intermediate frequency maps multiple timestamps to {str(lidar_path)!r}"
                )
            if not lidar_path.is_file():
                raise FileNotFoundError(
                    f"Missing intermediate LiDAR frame at offset {offset}: {lidar_path}"
                )
            used_lidar_paths.add(lidar_path)

            info = deepcopy(current)
            info["sample_token"] = (
                f"{current['sample_token']}:prediction:{lidar_path.name.split('.', 1)[0]}"
            )
            info["timestamp"] = current_timestamp + duration * fraction
            info["lidar_path"] = str(lidar_path)
            info["sweeps"] = []
            info["instances"] = []
            info["label_to_category"] = {}
            info["has_detection_ground_truth"] = False
            info["has_segmentation_ground_truth"] = False
            info["pts_semantic_mask_categories"] = {}
            info.pop("pts_semantic_mask_path", None)

            interpolated_images: dict[str, Any] = {}
            for camera_name, current_camera in current_images.items():
                next_camera = next_images.get(camera_name)
                if not isinstance(next_camera, Mapping):
                    raise TypeError(
                        f"Following camera metadata for {camera_name!r} must be a mapping"
                    )
                image_path = _interpolate_numbered_path(
                    current_camera["img_path"], next_camera["img_path"], fraction
                )
                if image_path is None:
                    raise ValueError(
                        f"Could not interpolate camera {camera_name!r} between "
                        f"{current_camera['img_path']!r} and {next_camera['img_path']!r}"
                    )
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Missing intermediate image for {camera_name!r} at offset "
                        f"{offset}: {image_path}"
                    )
                camera_info = deepcopy(current_camera)
                camera_info["img_path"] = str(image_path)
                if (
                    current_camera.get("timestamp") is not None
                    and next_camera.get("timestamp") is not None
                ):
                    camera_info["timestamp"] = (
                        float(current_camera["timestamp"])
                        + (float(next_camera["timestamp"]) - float(current_camera["timestamp"]))
                        * fraction
                    )
                interpolated_images[camera_name] = camera_info
            info["images"] = interpolated_images
            intermediate_infos.append(info)

        return intermediate_infos


class T4SegmentationDetection3DDataModule(DataModule):
    """Create T4 dataloaders for combined PTv3 segmentation+detection evaluation."""

    def __init__(
        self,
        data_root: str,
        train_ann_file: str | Sequence[Mapping[str, Any]],
        val_ann_file: str | Sequence[Mapping[str, Any]],
        test_ann_file: str | Sequence[Mapping[str, Any]],
        class_names: list[str],
        name_mapping: Mapping[str, str],
        filter_attributes: list[list[str]] | None = None,
        train_frame_sampling: FrameSamplingConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the combined T4 segmentation+detection datamodule.

        Args:
            data_root: Dataset root directory.
            train_ann_file: Training annotation file path, or a list of
                explicit source mappings with exactly the keys ``path``,
                ``det3d``, ``seg3d``, and ``repeat``. See
                :func:`coerce_annotation_sources`.
            val_ann_file: Validation annotation file path or source list.
            test_ann_file: Test annotation file path or source list (also
                used for the predict split).
            class_names: Ordered detector class names.
            name_mapping: Raw-name to detector-class mapping.
            filter_attributes: Attribute pairs excluded from annotations.
            train_frame_sampling: Repeat-factor frame sampling configuration.
            **kwargs: Keyword arguments forwarded to :class:`DataModule`.
        """
        super().__init__(**kwargs)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = name_mapping
        self.filter_attributes = normalize_filter_attributes(filter_attributes)
        self.train_frame_sampling = coerce_frame_sampling(train_frame_sampling)

        self.ann_sources = {
            "train": coerce_annotation_sources(train_ann_file, data_root),
            "val": coerce_annotation_sources(val_ann_file, data_root),
            "test": coerce_annotation_sources(test_ann_file, data_root),
            "predict": coerce_annotation_sources(test_ann_file, data_root),
        }

    def _create_dataset(
        self, split: str, dataset_transforms: TransformsCompose | None = None
    ) -> Dataset:
        """Instantiate the combined dataset for one split."""
        return T4SegmentationDetection3DDataset(
            data_root=self.data_root,
            ann_sources=self.ann_sources[split],
            class_names=self.class_names,
            name_mapping=self.name_mapping,
            filter_attributes=self.filter_attributes,
            frame_sampling=self.train_frame_sampling if split == "train" else None,
            dataset_transforms=dataset_transforms,
        )

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create a joint dataloader with optional train RFS sampling."""
        return build_detection_dataloader(
            dataset=getattr(self, f"{split}_dataset"),
            dataloader_cfg=getattr(self, f"{split}_dataloader_cfg"),
            is_train=split == "train",
            train_frame_sampling=self.train_frame_sampling,
            collate_fn=self.collate_fn,
        )
