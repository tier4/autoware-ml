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

import pickle
from collections.abc import Mapping, Sequence
from typing import Any

from torch.utils.data import DataLoader

from autoware_ml.datamodule.base import DataModule, Dataset
from autoware_ml.datamodule.common.detection3d import (
    build_detection_dataloader,
    build_label_to_category,
    normalize_detection_sample,
    resolve_data_path,
    resolve_sweep_paths,
)
from autoware_ml.datamodule.common.sources import AnnotationSource, coerce_annotation_sources
from autoware_ml.datamodule.t4dataset.detection3d import (
    FrameSamplingConfig,
    coerce_frame_sampling,
    compute_frame_sampling_weights,
)
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.boxes3d.annotations import normalize_filter_attributes


class T4SegmentationDetection3DDataset(Dataset):
    """T4 dataset for combined PTv3 segmentation+detection from annotation sources."""

    def __init__(
        self,
        data_root: str,
        ann_sources: Sequence[AnnotationSource],
        class_names: list[str],
        name_mapping: Mapping[str, str],
        filter_attributes: list[list[str]] | None = None,
        use_valid_flag: bool = False,
        frame_sampling: FrameSamplingConfig | None = None,
        dataset_transforms: TransformsCompose | None = None,
    ) -> None:
        """Initialize the combined T4 segmentation+detection dataset."""
        super().__init__(dataset_transforms=dataset_transforms)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = name_mapping
        self.filter_attributes = normalize_filter_attributes(filter_attributes)
        self.use_valid_flag = use_valid_flag
        self.frame_sampling = frame_sampling

        self.data_infos: list[dict[str, Any]] = []
        for source in ann_sources:
            self.data_infos.extend(self._load_source(source))

        self.frame_weights = compute_frame_sampling_weights(
            self.data_infos,
            self.class_names,
            self.name_mapping,
            self.frame_sampling,
            self.filter_attributes,
            self.use_valid_flag,
        )

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
        """Build one combined metadata record consumed by the transform pipeline."""
        sample = self.data_infos[index]
        return {
            "instances": sample.get("instances", []),
            "class_names": self.class_names,
            "name_mapping": self.name_mapping,
            "label_to_category": sample["label_to_category"],
            "sample_token": sample["token"],
            "lidar_path": resolve_data_path(self.data_root, sample["lidar_path"]),
            "num_pts_feats": int(sample["lidar_points"].get("num_pts_feats", 5)),
            "sweeps": resolve_sweep_paths(sample, self.data_root),
            "pts_semantic_mask_categories": sample["pts_semantic_mask_categories"],
            "pts_semantic_mask_path": resolve_data_path(
                self.data_root, sample["pts_semantic_mask_path"]
            ),
        }


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
        use_valid_flag: bool = False,
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
            use_valid_flag: Whether per-instance validity flags filter boxes.
            train_frame_sampling: Repeat-factor frame sampling configuration.
            **kwargs: Keyword arguments forwarded to :class:`DataModule`.
        """
        super().__init__(**kwargs)
        self.data_root = data_root
        self.class_names = class_names
        self.name_mapping = name_mapping
        self.filter_attributes = normalize_filter_attributes(filter_attributes)
        self.use_valid_flag = use_valid_flag
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
            use_valid_flag=self.use_valid_flag,
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
