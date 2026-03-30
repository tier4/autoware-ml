"""Pipeline context utilities for metadata-first dataset pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autoware_ml.transforms.base import TransformsCompose

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Provide dataset access for context-aware transforms.

    The context keeps orchestration state out of sample dictionaries while
    still allowing transforms such as sample-mixing augmentations to request
    secondary examples.
    """

    dataset: Any
    index: int
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def get_data_info(self, index: int) -> dict[str, Any]:
        """Load raw metadata for the requested dataset index.

        Args:
            index: Dataset index.

        Returns:
            Metadata dictionary for the sample.
        """
        return self.dataset.get_data_info(index)

    def sample_secondary(
        self,
        pre_transform: TransformsCompose | None = None,
    ) -> dict[str, Any]:
        """Sample and optionally preprocess a secondary dataset example.

        Args:
            pre_transform: Optional pipeline applied to the sampled metadata.

        Returns:
            Secondary sample dictionary, optionally materialized by
            ``pre_transform``.
        """
        dataset_length = len(self.dataset)
        if dataset_length <= 1:
            logger.warning(
                "Dataset contains only one sample; reusing the current sample as the secondary sample."
            )
            secondary_index = self.index
        else:
            # Ensure the secondary index is different from the current index
            secondary_index = int(self.rng.integers(0, dataset_length - 1))
            if secondary_index >= self.index:
                secondary_index += 1

        sample = self.get_data_info(secondary_index)
        if pre_transform is None:
            return sample

        secondary_context = PipelineContext(
            dataset=self.dataset, index=secondary_index, rng=self.rng
        )
        return self.dataset.apply_transforms(sample, pre_transform, secondary_context)
