# Copyright 2025 TIER IV, Inc.
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

"""Base classes and composition utilities for data transforms.

This module defines the core transform protocol used across Autoware-ML data
pipelines and provides sequential composition helpers.
"""

from abc import abstractmethod
from typing import Sequence, Protocol

import numpy as np

from autoware_ml.datamodule.multi_tasks.dataclasses.multi_task_samples import MultiTaskGTSample


class MultiTaskBaseTransform(Protocol):
    """Abstract base class for MultiTaskGTSample data transformations.

    Class Attributes (override in subclasses):
        p: Probability of applying the transform (0.0=never, 1.0=always).
           Set to None for transforms that always run.
        _required_keys: List of keys that must exist in input_dict.
        _optional_keys: List of keys that may be missing (triggers apply_defaults).

    Subclasses should document their key contracts in the class docstring:
        - Required keys: Keys that must exist (KeyError raised otherwise)
        - Optional keys: Keys that are used if present (apply_defaults called if missing)
        - Generated keys: Keys added/modified by the transform
    """

    required_keys: Sequence[str]

    def __init__(self, probability: float | None = None) -> None:
        """Initialize the transform with required keys.

        Args:
            required_keys: List of keys that must exist in the input MultiTaskGTSample.
            probability: Probability of applying the transform (0.0=never, 1.0=always).
                         Set to None if the transform should always run.
        """
        self._probability = probability

    def __call__(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Execute transform with probability and key validation.

        Order of operations:
            1. Validate required keys (raises KeyError if any missing)
            2. Check probability (skip if not triggered)
            3. Execute the actual transform

        Args:
            multi_task_gt_sample: Dataclass to hold inputs for each sample.

        Returns:
            Updated MultiTaskGTSample.
        """
        # 1. Validate required keys (raises error if any missing)
        self._validate_required_keys(multi_task_gt_sample)

        # 3. Check probability (skip if not triggered)
        if not self._should_apply():
            return self.on_skip(multi_task_gt_sample)

        # 4. Execute the actual transform
        return self.transform(multi_task_gt_sample)

    def _validate_required_keys(self, multi_task_gt_sample: MultiTaskGTSample) -> None:
        """Raise ``KeyError`` when any required key is missing.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance validated before transform execution.

        Raises:
            KeyError: If a required key defined by the transform is absent.
        """
        for key in self._required_keys:
            required_attr = getattr(multi_task_gt_sample, key, None)
            if required_attr is None:
                raise KeyError(f"{self.__class__.__name__}: Missing required key '{key}'")

    def _should_apply(self) -> bool:
        """Determine if transform should be applied based on probability.

        Returns:
            True if transform should be applied, False to skip.
        """
        if self._probability is None:
            return True
        if self._probability <= 0.0:
            return False
        if self._probability >= 1.0:
            return True
        return np.random.rand() < self._probability

    def on_skip(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Called when transform is skipped due to probability.

        Override for custom behavior when transform is skipped.
        Default implementation returns input unchanged.

        Args:
            input_dict: The input dictionary.

        Returns:
            The (possibly modified) input dictionary.
        """
        return multi_task_gt_sample

    @abstractmethod
    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Process input dictionary and return updated mapping.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance with required keys present, optional keys
                                   populated by apply_defaults() if they were missing.

        Returns:
            Updated dictionary (may be the same object modified in-place).
        """
        raise NotImplementedError


class MultiTaskTransformsCompose:
    """Apply a sequence of transforms in order.

    The composed transform forwards one sample dictionary through every
    configured transform and returns the final result.
    """

    def __init__(self, pipeline: Sequence[MultiTaskBaseTransform]):
        """Initialize the transform pipeline.

        Args:
            pipeline: Ordered transforms applied to each input dictionary.
        """
        self.pipeline = pipeline

    def __call__(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Apply each transform sequentially, merging updates.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance passed through the configured transforms.

        Returns:
            Transformed MultiTaskGTSample instance after all pipeline stages have been applied.
        """
        for transform in self.pipeline:
            multi_task_gt_sample = transform(multi_task_gt_sample)

        return multi_task_gt_sample

    def __repr__(self) -> str:
        """Return a formatted string representation of the composition.

        Returns:
            Multi-line string showing the ordered transform pipeline.
        """
        if not self.pipeline:
            return f"{self.__class__.__name__}(pipeline=[])"

        format_string = [f"{self.__class__.__name__}("]
        for i, transform in enumerate(self.pipeline):
            format_string.append(f"  ({i}): {transform}")
        format_string.append(")")
        return "\n".join(format_string)
