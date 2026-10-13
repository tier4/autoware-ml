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

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample


class BaseTransform(ABC):
    """Abstract base class for ModelGTSample data transformations.

    Class Attributes (override in subclasses):
        _required_keys: Fields of the sample that must be set before the transform runs.
    """

    _required_keys: Sequence[str] = ()

    def __init__(self, probability: float | None = None) -> None:
        """Initialize the transform.

        Args:
            probability: Probability of applying the transform (0.0=never, 1.0=always).
                         Set to None if the transform should always run.
        """
        self._probability = probability

    def __call__(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Execute transform with probability and key validation.

        Order of operations:
            1. Validate required keys (raises KeyError if any missing)
            2. Check probability (skip if not triggered)
            3. Execute the actual transform

        Args:
            model_gt_sample: Dataclass to hold inputs for each sample.

        Returns:
            Updated ModelGTSample.
        """
        # 1. Validate required keys (raises error if any missing)
        self._validate_required_keys(model_gt_sample)

        # 2. Check probability (skip if not triggered)
        if not self._should_apply():
            return self.on_skip(model_gt_sample)

        # 3. Execute the actual transform
        return self.transform(model_gt_sample)

    def _validate_required_keys(self, model_gt_sample: ModelGTSample) -> None:
        """Raise ``KeyError`` when any required key is missing.

        Args:
            model_gt_sample: ModelGTSample instance validated before transform execution.

        Raises:
            KeyError: If a required key defined by the transform is absent.
        """
        for key in self._required_keys:
            if getattr(model_gt_sample, key) is None:
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

    def on_skip(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Called when transform is skipped due to probability.

        Override for custom behavior when transform is skipped.
        Default implementation returns input unchanged.

        Args:
            model_gt_sample: The sample the transform skipped.

        Returns:
            The sample, unchanged by default.
        """
        return model_gt_sample

    @abstractmethod
    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Transform one sample.

        Args:
            model_gt_sample: ModelGTSample instance with its required fields set.

        Returns:
            Updated ModelGTSample, possibly the same instance modified in place.
        """
        raise NotImplementedError


class TransformsCompose:
    """Apply a sequence of transforms in order.

    The composed transform forwards one sample through every configured transform
    and returns the final result.
    """

    def __init__(self, pipeline: Sequence[BaseTransform]):
        """Initialize the transform pipeline.

        Args:
            pipeline: Ordered transforms applied to each sample.
        """
        self.pipeline = pipeline

    def __call__(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Apply each transform in order.

        Args:
            model_gt_sample: ModelGTSample instance passed through the configured transforms.

        Returns:
            Transformed ModelGTSample instance after all pipeline stages have been applied.
        """
        for transform in self.pipeline:
            model_gt_sample = transform(model_gt_sample)

        return model_gt_sample

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
