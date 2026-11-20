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


def random_source(random_state: np.random.RandomState | None) -> np.random.RandomState:
    """Return the random state a transform draws from.

    Args:
        random_state: Random state of a stream, None outside a stream.

    Returns:
        The given random state, or the global random state of NumPy outside a stream: the one
        ``np.random.uniform`` and the other functions of ``np.random`` draw from and
        ``np.random.seed`` seeds.
    """
    if random_state is not None:
        return random_state
    # NumPy exposes its global RandomState only under this name, which its type stubs declare
    return np.random.mtrand._rand


class BaseTransform(ABC):
    """Abstract base class for ModelGTSample data transformations.

    A temporal model reads the frames of a stream one after another and needs some
    augmentations alike in every frame, the global rotation or the flip. Such a transform
    sets ``stream_consistent`` and implements :meth:`transform_in_stream`, which draws from
    the random state it is given instead of the global one of NumPy. ``TransformsCompose``
    seeds that random state by the stream for every frame, so every frame of a stream draws
    the same values and the transform keeps nothing between the frames.

    Class Attributes (override in subclasses):
        _required_keys: Fields of the sample that must be set before the transform runs.
        stream_consistent: Whether the transform augments every frame of a stream alike, see
            :meth:`transform_in_stream`. A transform without it draws for every frame alone,
            also inside a stream.
    """

    _required_keys: Sequence[str] = ()
    stream_consistent: bool = False

    def __init__(self, probability: float | None = None) -> None:
        """Initialize the transform.

        Args:
            probability: Probability of applying the transform (0.0=never, 1.0=always).
                         Set to None if the transform should always run.
        """
        self._probability = probability

    def __call__(
        self, model_gt_sample: ModelGTSample, random_state: np.random.RandomState | None = None
    ) -> ModelGTSample:
        """Execute transform with probability and key validation.

        Order of operations:
            1. Validate required keys (raises KeyError if any missing)
            2. Check probability (skip if not triggered)
            3. Execute the actual transform

        Args:
            model_gt_sample: Dataclass to hold inputs for each sample.
            random_state: Random state of the stream of the sample, which the probability and
                the augmentation are drawn from. None for a sample augmented on its own,
                which draws from the global random state of NumPy.

        Returns:
            Updated ModelGTSample.
        """
        # 1. Validate required keys (raises error if any missing)
        self._validate_required_keys(model_gt_sample)

        # 2. Check probability (skip if not triggered), drawn from the stream when there is one
        if not self._should_apply(random_state):
            return self.on_skip(model_gt_sample)

        # 3. Execute the actual transform
        if random_state is None:
            return self.transform(model_gt_sample)
        return self.transform_in_stream(model_gt_sample, random_state)

    def transform_in_stream(
        self, model_gt_sample: ModelGTSample, random_state: np.random.RandomState
    ) -> ModelGTSample:
        """Transform one frame of a stream, as :meth:`transform` with the draws of the stream.

        Override it in a transform that sets ``stream_consistent``. Every random draw of the
        augmentation has to come from ``random_state``, in an order and a number that do not
        depend on the sample, so every frame of the stream draws the same values.

        Args:
            model_gt_sample: ModelGTSample instance with its required fields set.
            random_state: Random state seeded by the stream, the same for every frame of it.

        Returns:
            Updated ModelGTSample, possibly the same instance modified in place.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} sets stream_consistent but has no transform_in_stream."
        )

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

    def _should_apply(self, random_state: np.random.RandomState | None = None) -> bool:
        """Determine if transform should be applied based on probability.

        Args:
            random_state: Random state to draw from, None for the global one of NumPy.

        Returns:
            True if transform should be applied, False to skip.
        """
        if self._probability is None:
            return True
        if self._probability <= 0.0:
            return False
        if self._probability >= 1.0:
            return True
        return random_source(random_state).rand() < self._probability

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

    A sample of a stream is called with the seed of its stream. Every ``stream_consistent``
    transform then draws from a random state seeded by the stream and by the position of the
    transform in the pipeline, so the frames of a stream are augmented alike. The pipeline
    keeps nothing between the calls: the stream of a sample is named by the call alone.
    """

    def __init__(self, pipeline: Sequence[BaseTransform]):
        """Initialize the transform pipeline.

        Args:
            pipeline: Ordered transforms applied to each sample.
        """
        self.pipeline = pipeline

    def __call__(
        self, model_gt_sample: ModelGTSample, stream_seed: Sequence[int] | None = None
    ) -> ModelGTSample:
        """Apply each transform in order.

        Args:
            model_gt_sample: ModelGTSample instance passed through the configured transforms.
            stream_seed: Integers naming the stream of the sample, the same for every frame
                of it and each from 0 to 2**32 - 1. The sample then gets the augmentation of
                the stream. None for a sample augmented on its own.

        Returns:
            Transformed ModelGTSample instance after all pipeline stages have been applied.
        """
        for transform_index, transform in enumerate(self.pipeline):
            if stream_seed is not None and transform.stream_consistent:
                # A random state per transform, so the draws of one transform do not depend
                # on how many values the transforms before it drew for this frame.
                random_state = np.random.RandomState(list(stream_seed) + [transform_index])
                model_gt_sample = transform(model_gt_sample, random_state)
            else:
                model_gt_sample = transform(model_gt_sample)

        return model_gt_sample

    def requires(self, key: str) -> bool:
        """Whether a transform of the pipeline needs a field of the sample."""
        return any(key in transform._required_keys for transform in self.pipeline)

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
