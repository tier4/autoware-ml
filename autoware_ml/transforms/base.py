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
from typing import Any, NamedTuple

import numpy as np

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample


class StreamParameters(NamedTuple):
    """Augmentation one transform applies to every frame of a stream.

    A temporal model reads the frames of a stream one after another. The parameters are drawn
    once for the stream, see :meth:`BaseTransform.sample_stream_parameters`, and handed to the
    transform with every frame of it, so the frames are augmented alike.

    Attributes:
        apply: Whether the transform runs on the frames of the stream, the outcome of its
            probability.
        values: Parameters the transform drew, None for a transform the stream skips.
    """

    apply: bool
    values: Any


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

    def __call__(
        self, model_gt_sample: ModelGTSample, stream_parameters: StreamParameters | None = None
    ) -> ModelGTSample:
        """Execute transform with probability and key validation.

        Order of operations:
            1. Validate required keys (raises KeyError if any missing)
            2. Check probability (skip if not triggered)
            3. Execute the actual transform

        Args:
            model_gt_sample: Dataclass to hold inputs for each sample.
            stream_parameters: Augmentation drawn for the stream of the sample, which replaces
                the probability and the draw of the transform. None draws for the sample alone.

        Returns:
            Updated ModelGTSample.
        """
        # 1. Validate required keys (raises error if any missing)
        self._validate_required_keys(model_gt_sample)

        # 2. Check probability (skip if not triggered)
        apply = self._should_apply() if stream_parameters is None else stream_parameters.apply
        if not apply:
            return self.on_skip(model_gt_sample)

        # 3. Execute the actual transform
        if stream_parameters is None:
            return self.transform(model_gt_sample)
        return self.transform_with_parameters(model_gt_sample, stream_parameters.values)

    def sample_parameters(self) -> Any:
        """Draw the parameters the transform shares between the frames of a stream.

        Override it together with :meth:`transform_with_parameters` in a transform whose
        augmentation a temporal model needs alike in every frame, the global rotation or the
        flip. The parameters are drawn before any frame is loaded and are pickled to the
        dataloader workers, so they are plain values that depend on no sample.

        Returns:
            The drawn parameters, None for a transform that draws for every sample alone.
        """
        return None

    def sample_stream_parameters(self) -> StreamParameters | None:
        """Draw the augmentation of one stream: the probability, then the parameters.

        Returns:
            The augmentation every frame of the stream gets, None for a transform that draws
            for every sample alone.
        """
        if type(self).sample_parameters is BaseTransform.sample_parameters:
            return None
        if not self._should_apply():
            return StreamParameters(apply=False, values=None)
        return StreamParameters(apply=True, values=self.sample_parameters())

    def transform_with_parameters(
        self, model_gt_sample: ModelGTSample, parameters: Any
    ) -> ModelGTSample:
        """Transform one sample with the given parameters instead of drawing them.

        Args:
            model_gt_sample: ModelGTSample instance with its required fields set.
            parameters: Parameters drawn by :meth:`sample_parameters`.

        Returns:
            Updated ModelGTSample, possibly the same instance modified in place.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} draws parameters for a stream but cannot apply them."
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

    def __call__(
        self,
        model_gt_sample: ModelGTSample,
        stream_parameters: Sequence[StreamParameters | None] | None = None,
    ) -> ModelGTSample:
        """Apply each transform in order.

        Args:
            model_gt_sample: ModelGTSample instance passed through the configured transforms.
            stream_parameters: Augmentation of the stream of the sample, one entry per
                transform as drawn by :meth:`sample_stream_parameters`. None draws every
                augmentation for the sample alone.

        Returns:
            Transformed ModelGTSample instance after all pipeline stages have been applied.

        Raises:
            ValueError: If the stream parameters were drawn for a pipeline of another length.
        """
        if stream_parameters is None:
            stream_parameters = [None] * len(self.pipeline)
        if len(stream_parameters) != len(self.pipeline):
            raise ValueError(
                f"Got the stream parameters of {len(stream_parameters)} transform(s) for a "
                f"pipeline of {len(self.pipeline)}."
            )
        for transform, parameters in zip(self.pipeline, stream_parameters, strict=True):
            model_gt_sample = transform(model_gt_sample, parameters)

        return model_gt_sample

    def sample_stream_parameters(self) -> tuple[StreamParameters | None, ...]:
        """Draw the augmentation of one stream, to hand to every frame of it.

        Returns:
            One entry per transform, None where the transform draws for every sample alone.
        """
        return tuple(transform.sample_stream_parameters() for transform in self.pipeline)

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
