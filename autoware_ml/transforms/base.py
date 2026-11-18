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
from collections import OrderedDict
from collections.abc import Sequence
import copy

import numpy as np

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample


class BaseTransform(ABC):
    """Abstract base class for ModelGTSample data transformations.

    A temporal model reads the frames of a stream one after another and needs some
    augmentations alike in every frame, the global rotation or the flip. Such a transform
    keeps the parameters it drew in an attribute of its own: :meth:`draw_parameters` draws
    and keeps them, and :meth:`transform` applies the kept parameters whenever there are some
    and draws for the sample alone otherwise. ``TransformsCompose`` holds one copy of such a
    transform per stream, whose parameters are drawn once and kept for every frame of it.

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
        # Outcome of the probability kept for the frames of a stream, None to decide it for
        # every sample alone.
        self._stream_apply: bool | None = None

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

        # 2. Check probability (skip if not triggered), decided once for the frames of a stream
        apply = self._should_apply() if self._stream_apply is None else self._stream_apply
        if not apply:
            return self.on_skip(model_gt_sample)

        # 3. Execute the actual transform
        return self.transform(model_gt_sample)

    def draw_parameters(self) -> bool:
        """Draw the parameters of the transform and keep them for the samples that follow.

        Override it in a transform that keeps its augmentation alike over the frames of a
        stream. The parameters are drawn from ``np.random`` before a frame is loaded, so they
        depend on no sample.

        Returns:
            Whether the transform keeps parameters, False for one that draws for every sample
            alone.
        """
        return False

    def draw_stream_parameters(self) -> bool:
        """Draw the augmentation of one stream and keep it: the probability, then the parameters.

        A transform that keeps no parameters is left deciding its probability and drawing for
        every sample alone.

        Returns:
            Whether the transform keeps the augmentation of the stream.
        """
        apply = self._should_apply()
        keeps_parameters = self.draw_parameters()
        self._stream_apply = apply if keeps_parameters else None
        return keeps_parameters

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

    For the frames of a stream it keeps the augmentation drawn for the stream, see
    :meth:`keep_stream_parameters`: the transforms that keep parameters are copied once per
    stream, the copies draw their parameters and hold them for every frame of the stream.

    Class Attributes:
        MAX_KEPT_STREAMS: Number of streams whose augmentation is kept, the least recently
            read one is dropped beyond it. A dropped stream that comes back is drawn from its
            seed again, with the same values.
    """

    MAX_KEPT_STREAMS = 256

    def __init__(self, pipeline: Sequence[BaseTransform]):
        """Initialize the transform pipeline.

        Args:
            pipeline: Ordered transforms applied to each sample.
        """
        self.pipeline = pipeline
        # Transforms of every kept stream by the seed of the stream, holding its augmentation
        self._stream_pipelines: OrderedDict[tuple[int, ...], Sequence[BaseTransform]] = (
            OrderedDict()
        )
        # Seed of the stream of the next sample, None for a sample augmented on its own
        self._stream_seed: tuple[int, ...] | None = None

    def __call__(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Apply each transform in order.

        Args:
            model_gt_sample: ModelGTSample instance passed through the configured transforms.

        Returns:
            Transformed ModelGTSample instance after all pipeline stages have been applied.
        """
        for transform in self.active_pipeline:
            model_gt_sample = transform(model_gt_sample)

        return model_gt_sample

    @property
    def active_pipeline(self) -> Sequence[BaseTransform]:
        """Transforms applied to the next sample, the ones of its stream when it has one."""
        if self._stream_seed is None:
            return self.pipeline
        return self._stream_pipelines[self._stream_seed]

    def keep_stream_parameters(self, stream_seed: Sequence[int]) -> None:
        """Augment the samples that follow as the frames of one stream.

        The first frame of a stream draws the augmentation of the stream, which is then kept
        and applied to every later frame of it, also when frames of other streams are read
        in between. The parameters are drawn from ``np.random`` seeded by the stream, so
        every dataloader worker and every rank draws the same values for one stream.

        Args:
            stream_seed: Integers naming the stream, the same for every frame of it.
        """
        stream_seed = tuple(stream_seed)
        if stream_seed not in self._stream_pipelines:
            self._stream_pipelines[stream_seed] = self._draw_stream_pipeline(stream_seed)
            if len(self._stream_pipelines) > self.MAX_KEPT_STREAMS:
                self._stream_pipelines.popitem(last=False)
        self._stream_pipelines.move_to_end(stream_seed)
        self._stream_seed = stream_seed

    def clear_stream_parameters(self) -> None:
        """Augment the samples that follow on their own again, the kept streams stay kept."""
        self._stream_seed = None

    def _draw_stream_pipeline(self, stream_seed: tuple[int, ...]) -> Sequence[BaseTransform]:
        """Draw the augmentation of one stream into copies of the transforms that keep it.

        A copy shares the settings of its transform and holds the parameters of the stream.
        A transform that keeps no parameters is used as it is. The random state of NumPy is
        restored after the draw.

        Args:
            stream_seed: Integers naming the stream, the seed of its draw.

        Returns:
            The transforms to apply to the frames of the stream, in pipeline order.
        """
        state = np.random.get_state()
        np.random.seed(list(stream_seed))
        try:
            stream_pipeline = []
            for transform in self.pipeline:
                stream_transform = copy.copy(transform)
                keeps_parameters = stream_transform.draw_stream_parameters()
                stream_pipeline.append(stream_transform if keeps_parameters else transform)
        finally:
            np.random.set_state(state)
        return stream_pipeline

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
