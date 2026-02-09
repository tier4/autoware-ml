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

"""Augmentation module for data transformations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseTransform(ABC):
    """Abstract base class for dict-to-dict data transformations.

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

    p: float | None = None  # None means always run (no probability)
    _required_keys: List[str] = []
    _optional_keys: List[str] = []

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transform with probability and key validation.

        Order of operations:
            1. Validate required keys (raises KeyError if any missing)
            2. Handle optional keys (call apply_defaults if any missing)
            3. Check probability (skip if not triggered)
            4. Execute the actual transform
        """
        # 1. Validate required keys (raises error if any missing)
        self._validate_required_keys(input_dict)

        # 2. Handle optional keys (call apply_defaults if any missing)
        self._handle_optional_keys(input_dict)

        # 3. Check probability (skip if not triggered)
        if not self._should_apply():
            return self.on_skip(input_dict)

        # 4. Execute the actual transform
        return self.transform(input_dict)

    def _validate_required_keys(self, input_dict: Dict[str, Any]) -> None:
        """Raise KeyError if any required key is missing."""
        for key in self._required_keys:
            if key not in input_dict:
                raise KeyError(f"{self.__class__.__name__}: Missing required key '{key}'")

    def _handle_optional_keys(self, input_dict: Dict[str, Any]) -> None:
        """Check optional keys, call apply_defaults if any missing."""
        missing = [key for key in self._optional_keys if key not in input_dict]
        if missing:
            self.apply_defaults(input_dict)

    def _should_apply(self) -> bool:
        """Determine if transform should be applied based on probability.

        Returns:
            True if transform should be applied, False to skip.
        """
        if self.p is None:
            return True  # Always apply
        if self.p <= 0.0:
            return False  # Never apply
        if self.p >= 1.0:
            return True  # Always apply
        return np.random.rand() < self.p

    def apply_defaults(self, input_dict: Dict[str, Any]) -> None:
        """Set default values for missing optional keys. Override in subclasses.

        Base implementation raises error - subclasses with optional keys MUST override.
        Classes with no optional keys don't need to override (empty list never triggers).

        Args:
            input_dict: The input dictionary to modify in-place with default values.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}: Missing optional keys but apply_defaults() not implemented"
        )

    def on_skip(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Called when transform is skipped due to probability.

        Override for custom behavior when transform is skipped.
        Default implementation returns input unchanged.

        Args:
            input_dict: The input dictionary.

        Returns:
            The (possibly modified) input dictionary.
        """
        return input_dict

    @abstractmethod
    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process input dictionary and return updated mapping.

        Args:
            input_dict: Dictionary with required keys present, optional keys
                        populated by apply_defaults() if they were missing.

        Returns:
            Updated dictionary (may be the same object modified in-place).
        """
        raise NotImplementedError


class TransformsCompose:
    """Apply a sequence of transforms in order."""

    def __init__(self, pipeline: Optional[List["BaseTransform"]] = None):
        """Initialize the transform pipeline."""
        self.pipeline = pipeline or []

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply each transform sequentially, merging updates."""
        for transform in self.pipeline:
            input_dict |= transform(input_dict)

        return input_dict

    def __repr__(self) -> str:
        """Returns a formatted string representation of the composition."""
        if not self.pipeline:
            return f"{self.__class__.__name__}(pipeline=[])"

        format_string = [f"{self.__class__.__name__}("]
        for i, transform in enumerate(self.pipeline):
            format_string.append(f"  ({i}): {transform}")
        format_string.append(")")
        return "\n".join(format_string)
