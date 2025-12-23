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


class BaseTransform(ABC):
    """Abstract base class for dict-to-dict data transformations."""

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to transform() to support callables in pipelines."""
        return self.transform(input_dict)

    @abstractmethod
    def transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process input dictionary and return updated mapping."""
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
