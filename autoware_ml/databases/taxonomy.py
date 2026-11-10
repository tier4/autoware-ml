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

"""Label vocabularies of a database and label taxonomies of a model.

A vocabulary maps the raw label names of a dataset family onto fine label names, the finest
distinction the corpus supports, and names the raw labels that are outside every taxonomy. A raw
label the vocabulary does not list is an error, so a new category of a corpus is discovered
instead of being ignored. The record table stores the fine name of every box. A taxonomy holds
the classes a model trains and folds every fine name onto one of them or drops it. The tasks
resolve the fine names of the boxes and the raw categories of the masks through it when a
sample is loaded, so the record table serves every taxonomy.
"""

from __future__ import annotations

from types import MappingProxyType
from collections.abc import Iterable, Mapping, Sequence


class LabelVocabulary:
    """Raw label names of a dataset family mapped onto fine label names."""

    def __init__(self, class_renaming: Mapping[str, str | None]) -> None:
        """
        Initialize the vocabulary.

        Args:
          class_renaming: Raw label name to fine label name, None for a raw label the corpora
            carry that is outside every taxonomy. Every raw label of the corpora needs an entry.
        """

        for raw_name, fine_name in class_renaming.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(f"Raw label names must be non-empty strings, got {raw_name!r}.")
            if fine_name is not None and (not isinstance(fine_name, str) or not fine_name):
                raise ValueError(
                    f"Fine label names must be non-empty strings or None, got {fine_name!r} "
                    f"for raw label {raw_name!r}."
                )
        fine_names = {fine_name for fine_name in class_renaming.values() if fine_name is not None}
        if not fine_names:
            raise ValueError("A label vocabulary requires at least one fine label name.")
        self._class_renaming = dict(class_renaming)
        self._fine_names = tuple(sorted(fine_names))

    @property
    def fine_names(self) -> tuple[str, ...]:
        """Fine label names of the vocabulary, sorted."""
        return self._fine_names

    @property
    def class_renaming(self) -> Mapping[str, str | None]:
        """Raw label name to fine label name, None for a raw label outside every taxonomy."""
        return MappingProxyType(self._class_renaming)

    def unlisted(self, raw_names: Iterable[str]) -> list[str]:
        """
        Raw label names the vocabulary does not list.

        Args:
          raw_names: Raw label names of a corpus.

        Returns:
          list[str]: The unlisted names, sorted.
        """

        return sorted(set(raw_names) - set(self._class_renaming))

    def fine_name(self, raw_name: str) -> str | None:
        """
        Fine label name of a raw label name.

        Args:
          raw_name: Raw label name of the dataset.

        Returns:
          str | None: The fine label name, None for a raw label outside every taxonomy.
        """

        if raw_name not in self._class_renaming:
            raise KeyError(
                f"Raw label name {raw_name!r} is not listed in the vocabulary. List it with "
                "its fine label name, or with null when it is outside every taxonomy."
            )
        return self._class_renaming[raw_name]

    def __str__(self) -> str:
        """Canonical string form, the input of the database hash."""
        entries = ", ".join(
            f"{raw_name}: {fine_name}"
            for raw_name, fine_name in sorted(self._class_renaming.items())
        )
        return f"{self.__class__.__name__}({entries})"

    def __eq__(self, other: object) -> bool:
        """Compare two vocabularies by their mapping."""
        return type(self) is type(other) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the vocabulary by its string form."""
        return hash(str(self))


class LabelTaxonomy:
    """Classes a model trains over a vocabulary."""

    def __init__(
        self,
        vocabulary: LabelVocabulary,
        class_names: Sequence[str],
        class_mapping: Mapping[str, str | None],
        ignore_index: int,
    ) -> None:
        """
        Initialize the taxonomy.

        Args:
          vocabulary: Raw label names mapped onto fine label names.
          class_names: Classes the model trains, in index order.
          class_mapping: Fine label name to class name, null for a fine label the model does not
            train. Every fine name of the vocabulary needs an entry. A class no fine label
            coarsens to is a placeholder the model trains without data.
          ignore_index: Label index of a label outside the trained classes.
        """

        if not len(class_names):
            raise ValueError("A label taxonomy requires at least one class name.")
        if len(set(class_names)) != len(class_names):
            raise ValueError(f"Class names must be unique, got {list(class_names)}.")
        for class_name in class_names:
            if not isinstance(class_name, str) or not class_name:
                raise ValueError(f"Class names must be non-empty strings, got {class_name!r}.")
        if 0 <= ignore_index < len(class_names):
            raise ValueError(
                f"The ignore index {ignore_index} collides with the class indices "
                f"0..{len(class_names) - 1}."
            )

        fine_names = set(vocabulary.fine_names)
        coarsened_names = set(class_mapping)
        if coarsened_names != fine_names:
            raise ValueError(
                "The class_mapping must cover exactly the fine names of the vocabulary, missing "
                f"{sorted(fine_names - coarsened_names)}, unknown "
                f"{sorted(coarsened_names - fine_names)}."
            )
        class_set = set(class_names)
        for fine_name, class_name in class_mapping.items():
            if class_name is not None and class_name not in class_set:
                raise ValueError(
                    f"Fine label {fine_name!r} coarsens to {class_name!r}, which is not a trained "
                    f"class of {list(class_names)}."
                )

        self._vocabulary = vocabulary
        self._class_names = tuple(class_names)
        self._class_mapping = dict(class_mapping)
        self._ignore_index = ignore_index
        self._class_indices = {name: index for index, name in enumerate(self._class_names)}

    @property
    def vocabulary(self) -> LabelVocabulary:
        """Raw label names mapped onto fine label names."""
        return self._vocabulary

    @property
    def class_names(self) -> tuple[str, ...]:
        """Trained classes, in index order."""
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Number of trained classes."""
        return len(self._class_names)

    @property
    def ignore_index(self) -> int:
        """Label index of a label outside the trained classes."""
        return self._ignore_index

    @property
    def class_mapping(self) -> Mapping[str, str | None]:
        """Fine label name to class name, None for a fine label the model does not train."""
        return MappingProxyType(self._class_mapping)

    def fine_name(self, raw_name: str) -> str | None:
        """
        Fine label name of a raw label name.

        Args:
          raw_name: Raw label name of the dataset, listed in the vocabulary.

        Returns:
          str | None: The fine label name, None for a raw label outside every taxonomy.
        """

        return self._vocabulary.fine_name(raw_name)

    def class_name(self, fine_name: str | None) -> str | None:
        """
        Trained class a fine label name coarsens to.

        Args:
          fine_name: Fine label name of the vocabulary, None for a label outside every taxonomy.

        Returns:
          str | None: The class name, None for a label the model does not train.
        """

        if fine_name is None:
            return None
        if fine_name not in self._class_mapping:
            raise KeyError(f"{fine_name!r} is not a fine label name of the vocabulary.")
        return self._class_mapping[fine_name]

    def class_index(self, fine_name: str | None) -> int:
        """
        Label index of a fine label name.

        Args:
          fine_name: Fine label name of the vocabulary, None for a label outside every taxonomy.

        Returns:
          int: The class index, the ignore index for a label the model does not train.
        """

        class_name = self.class_name(fine_name)
        if class_name is None:
            return self._ignore_index
        return self._class_indices[class_name]

    def resolve_index(self, raw_name: str) -> int:
        """
        Label index of a raw label name.

        Args:
          raw_name: Raw label name of the dataset, listed in the vocabulary.

        Returns:
          int: The class index, the ignore index for a label the model does not train.
        """

        return self.class_index(self.fine_name(raw_name))

    def __str__(self) -> str:
        """Canonical string form, which identifies the taxonomy."""
        class_mapping = ", ".join(
            f"{fine_name}: {class_name}"
            for fine_name, class_name in sorted(self._class_mapping.items())
        )
        return (
            f"{self.__class__.__name__}(class_names={list(self._class_names)}, "
            f"ignore_index={self._ignore_index}, class_mapping=({class_mapping}), "
            f"vocabulary={self._vocabulary})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two taxonomies by their definition."""
        return type(self) is type(other) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the taxonomy by its string form."""
        return hash(str(self))
