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

"""Label taxonomies of a database.

A vocabulary maps the raw label names of a dataset family onto fine label names, the finest
distinction the corpus supports, and names the raw labels that are outside every level. A raw
label the vocabulary does not list is an error, so a new category of a corpus is discovered
instead of being ignored. A taxonomy selects the classes trained at one level of granularity
and folds every fine name onto one of them or drops it. Boxes are baked with the fine name and
the class index of the level when the record table is generated, and masks are resolved
through the same objects when they are loaded, so one definition decides the labels of both
heads.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from autoware_ml.types.metrics import AgentKind


class LabelVocabulary:
    """Raw label names of a dataset family mapped onto fine label names."""

    def __init__(self, name_mapping: Mapping[str, str | None]) -> None:
        """
        Initialize the vocabulary.

        Args:
          name_mapping: Raw label name to fine label name, None for a raw label the corpora
            carry that is outside every level. Every raw label of the corpora needs an entry.
        """

        for raw_name, fine_name in name_mapping.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(f"Raw label names must be non-empty strings, got {raw_name!r}.")
            if fine_name is not None and (not isinstance(fine_name, str) or not fine_name):
                raise ValueError(
                    f"Fine label names must be non-empty strings or None, got {fine_name!r} "
                    f"for raw label {raw_name!r}."
                )
        fine_names = {fine_name for fine_name in name_mapping.values() if fine_name is not None}
        if not fine_names:
            raise ValueError("A label vocabulary requires at least one fine label name.")
        self._name_mapping = dict(name_mapping)
        self._fine_names = tuple(sorted(fine_names))

    @property
    def fine_names(self) -> tuple[str, ...]:
        """Fine label names of the vocabulary, sorted."""
        return self._fine_names

    @property
    def name_mapping(self) -> Mapping[str, str | None]:
        """Raw label name to fine label name, None for a raw label outside every level."""
        return self._name_mapping

    def unlisted(self, raw_names: Iterable[str]) -> list[str]:
        """
        Raw label names the vocabulary does not list.

        Args:
          raw_names: Raw label names of a corpus.

        Returns:
          list[str]: The unlisted names, sorted.
        """

        return sorted(set(raw_names) - set(self._name_mapping))

    def fine_name(self, raw_name: str) -> str | None:
        """
        Fine label name of a raw label name.

        Args:
          raw_name: Raw label name of the dataset.

        Returns:
          str | None: The fine label name, None for a raw label outside every level.
        """

        if raw_name not in self._name_mapping:
            raise KeyError(
                f"Raw label name {raw_name!r} is not listed in the vocabulary. List it with "
                "its fine label name, or with null when it is outside every level."
            )
        return self._name_mapping[raw_name]

    def __str__(self) -> str:
        """Canonical string form, the input of the database hash."""
        entries = ", ".join(
            f"{raw_name}: {fine_name}" for raw_name, fine_name in sorted(self._name_mapping.items())
        )
        return f"{self.__class__.__name__}({entries})"

    def __eq__(self, other: object) -> bool:
        """Compare two vocabularies by their mapping."""
        return type(self) is type(other) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the vocabulary by its string form."""
        return hash(str(self))


class LabelTaxonomy:
    """Classes trained at one level of granularity over a vocabulary."""

    def __init__(
        self,
        vocabulary: LabelVocabulary,
        class_names: Sequence[str],
        coarsening: Mapping[str, str | None],
        ignore_index: int,
        class_groups: Mapping[str, Sequence[str]],
    ) -> None:
        """
        Initialize the taxonomy.

        Args:
          vocabulary: Raw label names mapped onto fine label names.
          class_names: Classes of the level, in index order.
          coarsening: Fine label name to class name, null for a fine label the level drops.
            Every fine name of the vocabulary needs an entry. A class no fine label coarsens
            to is a placeholder the level trains without data.
          ignore_index: Label index of a label outside the classes of the level.
          class_groups: Behaviour groups the metrics report besides the classes, group name to
            its classes. Every class belongs to exactly one group.
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
        coarsened_names = set(coarsening)
        if coarsened_names != fine_names:
            raise ValueError(
                "The coarsening must cover exactly the fine names of the vocabulary, missing "
                f"{sorted(fine_names - coarsened_names)}, unknown "
                f"{sorted(coarsened_names - fine_names)}."
            )
        class_set = set(class_names)
        for fine_name, class_name in coarsening.items():
            if class_name is not None and class_name not in class_set:
                raise ValueError(
                    f"Fine label {fine_name!r} coarsens to {class_name!r}, which is not a class "
                    f"of the level {list(class_names)}."
                )
        grouped = [name for members in class_groups.values() for name in members]
        if sorted(grouped) != sorted(class_names):
            raise ValueError(
                "Every class must belong to exactly one class group, the groups list "
                f"{sorted(grouped)} for the classes {sorted(class_names)}."
            )

        self._vocabulary = vocabulary
        self._class_names = tuple(class_names)
        self._coarsening = dict(coarsening)
        self._ignore_index = ignore_index
        self._class_groups = {name: tuple(members) for name, members in class_groups.items()}
        self._class_indices = {name: index for index, name in enumerate(self._class_names)}

    @staticmethod
    def _validate_class_table(
        table: Mapping[str, object], class_names: Sequence[str], table_name: str
    ) -> None:
        """
        Check that a class keyed table has exactly one entry per class of the level.

        Args:
          table: Class name to value.
          class_names: Classes of the level.
          table_name: Name of the table for the error message.
        """

        if set(table) != set(class_names):
            raise ValueError(
                f"{table_name} must have one entry per class, missing "
                f"{sorted(set(class_names) - set(table))}, unknown "
                f"{sorted(set(table) - set(class_names))}."
            )

    @property
    def vocabulary(self) -> LabelVocabulary:
        """Raw label names mapped onto fine label names."""
        return self._vocabulary

    @property
    def class_names(self) -> tuple[str, ...]:
        """Classes of the level, in index order."""
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Number of classes of the level."""
        return len(self._class_names)

    @property
    def ignore_index(self) -> int:
        """Label index of a label outside the classes of the level."""
        return self._ignore_index

    @property
    def coarsening(self) -> Mapping[str, str | None]:
        """Fine label name to class name, None for a dropped fine label."""
        return self._coarsening

    @property
    def class_groups(self) -> Mapping[str, tuple[str, ...]]:
        """Behaviour groups the metrics report, group name to its classes."""
        return self._class_groups

    def fine_name(self, raw_name: str) -> str | None:
        """
        Fine label name of a raw label name.

        Args:
          raw_name: Raw label name of the dataset, listed in the vocabulary.

        Returns:
          str | None: The fine label name, None for a raw label outside every level.
        """

        return self._vocabulary.fine_name(raw_name)

    def class_name(self, fine_name: str | None) -> str | None:
        """
        Class of the level a fine label name coarsens to.

        Args:
          fine_name: Fine label name of the vocabulary, None for a label outside every level.

        Returns:
          str | None: The class name, None for a label the level drops.
        """

        if fine_name is None:
            return None
        if fine_name not in self._coarsening:
            raise KeyError(f"{fine_name!r} is not a fine label name of the vocabulary.")
        return self._coarsening[fine_name]

    def class_index(self, fine_name: str | None) -> int:
        """
        Label index of a fine label name at the level.

        Args:
          fine_name: Fine label name of the vocabulary, None for a label outside every level.

        Returns:
          int: The class index, the ignore index for a label the level drops.
        """

        class_name = self.class_name(fine_name)
        if class_name is None:
            return self._ignore_index
        return self._class_indices[class_name]

    def resolve_index(self, raw_name: str) -> int:
        """
        Label index of a raw label name at the level.

        Args:
          raw_name: Raw label name of the dataset, listed in the vocabulary.

        Returns:
          int: The class index, the ignore index for a label outside the level.
        """

        return self.class_index(self.fine_name(raw_name))

    def __str__(self) -> str:
        """Canonical string form, the input of the database hash."""
        coarsening = ", ".join(
            f"{fine_name}: {class_name}"
            for fine_name, class_name in sorted(self._coarsening.items())
        )
        return (
            f"{self.__class__.__name__}(class_names={list(self._class_names)}, "
            f"ignore_index={self._ignore_index}, coarsening=({coarsening}), "
            f"vocabulary={self._vocabulary})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two taxonomies by their definition."""
        return type(self) is type(other) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the taxonomy by its string form."""
        return hash(str(self))


class DetectionTaxonomy(LabelTaxonomy):
    """
    Detection classes of one level together with the class keyed evaluation tables of the
    detection metrics. The tables are validated against the class list here and read by the
    dataset configs, they do not enter the database hash because they do not change the table.
    """

    def __init__(
        self,
        vocabulary: LabelVocabulary,
        class_names: Sequence[str],
        coarsening: Mapping[str, str | None],
        ignore_index: int,
        class_groups: Mapping[str, Sequence[str]],
        eval_range: Mapping[str, float],
        collision_kinds: Mapping[str, str],
        living_speeds: Mapping[str, float],
        partial_detection_classes: Sequence[str],
        heatmap_pooling_classes: Sequence[str],
    ) -> None:
        """
        Initialize the detection taxonomy.

        Args:
          vocabulary: Raw label names mapped onto fine label names.
          class_names: Classes of the level, in index order.
          coarsening: Fine label name to class name, null for a dropped fine label.
          ignore_index: Label index of a label outside the classes of the level.
          class_groups: Behaviour groups the metrics report, group name to its classes.
          eval_range: Range in meters up to which every class is evaluated.
          collision_kinds: Reachable set kind of every class in the collision metrics.
          living_speeds: Run speed in meters per second of every living class.
          partial_detection_classes: Classes the partial detection score of the joint metrics
            reports, the small objects where a few correctly segmented points already matter.
          heatmap_pooling_classes: Classes whose dense heatmap the detection heads pool before
            the proposal selection, the vehicles large enough to raise several peaks.
        """

        super().__init__(vocabulary, class_names, coarsening, ignore_index, class_groups)
        unknown_partial = sorted(set(partial_detection_classes) - set(class_names))
        if unknown_partial or not len(partial_detection_classes):
            raise ValueError(
                "partial_detection_classes must name at least one class of the level, unknown "
                f"{unknown_partial}."
            )
        self._validate_class_table(eval_range, class_names, "eval_range")
        self._validate_class_table(collision_kinds, class_names, "collision_kinds")
        unknown_kinds = sorted(set(collision_kinds.values()) - set(AgentKind))
        if unknown_kinds:
            raise ValueError(
                f"Unknown collision kinds {unknown_kinds}, valid kinds are "
                f"{[kind.value for kind in AgentKind]}."
            )
        living_classes = {
            name for name, kind in collision_kinds.items() if kind == AgentKind.LIVING
        }
        if set(living_speeds) != living_classes:
            raise ValueError(
                "living_speeds must list exactly the living classes, missing "
                f"{sorted(living_classes - set(living_speeds))}, unknown "
                f"{sorted(set(living_speeds) - living_classes)}."
            )
        self._eval_range = {name: float(value) for name, value in eval_range.items()}
        self._collision_kinds = {name: AgentKind(kind) for name, kind in collision_kinds.items()}
        self._living_speeds = {name: float(speed) for name, speed in living_speeds.items()}
        self._partial_detection_classes = tuple(partial_detection_classes)
        unknown_pooling = sorted(set(heatmap_pooling_classes) - set(class_names))
        if unknown_pooling:
            raise ValueError(
                f"heatmap_pooling_classes must name classes of the level, unknown {unknown_pooling}."
            )
        self._heatmap_pooling_classes = tuple(heatmap_pooling_classes)

    @property
    def partial_detection_classes(self) -> tuple[str, ...]:
        """Classes the partial detection score of the joint metrics reports."""
        return self._partial_detection_classes

    @property
    def heatmap_pooling_classes(self) -> tuple[str, ...]:
        """Classes whose dense heatmap the detection heads pool before the proposal selection."""
        return self._heatmap_pooling_classes

    @property
    def eval_range(self) -> Mapping[str, float]:
        """Range in meters up to which every class is evaluated."""
        return self._eval_range

    @property
    def collision_kinds(self) -> Mapping[str, AgentKind]:
        """Reachable set kind of every class in the collision metrics."""
        return self._collision_kinds

    @property
    def living_speeds(self) -> Mapping[str, float]:
        """Run speed in meters per second of every living class."""
        return self._living_speeds


class SegmentationTaxonomy(LabelTaxonomy):
    """Segmentation classes of one level together with the behaviour groups of the metrics."""


class DatabaseTaxonomy:
    """Taxonomies of the label spaces a database describes."""

    def __init__(
        self, detection3d: DetectionTaxonomy, segmentation3d: SegmentationTaxonomy
    ) -> None:
        """
        Initialize the database taxonomy.

        Args:
          detection3d: Taxonomy the box labels are baked with.
          segmentation3d: Taxonomy the semantic mask categories are resolved with.
        """

        if not isinstance(detection3d, DetectionTaxonomy):
            raise TypeError(
                f"detection3d must be a DetectionTaxonomy, got {type(detection3d).__name__}."
            )
        if not isinstance(segmentation3d, SegmentationTaxonomy):
            raise TypeError(
                "segmentation3d must be a SegmentationTaxonomy, got "
                f"{type(segmentation3d).__name__}."
            )
        self._detection3d = detection3d
        self._segmentation3d = segmentation3d

    @property
    def detection3d(self) -> DetectionTaxonomy:
        """Taxonomy the box labels are baked with."""
        return self._detection3d

    @property
    def segmentation3d(self) -> SegmentationTaxonomy:
        """Taxonomy the semantic mask categories are resolved with."""
        return self._segmentation3d

    def __str__(self) -> str:
        """Canonical string form, the input of the database hash."""
        return (
            f"{self.__class__.__name__}(detection3d={self._detection3d}, "
            f"segmentation3d={self._segmentation3d})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two database taxonomies by their definition."""
        return type(self) is type(other) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the database taxonomy by its string form."""
        return hash(str(self))
