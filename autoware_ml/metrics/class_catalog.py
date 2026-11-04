"""Metric settings of the classes of a dataset family.

A class catalog holds one row per class name with the settings the metrics read for that class.
A model trains some of the classes of the catalog. The functions below select the rows of the
classes a model trains and build the tables the metrics take, so a trained class the catalog does
not list stops the run instead of being evaluated with made up settings. A group follows the
order the catalog first names it in, and lists its classes in catalog order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from autoware_ml.types.metrics import MotionModel


@dataclass(frozen=True)
class Detection3DClassProfile:
    """
    Metric settings of one detection class.

    Attributes:
      group: Behaviour group the grouped metrics fold the class into.
      max_distance: Distance in meters up to which the class is evaluated.
      motion_model: Motion model of the class in the collision metrics.
      max_speed: Max speed in meters per second, set for exactly the omnidirectional classes.
      partial_detection: Whether the partial detection score checks the boxes of the class
        against the semantic masks.
    """

    group: str
    max_distance: float
    motion_model: MotionModel
    max_speed: float | None = None
    partial_detection: bool = False

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.max_distance <= 0.0:
            raise ValueError(f"max_distance must be positive, got {self.max_distance}.")
        if (self.motion_model == MotionModel.OMNIDIRECTIONAL) != (self.max_speed is not None):
            raise ValueError(
                f"max_speed is set for exactly the {MotionModel.OMNIDIRECTIONAL.value} classes, got "
                f"{self.max_speed} for a {self.motion_model.value} class."
            )


@dataclass(frozen=True)
class Segmentation3DClassProfile:
    """
    Metric settings of one segmentation class.

    Attributes:
      group: Behaviour group the grouped metrics fold the class into.
    """

    group: str


def _rows(catalog: Mapping[str, Any], class_names: Sequence[str]) -> list[str]:
    """
    Check that the catalog lists every trained class.

    Args:
      catalog: Class name to its settings.
      class_names: Classes the model trains.

    Returns:
      list[str]: The trained classes in catalog order.
    """
    missing = [name for name in class_names if name not in catalog]
    if missing:
        raise ValueError(f"The class catalog has no row for the trained classes {missing}.")
    trained = set(class_names)
    return [name for name in catalog if name in trained]


def _detection3d_profiles(
    catalog: Mapping[str, Mapping[str, Any]], class_names: Sequence[str]
) -> dict[str, Detection3DClassProfile]:
    """Detection settings of the trained classes, in catalog order."""
    return {
        name: Detection3DClassProfile(
            **{**catalog[name], "motion_model": MotionModel(catalog[name]["motion_model"])}
        )
        for name in _rows(catalog, class_names)
    }


def _segmentation3d_profiles(
    catalog: Mapping[str, Mapping[str, Any]], class_names: Sequence[str]
) -> dict[str, Segmentation3DClassProfile]:
    """Segmentation settings of the trained classes, in catalog order."""
    return {
        name: Segmentation3DClassProfile(**catalog[name]) for name in _rows(catalog, class_names)
    }


def _group_table(groups: Mapping[str, str]) -> dict[str, list[str]]:
    """Group name to its classes, from class name to group name in catalog order."""
    table: dict[str, list[str]] = {}
    for name, group in groups.items():
        table.setdefault(group, []).append(name)
    return table


def detection3d_class_groups(
    catalog: Mapping[str, Mapping[str, Any]], class_names: Sequence[str]
) -> dict[str, list[str]]:
    """
    Behaviour groups of the trained detection classes.

    Args:
      catalog: Detection class name to its settings.
      class_names: Detection classes the model trains.

    Returns:
      dict[str, list[str]]: Group name to its trained classes.
    """
    profiles = _detection3d_profiles(catalog, class_names)
    return _group_table({name: profile.group for name, profile in profiles.items()})


def segmentation3d_class_groups(
    catalog: Mapping[str, Mapping[str, Any]], class_names: Sequence[str]
) -> dict[str, list[str]]:
    """
    Behaviour groups of the trained segmentation classes.

    Args:
      catalog: Segmentation class name to its settings.
      class_names: Segmentation classes the model trains.

    Returns:
      dict[str, list[str]]: Group name to its trained classes.
    """
    profiles = _segmentation3d_profiles(catalog, class_names)
    return _group_table({name: profile.group for name, profile in profiles.items()})


def max_distances(
    catalog: Mapping[str, Mapping[str, Any]], class_names: Sequence[str]
) -> dict[str, float]:
    """
    Evaluation distance of every trained detection class.

    Args:
      catalog: Detection class name to its settings.
      class_names: Detection classes the model trains.

    Returns:
      dict[str, float]: Class name to the distance in meters up to which it is evaluated.
    """
    profiles = _detection3d_profiles(catalog, class_names)
    return {name: profiles[name].max_distance for name in class_names}


def motion_models(
    catalog: Mapping[str, Mapping[str, Any]], class_names: Sequence[str]
) -> dict[str, MotionModel]:
    """
    Collision motion model of every trained detection class.

    Args:
      catalog: Detection class name to its settings.
      class_names: Detection classes the model trains.

    Returns:
      dict[str, MotionModel]: Class name to its motion model.
    """
    profiles = _detection3d_profiles(catalog, class_names)
    return {name: profiles[name].motion_model for name in class_names}


def max_speeds(
    catalog: Mapping[str, Mapping[str, Any]], class_names: Sequence[str]
) -> dict[str, float]:
    """
    Max speed of every trained omnidirectional detection class.

    Args:
      catalog: Detection class name to its settings.
      class_names: Detection classes the model trains.

    Returns:
      dict[str, float]: Omnidirectional class name to its max speed in meters per second.
    """
    profiles = _detection3d_profiles(catalog, class_names)
    return {
        name: profiles[name].max_speed
        for name in class_names
        if profiles[name].max_speed is not None
    }


def partial_detection_classes(
    catalog: Mapping[str, Mapping[str, Any]], class_names: Sequence[str]
) -> list[str]:
    """
    Trained detection classes the partial detection score checks.

    Args:
      catalog: Detection class name to its settings.
      class_names: Detection classes the model trains.

    Returns:
      list[str]: The checked classes in class order.
    """
    profiles = _detection3d_profiles(catalog, class_names)
    return [name for name in class_names if profiles[name].partial_detection]
