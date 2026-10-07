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

"""Visualization-neutral scene event definitions.

Task-specific visualization helpers emit these event dataclasses, and concrete
backends such as Rerun translate them into backend-specific logging calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float32]
IntArray: TypeAlias = npt.NDArray[np.int64]
ColorArray: TypeAlias = npt.NDArray[np.uint8]
ImageArray: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.float32]
Radius: TypeAlias = FloatArray | float


@dataclass(frozen=True)
class AnnotationInfo:
    """Describe one semantic class for backend legends."""

    id: int
    label: str
    color: tuple[int, int, int, int]


@dataclass(frozen=True)
class AnnotationContextEvent:
    """Log a semantic annotation context for descendant entities."""

    path: str
    annotations: list[AnnotationInfo]


@dataclass(frozen=True)
class ImageEvent:
    """Log a 2D image."""

    path: str
    image: ImageArray


@dataclass(frozen=True)
class PointCloud3DEvent:
    """Log a 3D point cloud."""

    path: str
    positions: FloatArray
    colors: ColorArray | None = None
    labels: list[str] | None = None
    radii: Radius | None = None
    class_ids: IntArray | None = None


@dataclass(frozen=True)
class Points2DEvent:
    """Log 2D points, typically as image overlays."""

    path: str
    positions: FloatArray
    colors: ColorArray | None = None
    labels: list[str] | None = None
    radii: Radius | None = None
    class_ids: IntArray | None = None


@dataclass(frozen=True)
class LineStrips2DEvent:
    """Log one or more 2D polylines, typically projected 3D box edges."""

    path: str
    strips: list[FloatArray]
    colors: ColorArray | None = None
    labels: list[str] | None = None
    radii: Radius | None = None
    class_ids: IntArray | None = None


@dataclass(frozen=True)
class Boxes3DEvent:
    """Log 3D boxes in metric space."""

    path: str
    centers: FloatArray
    sizes: FloatArray
    yaws: FloatArray
    colors: ColorArray | None = None
    labels: list[str] | None = None
    radii: Radius | None = None
    class_ids: IntArray | None = None


@dataclass(frozen=True)
class Transform3DEvent:
    """Log a rigid child-from-parent transform."""

    path: str
    translation: FloatArray
    rotation_matrix: FloatArray


@dataclass(frozen=True)
class PinholeEvent:
    """Log pinhole camera intrinsics."""

    path: str
    image_from_camera: FloatArray
    resolution: tuple[int, int]


@dataclass(frozen=True)
class ScalarEvent:
    """Log a scalar value."""

    path: str
    value: float


@dataclass(frozen=True)
class TextEvent:
    """Log human-readable text."""

    path: str
    text: str
    level: Literal["TRACE", "DEBUG", "INFO", "WARN", "ERROR"] = "INFO"


@dataclass(frozen=True)
class ClearEvent:
    """Clear current and historical components from an entity path."""

    path: str
    recursive: bool = True


@dataclass(frozen=True)
class ViewOverride:
    """Describe backend-neutral per-entity behavior inside one view."""

    path: str
    visible: bool | None = None
    show_labels: bool | None = None
    series_name: str | None = None
    marker_size: float | None = None


@dataclass(frozen=True)
class ViewSpec:
    """Describe one backend-neutral spatial, plot, or text view."""

    kind: Literal["spatial3d", "spatial2d", "time_series", "text_log"]
    name: str
    origin: str
    contents: tuple[str, ...]
    overrides: tuple[ViewOverride, ...] = ()
    y_range: tuple[float, float] | None = None
    visible_time_range: tuple[int, int] | None = None
    timeline: str | None = None


@dataclass(frozen=True)
class LayoutGroup:
    """Compose views into a backend-neutral row, column, or tab group."""

    kind: Literal["horizontal", "vertical", "tabs"]
    children: tuple[ViewSpec | LayoutGroup, ...]
    name: str | None = None
    shares: tuple[float, ...] | None = None
    active: int | None = None


@dataclass(frozen=True)
class BlueprintEvent:
    """Request a viewer layout without exposing a concrete backend SDK."""

    layout: ViewSpec | LayoutGroup
    blueprint_panel_expanded: bool | None = None
    selection_panel_expanded: bool | None = None
    time_panel_expanded: bool | None = None
    auto_layout: bool | None = None
    auto_views: bool = False
    make_active: bool | None = None
    make_default: bool | None = None


VisualizationEvent: TypeAlias = (
    AnnotationContextEvent
    | ImageEvent
    | PointCloud3DEvent
    | Points2DEvent
    | LineStrips2DEvent
    | Boxes3DEvent
    | Transform3DEvent
    | PinholeEvent
    | ScalarEvent
    | TextEvent
    | ClearEvent
    | BlueprintEvent
)
