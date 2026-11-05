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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float32, Int64, UInt8
import numpy as np


@dataclass(frozen=True)
class VisualizationEvent:
    """Base of every event a task adapter emits and a backend logs."""


@dataclass(frozen=True)
class AnnotationInfo:
    """Describe one semantic class for backend legends."""

    id: int
    label: str
    color: tuple[int, int, int, int]


@dataclass(frozen=True)
class AnnotationContextEvent(VisualizationEvent):
    """Log a semantic annotation context for descendant entities."""

    path: str
    annotations: Sequence[AnnotationInfo]


@dataclass(frozen=True)
class ImageEvent(VisualizationEvent):
    """Log a 2D image."""

    path: str
    image: UInt8[np.ndarray, "height width channels"]


@dataclass(frozen=True)
class PointCloud3DEvent(VisualizationEvent):
    """Log a 3D point cloud."""

    path: str
    positions: Float32[np.ndarray, "num_points 3"]
    colors: UInt8[np.ndarray, "num_points 4"] | None = None
    labels: Sequence[str] | None = None
    radii: Float32[np.ndarray, " num_points"] | float | None = None
    class_ids: Int64[np.ndarray, " num_points"] | None = None


@dataclass(frozen=True)
class Points2DEvent(VisualizationEvent):
    """Log 2D points, typically as image overlays."""

    path: str
    positions: Float32[np.ndarray, "num_points 2"]
    colors: UInt8[np.ndarray, "num_points 4"] | None = None
    labels: Sequence[str] | None = None
    radii: Float32[np.ndarray, " num_points"] | float | None = None
    class_ids: Int64[np.ndarray, " num_points"] | None = None


@dataclass(frozen=True)
class LineStrips2DEvent(VisualizationEvent):
    """Log one or more 2D polylines, typically projected 3D box edges."""

    path: str
    strips: Sequence[Float32[np.ndarray, "num_vertices 2"]]
    colors: UInt8[np.ndarray, "num_strips 4"] | None = None
    labels: Sequence[str] | None = None
    radii: Float32[np.ndarray, " num_strips"] | float | None = None
    class_ids: Int64[np.ndarray, " num_strips"] | None = None


@dataclass(frozen=True)
class Boxes3DEvent(VisualizationEvent):
    """Log 3D boxes in metric space."""

    path: str
    centers: Float32[np.ndarray, "num_boxes 3"]
    sizes: Float32[np.ndarray, "num_boxes 3"]
    yaws: Float32[np.ndarray, " num_boxes"]
    colors: UInt8[np.ndarray, "num_boxes 4"] | None = None
    labels: Sequence[str] | None = None
    radii: Float32[np.ndarray, " num_boxes"] | float | None = None
    class_ids: Int64[np.ndarray, " num_boxes"] | None = None


@dataclass(frozen=True)
class Transform3DEvent(VisualizationEvent):
    """Log a rigid child-from-parent transform."""

    path: str
    translation: Float32[np.ndarray, " 3"]
    rotation_matrix: Float32[np.ndarray, "3 3"]


@dataclass(frozen=True)
class PinholeEvent(VisualizationEvent):
    """Log pinhole camera intrinsics."""

    path: str
    image_from_camera: Float32[np.ndarray, "3 3"]
    resolution: tuple[int, int]


@dataclass(frozen=True)
class ScalarEvent(VisualizationEvent):
    """Log a scalar value."""

    path: str
    value: float


@dataclass(frozen=True)
class TextEvent(VisualizationEvent):
    """Log human-readable text."""

    path: str
    text: str
    level: Literal["TRACE", "DEBUG", "INFO", "WARN", "ERROR"] = "INFO"


@dataclass(frozen=True)
class TextDocumentEvent(VisualizationEvent):
    """Log a standalone document for a dedicated text view.

    Unlike :class:`TextEvent`, which lands in the viewer's log stream, a
    document is meant to be read on screen next to the spatial views, so it
    can carry markdown for headings and emphasis.
    """

    path: str
    text: str
    markdown: bool = True


@dataclass(frozen=True)
class ClearEvent(VisualizationEvent):
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

    kind: Literal["spatial3d", "spatial2d", "time_series", "text_log", "text_document"]
    name: str
    origin: str
    contents: tuple[str, ...]
    overrides: tuple[ViewOverride, ...] = ()
    y_range: tuple[float, float] | None = None
    visible_time_range: tuple[int, int] | None = None
    timeline: str | None = None
    #: Optional stable identity. Backends derive the view id from it, so a
    #: re-published layout keeps the viewer state of the view, such as its 3D eye.
    identity: str | None = None


@dataclass(frozen=True)
class LayoutGroup:
    """Compose views into a backend-neutral row, column, or tab group."""

    kind: Literal["horizontal", "vertical", "tabs"]
    children: tuple[ViewSpec | LayoutGroup, ...]
    name: str | None = None
    shares: tuple[float, ...] | None = None
    active: int | None = None


@dataclass(frozen=True)
class BlueprintEvent(VisualizationEvent):
    """Request a viewer layout without exposing a concrete backend SDK."""

    layout: ViewSpec | LayoutGroup
    blueprint_panel_expanded: bool | None = None
    selection_panel_expanded: bool | None = None
    time_panel_expanded: bool | None = None
    auto_layout: bool | None = None
    auto_views: bool = False
    make_active: bool | None = None
    make_default: bool | None = None
