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

"""Tests for backend-neutral task layout declarations."""

from __future__ import annotations

import numpy as np

from autoware_ml.visualization.events import ImageEvent, LayoutGroup, ViewSpec
from autoware_ml.visualization.layouts import (
    SCENE_EXTENT_PATH,
    SYNCED_LEFT_IDENTITY,
    SYNCED_RIGHT_IDENTITY,
    build_calibration_blueprint,
    build_scene_blueprint,
    build_synced_scene_layouts,
)

_MULTI_PATHS = {
    "scene/ground_truth/segmentation",
    "scene/prediction/segmentation",
    "scene/prediction/entropy",
    "scene/prediction/probability",
    "scene/ground_truth/detections",
    "scene/prediction/detections",
    "scene/lidar/intensity",
    "scene/cameras/front",
    "scene/cameras/front/projected/ground_truth/segmentation",
    "scene/cameras/front/projected/prediction/detections",
    "scene/metrics/detection/precision",
    "scene/metrics/detection/true_positives",
}


def _views(layout: ViewSpec | LayoutGroup) -> list[ViewSpec]:
    """Collect the views of a layout in display order."""
    if isinstance(layout, ViewSpec):
        return [layout]
    return [view for child in layout.children for view in _views(child)]


def _scene_views(layout: ViewSpec | LayoutGroup) -> list[ViewSpec]:
    return [view for view in _views(layout) if view.kind == "spatial3d"]


def test_calibration_layout_is_expressed_only_as_neutral_specs() -> None:
    blueprint = build_calibration_blueprint(
        [
            ImageEvent(
                path="calibration_status/camera/image",
                image=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            ImageEvent(
                path="calibration_status/camera/fused",
                image=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
        ]
    )

    assert blueprint is not None
    assert isinstance(blueprint.layout, LayoutGroup)
    assert [child.name for child in blueprint.layout.children] == [
        "Raw camera",
        "Fused camera",
    ]


def test_scene_layout_keeps_multi_prediction_left_and_comparison_selectable() -> None:
    paths = {
        "scene/ground_truth/segmentation",
        "scene/prediction/segmentation",
        "scene/prediction/entropy",
        "scene/prediction/probability",
        "scene/ground_truth/detections",
        "scene/prediction/detections",
        "scene/lidar/intensity",
        "scene/cameras/front",
        "scene/cameras/front/projected/ground_truth/segmentation",
        "scene/cameras/front/projected/prediction/detections",
        "scene/metrics/detection/precision",
        "scene/metrics/detection/true_positives",
    }

    blueprint = build_scene_blueprint(
        paths,
        ["scene/cameras/front"],
        point_color_mode="semantic",
        camera_frustums_visible=False,
        timeline="frame",
    )

    assert blueprint is not None
    serialized = repr(blueprint)
    assert "Semantic" not in serialized
    assert "Multi comparison" in serialized
    assert "Prediction · Multi" in serialized
    assert "Normalized entropy" in serialized
    assert "Probability" in serialized
    assert "front · GT" in serialized
    assert "front · Prediction" in serialized
    assert "3D IoU quality" in serialized
    assert "Detection matches" in serialized
    assert blueprint.blueprint_panel_expanded is True
    assert [child.name for child in blueprint.layout.children] == [
        "Multi comparison",
        "Cameras",
    ]

    multi_layout = blueprint.layout.children[0]
    assert isinstance(multi_layout, LayoutGroup)
    camera_switch = multi_layout.children[0]
    assert isinstance(camera_switch, LayoutGroup)
    assert camera_switch.kind == "tabs"
    assert camera_switch.name == "Multi comparison"
    assert camera_switch.active == 0
    assert [child.name for child in camera_switch.children] == [
        "Camera projections OFF",
        "Camera projections ON",
    ]

    comparison = camera_switch.children[0]
    assert isinstance(comparison, LayoutGroup)
    assert comparison.kind == "horizontal"
    prediction = comparison.children[0]
    comparison_options = comparison.children[1]
    assert isinstance(prediction, ViewSpec)
    assert prediction.name == "Prediction · Multi"
    assert prediction.contents[:2] == (
        "scene/prediction/segmentation",
        "scene/prediction/detections",
    )
    assert isinstance(comparison_options, LayoutGroup)
    assert comparison_options.kind == "tabs"
    assert comparison_options.active == 0
    assert [child.name for child in comparison_options.children] == [
        "GT · Multi",
        "Intensity",
        "Normalized entropy",
        "Probability",
    ]
    assert "scene/lidar/intensity" not in prediction.contents

    camera_layout = blueprint.layout.children[1]
    assert isinstance(camera_layout, LayoutGroup)
    front_comparison = camera_layout.children[0]
    assert isinstance(front_comparison, LayoutGroup)
    assert [child.name for child in front_comparison.children] == [
        "front · Prediction",
        "front · GT",
    ]
    assert "show_labels=True" in serialized


def test_scene_layout_can_start_with_intensity_on_the_right() -> None:
    blueprint = build_scene_blueprint(
        {
            "scene/prediction/segmentation",
            "scene/ground_truth/segmentation",
            "scene/lidar/intensity",
            "scene/prediction/entropy",
            "scene/prediction/probability",
        },
        [],
        point_color_mode="intensity",
        camera_frustums_visible=False,
        timeline="frame",
    )

    assert blueprint is not None
    assert isinstance(blueprint.layout, LayoutGroup)
    comparison = blueprint.layout.children[0]
    assert isinstance(comparison, LayoutGroup)
    comparison_options = comparison.children[1]
    assert isinstance(comparison_options, LayoutGroup)
    assert comparison_options.active == 1
    assert comparison_options.children[1].name == "Intensity"


def test_scene_layout_camera_switch_respects_visible_initial_state() -> None:
    blueprint = build_scene_blueprint(
        {
            "scene/prediction/segmentation",
            "scene/prediction/detections",
            "scene/cameras/front",
        },
        ["scene/cameras/front"],
        point_color_mode="semantic",
        camera_frustums_visible=True,
        timeline="frame",
    )

    assert blueprint is not None
    assert isinstance(blueprint.layout, LayoutGroup)
    camera_switch = blueprint.layout.children[0]
    assert isinstance(camera_switch, LayoutGroup)
    assert camera_switch.kind == "tabs"
    assert camera_switch.name == "Multi comparison"
    assert camera_switch.active == 1


def test_scene_layout_uses_task_name_for_single_task_predictions() -> None:
    blueprint = build_scene_blueprint(
        {
            "scene/prediction/segmentation",
            "scene/ground_truth/segmentation",
        },
        [],
        point_color_mode="semantic",
        camera_frustums_visible=False,
        timeline="frame",
    )

    assert blueprint is not None
    assert "Prediction · Segmentation3D" in repr(blueprint)
    assert "Semantic" not in repr(blueprint)


def test_detection_prediction_uses_solid_geometry_instead_of_intensity() -> None:
    blueprint = build_scene_blueprint(
        {
            "scene/prediction/detections",
            "scene/ground_truth/detections",
            "scene/lidar/solid",
            "scene/lidar/intensity",
        },
        [],
        point_color_mode="semantic",
        camera_frustums_visible=False,
        timeline="frame",
    )

    assert blueprint is not None
    assert isinstance(blueprint.layout, LayoutGroup)
    comparison = blueprint.layout.children[0]
    assert isinstance(comparison, LayoutGroup)
    prediction = comparison.children[0]
    assert isinstance(prediction, ViewSpec)
    assert prediction.name == "Prediction · Detection3D"
    assert prediction.contents == (
        "scene/lidar/solid",
        "scene/prediction/detections",
    )
    assert "scene/lidar/intensity" not in prediction.contents


def test_unknown_scene_adapter_gets_a_generic_view() -> None:
    blueprint = build_scene_blueprint(
        {"scene/new_task/result"},
        [],
        point_color_mode="solid",
        camera_frustums_visible=False,
        timeline="frame",
    )

    assert blueprint is not None
    assert isinstance(blueprint.layout, LayoutGroup)
    view = blueprint.layout.children[0]
    assert isinstance(view, ViewSpec)
    assert view.name == "Scene"
    assert view.contents == ("scene/new_task/result",)


def test_scene_views_list_the_shared_extent_when_it_is_logged() -> None:
    blueprint = build_scene_blueprint(
        _MULTI_PATHS | {SCENE_EXTENT_PATH},
        ["scene/cameras/front"],
        point_color_mode="semantic",
        camera_frustums_visible=False,
        timeline="frame",
    )

    assert blueprint is not None
    scene_views = _scene_views(blueprint.layout)
    assert scene_views
    assert all(SCENE_EXTENT_PATH in view.contents for view in scene_views)

    without_extent = build_scene_blueprint(
        _MULTI_PATHS,
        ["scene/cameras/front"],
        point_color_mode="semantic",
        camera_frustums_visible=False,
        timeline="frame",
    )
    assert without_extent is not None
    assert not any(
        SCENE_EXTENT_PATH in view.contents for view in _scene_views(without_extent.layout)
    )


def test_synced_layouts_give_every_variant_of_a_side_one_identity() -> None:
    layouts = build_synced_scene_layouts(
        _MULTI_PATHS | {SCENE_EXTENT_PATH},
        ["scene/cameras/front"],
        point_color_mode="semantic",
        camera_frustums_visible=False,
        timeline="frame",
    )

    assert layouts is not None
    assert layouts.camera_states == ("off", "on")
    assert [comparison.key for comparison in layouts.comparisons] == [
        "gt",
        "intensity",
        "entropy",
        "probability",
    ]
    assert [comparison.label for comparison in layouts.comparisons] == [
        "GT · Multi",
        "Intensity",
        "Normalized entropy",
        "Probability",
    ]
    assert layouts.initial_comparison == "gt"
    assert layouts.initial_camera_state == "off"

    for state, layout in layouts.left.items():
        (prediction,) = _scene_views(layout)
        assert prediction.identity == SYNCED_LEFT_IDENTITY
        assert prediction.name == "Prediction · Multi"
        assert "scene/prediction/segmentation" in prediction.contents
        assert "scene/prediction/detections" in prediction.contents
        assert SCENE_EXTENT_PATH in prediction.contents
        camera_override = next(o for o in prediction.overrides if o.path == "/scene/cameras")
        assert camera_override.visible is (state == "on")

    right_identities = {
        view.identity
        for states in layouts.right.values()
        for layout in states.values()
        for view in _scene_views(layout)
    }
    assert right_identities == {SYNCED_RIGHT_IDENTITY}
    (ground_truth,) = _scene_views(layouts.right["gt"]["off"])
    assert ground_truth.name == "GT · Multi"
    assert "scene/ground_truth/segmentation" in ground_truth.contents
    assert "scene/ground_truth/detections" in ground_truth.contents
    (intensity,) = _scene_views(layouts.right["intensity"]["on"])
    assert intensity.contents[0] == "scene/lidar/intensity"
    assert "scene/prediction/detections" not in intensity.contents

    # Both sides carry the same statistics row so mirrored input lands on the
    # same regions of both viewers.
    left_layout = layouts.left["off"]
    right_layout = layouts.right["gt"]["off"]
    assert isinstance(left_layout, LayoutGroup) and isinstance(right_layout, LayoutGroup)
    assert left_layout.shares == right_layout.shares == (2.0, 1.0)
    plot_names = [view.name for view in _views(left_layout) if view.kind == "time_series"]
    assert plot_names == ["3D IoU quality", "Detection matches"]
    assert plot_names == [view.name for view in _views(right_layout) if view.kind == "time_series"]
    assert all(view.identity for view in _views(left_layout) + _views(right_layout)), (
        "every synced view keeps its viewer state through a stable identity"
    )


def test_synced_layouts_without_cameras_or_statistics_are_single_views() -> None:
    layouts = build_synced_scene_layouts(
        {"scene/prediction/segmentation", "scene/ground_truth/segmentation"},
        [],
        point_color_mode="intensity",
        camera_frustums_visible=True,
        timeline="frame",
    )

    assert layouts is not None
    assert layouts.camera_states == ("off",)
    assert layouts.initial_camera_state == "off"
    assert [comparison.key for comparison in layouts.comparisons] == ["gt"]
    assert layouts.initial_comparison == "gt"
    assert isinstance(layouts.left["off"], ViewSpec)
    assert isinstance(layouts.right["gt"]["off"], ViewSpec)
    assert layouts.left["off"].name == "Prediction · Segmentation3D"
    assert layouts.right["gt"]["off"].name == "GT · Segmentation3D"


def test_synced_layouts_need_a_prediction_and_a_comparison() -> None:
    for paths in (
        {"scene/ground_truth/segmentation"},
        {"scene/prediction/segmentation"},
        {"scene/lidar/solid"},
        set(),
    ):
        assert (
            build_synced_scene_layouts(
                paths,
                [],
                point_color_mode="semantic",
                camera_frustums_visible=False,
                timeline="frame",
            )
            is None
        )
