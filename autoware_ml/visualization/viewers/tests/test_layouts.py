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

from collections.abc import Sequence
import unittest

import numpy as np

from autoware_ml.visualization.viewers.events import (
    ImageEvent,
    LayoutGroup,
    TextDocumentEvent,
    ViewSpec,
)
from autoware_ml.visualization.viewers.layouts import (
    SCENE_EXTENT_PATH,
    SYNCED_LEFT_IDENTITY,
    SYNCED_RIGHT_IDENTITY,
    build_calibration_blueprint,
    build_scene_blueprint,
    build_synced_scene_layouts,
)

_MULTI_PATHS = frozenset(
    {
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
)


def _views(layout: ViewSpec | LayoutGroup) -> Sequence[ViewSpec]:
    """Collect the views of a layout in display order."""
    if isinstance(layout, ViewSpec):
        return (layout,)
    return tuple(view for child in layout.children for view in _views(child))


def _scene_views(layout: ViewSpec | LayoutGroup) -> Sequence[ViewSpec]:
    """Collect the 3D views of a layout in display order."""
    return tuple(view for view in _views(layout) if view.kind == "spatial3d")


def _plot_views(layout: ViewSpec | LayoutGroup) -> Sequence[ViewSpec]:
    """Collect the time series views of a layout in display order."""
    return tuple(view for view in _views(layout) if view.kind == "time_series")


def _statistics_group(layout: ViewSpec | LayoutGroup) -> LayoutGroup | None:
    """Return the statistics row of a task comparison, if any."""
    if isinstance(layout, LayoutGroup):
        if layout.name == "Statistics":
            return layout
        for child in layout.children:
            found = _statistics_group(child)
            if found is not None:
                return found
    return None


def _calibration_image(path: str) -> ImageEvent:
    """Build a blank calibration image event on one entity path."""
    return ImageEvent(path=path, image=np.zeros((4, 4, 3), dtype=np.uint8))


class TestCalibrationBlueprint(unittest.TestCase):
    """The calibration layout places the camera images and the verdict."""

    def test_calibration_layout_is_expressed_only_as_neutral_specs(self) -> None:
        """The raw and fused images become neutral views in display order."""
        blueprint = build_calibration_blueprint(
            [
                _calibration_image("calibration_status/camera/image"),
                _calibration_image("calibration_status/camera/fused"),
            ]
        )

        self.assertIsNotNone(blueprint)
        self.assertIsInstance(blueprint.layout, LayoutGroup)
        self.assertEqual(
            [child.name for child in blueprint.layout.children], ["Raw camera", "Fused camera"]
        )

    def test_calibration_layout_puts_the_verdict_under_the_images(self) -> None:
        """With a verdict logged, the images share a row and the verdict gets its own pane."""
        blueprint = build_calibration_blueprint(
            [
                _calibration_image("calibration_status/camera/image"),
                _calibration_image("calibration_status/camera/fused"),
                TextDocumentEvent(
                    path="calibration_status/status/verdict", text="# Prediction: CALIBRATED"
                ),
            ]
        )

        self.assertIsNotNone(blueprint)
        self.assertIsInstance(blueprint.layout, LayoutGroup)
        self.assertEqual(blueprint.layout.kind, "vertical")
        self.assertEqual(blueprint.layout.shares, (0.7, 0.3))
        self.assertIs(blueprint.blueprint_panel_expanded, False)
        self.assertIs(blueprint.selection_panel_expanded, False)
        self.assertIs(blueprint.time_panel_expanded, False)
        images, verdict = blueprint.layout.children
        self.assertIsInstance(images, LayoutGroup)
        self.assertEqual(images.kind, "horizontal")
        self.assertEqual([child.name for child in images.children], ["Raw camera", "Fused camera"])
        self.assertIsInstance(verdict, ViewSpec)
        self.assertEqual(verdict.kind, "text_document")
        self.assertEqual(verdict.name, "Calibration verdict")
        self.assertEqual(verdict.contents, ("calibration_status/status/verdict",))


class TestDataPreviewBlueprint(unittest.TestCase):
    """Without predictions the scene layout is anchored on the ground truth."""

    def test_data_preview_layout_anchors_ground_truth_on_the_left(self) -> None:
        """Without predictions the ground truth is the fixed pane and the rest stay selectable."""
        blueprint = build_scene_blueprint(
            {
                "scene/ground_truth/segmentation",
                "scene/lidar/intensity",
                "scene/lidar/solid",
                "scene/metrics/segmentation/num_points",
            },
            [],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(blueprint)
        comparison = blueprint.layout.children[0]
        self.assertIsInstance(comparison, LayoutGroup)
        panes = comparison.children[0]
        self.assertIsInstance(panes, LayoutGroup)
        self.assertEqual(panes.kind, "horizontal")
        left, tabs = panes.children
        self.assertIsInstance(left, ViewSpec)
        self.assertEqual(left.name, "GT · Segmentation3D")
        self.assertIsInstance(tabs, LayoutGroup)
        self.assertEqual(tabs.kind, "tabs")
        self.assertEqual([view.name for view in tabs.children], ["Intensity"])
        statistics = _statistics_group(blueprint.layout)
        self.assertIsNotNone(statistics)
        self.assertEqual([view.name for view in statistics.children], ["Points per frame"])

    def test_detection_data_preview_plots_ground_truth_counts(self) -> None:
        """A detection data preview plots the ground-truth box count per frame."""
        blueprint = build_scene_blueprint(
            {
                "scene/ground_truth/detections",
                "scene/lidar/intensity",
                "scene/lidar/solid",
                "scene/metrics/detection/num_ground_truth",
            },
            [],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(blueprint)
        statistics = _statistics_group(blueprint.layout)
        self.assertIsNotNone(statistics)
        (counts,) = statistics.children
        self.assertEqual(counts.name, "Detection counts")
        self.assertEqual(counts.contents, ("scene/metrics/detection/num_ground_truth",))
        self.assertEqual(counts.overrides[0].series_name, "GT boxes")


class TestPredictionBlueprint(unittest.TestCase):
    """With predictions the scene layout compares them with selectable references."""

    def test_segmentation_prediction_layout_plots_confidence(self) -> None:
        """A segmentation prediction plots its confidence and entropy on a unit range."""
        blueprint = build_scene_blueprint(
            {
                "scene/prediction/segmentation",
                "scene/ground_truth/segmentation",
                "scene/lidar/intensity",
                "scene/metrics/segmentation/mean_confidence",
                "scene/metrics/segmentation/mean_entropy",
                "scene/metrics/segmentation/num_points",
            },
            [],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(blueprint)
        statistics = _statistics_group(blueprint.layout)
        self.assertIsNotNone(statistics)
        self.assertEqual(
            [view.name for view in statistics.children],
            ["Segmentation confidence", "Points per frame"],
        )
        confidence = statistics.children[0]
        self.assertEqual(confidence.y_range, (0.0, 1.0))
        self.assertEqual(
            [override.series_name for override in confidence.overrides],
            ["Mean confidence", "Mean normalized entropy"],
        )

    def test_scene_layout_keeps_multi_prediction_left_and_comparison_selectable(self) -> None:
        """A multi-task prediction stays on the left and every reference is a tab on the right."""
        blueprint = build_scene_blueprint(
            _MULTI_PATHS,
            ["scene/cameras/front"],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(blueprint)
        serialized = repr(blueprint)
        self.assertNotIn("Semantic", serialized)
        for label in (
            "Multi comparison",
            "Prediction · Multi",
            "Normalized entropy",
            "Probability",
            "front · GT",
            "front · Prediction",
            "3D IoU quality",
            "Detection matches",
            "show_labels=True",
        ):
            with self.subTest(label=label):
                self.assertIn(label, serialized)
        self.assertIs(blueprint.blueprint_panel_expanded, True)
        self.assertEqual(
            [child.name for child in blueprint.layout.children], ["Multi comparison", "Cameras"]
        )

        multi_layout = blueprint.layout.children[0]
        self.assertIsInstance(multi_layout, LayoutGroup)
        camera_switch = multi_layout.children[0]
        self.assertIsInstance(camera_switch, LayoutGroup)
        self.assertEqual(camera_switch.kind, "tabs")
        self.assertEqual(camera_switch.name, "Multi comparison")
        self.assertEqual(camera_switch.active, 0)
        self.assertEqual(
            [child.name for child in camera_switch.children],
            ["Camera projections OFF", "Camera projections ON"],
        )

        comparison = camera_switch.children[0]
        self.assertIsInstance(comparison, LayoutGroup)
        self.assertEqual(comparison.kind, "horizontal")
        prediction = comparison.children[0]
        comparison_options = comparison.children[1]
        self.assertIsInstance(prediction, ViewSpec)
        self.assertEqual(prediction.name, "Prediction · Multi")
        self.assertEqual(
            prediction.contents[:2],
            ("scene/prediction/segmentation", "scene/prediction/detections"),
        )
        self.assertIsInstance(comparison_options, LayoutGroup)
        self.assertEqual(comparison_options.kind, "tabs")
        self.assertEqual(comparison_options.active, 0)
        self.assertEqual(
            [child.name for child in comparison_options.children],
            ["GT · Multi", "Intensity", "Normalized entropy", "Probability"],
        )
        self.assertNotIn("scene/lidar/intensity", prediction.contents)

        camera_layout = blueprint.layout.children[1]
        self.assertIsInstance(camera_layout, LayoutGroup)
        front_comparison = camera_layout.children[0]
        self.assertIsInstance(front_comparison, LayoutGroup)
        self.assertEqual(
            [child.name for child in front_comparison.children],
            ["front · Prediction", "front · GT"],
        )

    def test_scene_layout_can_start_with_intensity_on_the_right(self) -> None:
        """The intensity point color mode opens the intensity tab on the right."""
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

        self.assertIsNotNone(blueprint)
        self.assertIsInstance(blueprint.layout, LayoutGroup)
        comparison = blueprint.layout.children[0]
        self.assertIsInstance(comparison, LayoutGroup)
        comparison_options = comparison.children[1]
        self.assertIsInstance(comparison_options, LayoutGroup)
        self.assertEqual(comparison_options.active, 1)
        self.assertEqual(comparison_options.children[1].name, "Intensity")

    def test_scene_layout_camera_switch_respects_visible_initial_state(self) -> None:
        """Visible camera frustums open the camera projections ON tab."""
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

        self.assertIsNotNone(blueprint)
        self.assertIsInstance(blueprint.layout, LayoutGroup)
        camera_switch = blueprint.layout.children[0]
        self.assertIsInstance(camera_switch, LayoutGroup)
        self.assertEqual(camera_switch.kind, "tabs")
        self.assertEqual(camera_switch.name, "Multi comparison")
        self.assertEqual(camera_switch.active, 1)

    def test_scene_layout_uses_task_name_for_single_task_predictions(self) -> None:
        """A single-task prediction is named after its task."""
        blueprint = build_scene_blueprint(
            {"scene/prediction/segmentation", "scene/ground_truth/segmentation"},
            [],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(blueprint)
        self.assertIn("Prediction · Segmentation3D", repr(blueprint))
        self.assertNotIn("Semantic", repr(blueprint))

    def test_detection_prediction_uses_solid_geometry_instead_of_intensity(self) -> None:
        """A detection prediction is drawn over the solid point cloud, not the intensity one."""
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

        self.assertIsNotNone(blueprint)
        self.assertIsInstance(blueprint.layout, LayoutGroup)
        comparison = blueprint.layout.children[0]
        self.assertIsInstance(comparison, LayoutGroup)
        prediction = comparison.children[0]
        self.assertIsInstance(prediction, ViewSpec)
        self.assertEqual(prediction.name, "Prediction · Detection3D")
        self.assertEqual(prediction.contents, ("scene/lidar/solid", "scene/prediction/detections"))
        self.assertNotIn("scene/lidar/intensity", prediction.contents)

    def test_unknown_scene_adapter_gets_a_generic_view(self) -> None:
        """Paths of an unknown adapter land in one generic scene view."""
        blueprint = build_scene_blueprint(
            {"scene/new_task/result"},
            [],
            point_color_mode="solid",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(blueprint)
        self.assertIsInstance(blueprint.layout, LayoutGroup)
        view = blueprint.layout.children[0]
        self.assertIsInstance(view, ViewSpec)
        self.assertEqual(view.name, "Scene")
        self.assertEqual(view.contents, ("scene/new_task/result",))

    def test_scene_views_list_the_shared_extent_when_it_is_logged(self) -> None:
        """Every 3D view includes the shared scene extent only when it was logged."""
        blueprint = build_scene_blueprint(
            _MULTI_PATHS | {SCENE_EXTENT_PATH},
            ["scene/cameras/front"],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(blueprint)
        scene_views = _scene_views(blueprint.layout)
        self.assertTrue(scene_views)
        for view in scene_views:
            with self.subTest(view=view.name):
                self.assertIn(SCENE_EXTENT_PATH, view.contents)

        without_extent = build_scene_blueprint(
            _MULTI_PATHS,
            ["scene/cameras/front"],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )
        self.assertIsNotNone(without_extent)
        for view in _scene_views(without_extent.layout):
            with self.subTest(view=view.name):
                self.assertNotIn(SCENE_EXTENT_PATH, view.contents)


class TestSyncedSceneLayouts(unittest.TestCase):
    """The synced page gets one layout per side, comparison, and camera state."""

    def test_synced_layouts_give_every_variant_of_a_side_one_identity(self) -> None:
        """Every variant of a side shares one 3D view identity and the same statistics row."""
        layouts = build_synced_scene_layouts(
            _MULTI_PATHS | {SCENE_EXTENT_PATH},
            ["scene/cameras/front"],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(layouts)
        self.assertEqual(layouts.camera_states, ("off", "on"))
        self.assertEqual(
            [comparison.key for comparison in layouts.comparisons],
            ["gt", "intensity", "entropy", "probability"],
        )
        self.assertEqual(
            [comparison.label for comparison in layouts.comparisons],
            ["GT · Multi", "Intensity", "Normalized entropy", "Probability"],
        )
        self.assertEqual(layouts.initial_comparison, "gt")
        self.assertEqual(layouts.initial_camera_state, "off")

        for state, layout in layouts.left.items():
            with self.subTest(state=state):
                (prediction,) = _scene_views(layout)
                self.assertEqual(prediction.identity, SYNCED_LEFT_IDENTITY)
                self.assertEqual(prediction.name, "Prediction · Multi")
                self.assertIn("scene/prediction/segmentation", prediction.contents)
                self.assertIn("scene/prediction/detections", prediction.contents)
                self.assertIn(SCENE_EXTENT_PATH, prediction.contents)
                camera_override = next(
                    override
                    for override in prediction.overrides
                    if override.path == "/scene/cameras"
                )
                self.assertIs(camera_override.visible, state == "on")

        right_identities = {
            view.identity
            for states in layouts.right.values()
            for layout in states.values()
            for view in _scene_views(layout)
        }
        self.assertEqual(right_identities, {SYNCED_RIGHT_IDENTITY})
        (ground_truth,) = _scene_views(layouts.right["gt"]["off"])
        self.assertEqual(ground_truth.name, "GT · Multi")
        self.assertIn("scene/ground_truth/segmentation", ground_truth.contents)
        self.assertIn("scene/ground_truth/detections", ground_truth.contents)
        (intensity,) = _scene_views(layouts.right["intensity"]["on"])
        self.assertEqual(intensity.contents[0], "scene/lidar/intensity")
        self.assertNotIn("scene/prediction/detections", intensity.contents)

        # Both sides carry the same statistics row so mirrored input lands on the
        # same regions of both viewers.
        left_layout = layouts.left["off"]
        right_layout = layouts.right["gt"]["off"]
        self.assertIsInstance(left_layout, LayoutGroup)
        self.assertIsInstance(right_layout, LayoutGroup)
        self.assertEqual(left_layout.shares, (2.0, 1.0))
        self.assertEqual(right_layout.shares, (2.0, 1.0))
        plot_names = [view.name for view in _plot_views(left_layout)]
        self.assertEqual(plot_names, ["3D IoU quality", "Detection matches"])
        self.assertEqual(plot_names, [view.name for view in _plot_views(right_layout)])
        for view in (*_views(left_layout), *_views(right_layout)):
            with self.subTest(view=view.name):
                self.assertTrue(
                    view.identity,
                    "every synced view keeps its viewer state through a stable identity",
                )

    def test_synced_layouts_without_cameras_or_statistics_are_single_views(self) -> None:
        """Without cameras or statistics each side is one view and one camera state."""
        layouts = build_synced_scene_layouts(
            {"scene/prediction/segmentation", "scene/ground_truth/segmentation"},
            [],
            point_color_mode="intensity",
            camera_frustums_visible=True,
            timeline="frame",
        )

        self.assertIsNotNone(layouts)
        self.assertEqual(layouts.camera_states, ("off",))
        self.assertEqual(layouts.initial_camera_state, "off")
        self.assertEqual([comparison.key for comparison in layouts.comparisons], ["gt"])
        self.assertEqual(layouts.initial_comparison, "gt")
        self.assertIsInstance(layouts.left["off"], ViewSpec)
        self.assertIsInstance(layouts.right["gt"]["off"], ViewSpec)
        self.assertEqual(layouts.left["off"].name, "Prediction · Segmentation3D")
        self.assertEqual(layouts.right["gt"]["off"].name, "GT · Segmentation3D")

    def test_synced_layouts_anchor_ground_truth_left_for_data_previews(self) -> None:
        """Without predictions the synced page pairs the ground truth with the other variants."""
        layouts = build_synced_scene_layouts(
            {
                "scene/ground_truth/detections",
                "scene/lidar/intensity",
                "scene/lidar/solid",
                "scene/metrics/detection/num_ground_truth",
                SCENE_EXTENT_PATH,
            },
            [],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(layouts)
        self.assertEqual([comparison.key for comparison in layouts.comparisons], ["intensity"])
        self.assertEqual(layouts.initial_comparison, "intensity")
        (left,) = _scene_views(layouts.left["off"])
        self.assertEqual(left.name, "GT · Detection3D")
        self.assertIn("scene/ground_truth/detections", left.contents)
        self.assertIn("scene/lidar/solid", left.contents)
        self.assertEqual(
            [plot.name for plot in _plot_views(layouts.left["off"])], ["Detection counts"]
        )

    def test_synced_layouts_plot_segmentation_statistics(self) -> None:
        """Segmentation statistics are plotted on the synced page as well."""
        layouts = build_synced_scene_layouts(
            {
                "scene/prediction/segmentation",
                "scene/ground_truth/segmentation",
                "scene/lidar/intensity",
                "scene/metrics/segmentation/mean_confidence",
                "scene/metrics/segmentation/num_points",
                SCENE_EXTENT_PATH,
            },
            [],
            point_color_mode="semantic",
            camera_frustums_visible=False,
            timeline="frame",
        )

        self.assertIsNotNone(layouts)
        self.assertEqual(
            [plot.name for plot in _plot_views(layouts.right["gt"]["off"])],
            ["Segmentation confidence", "Points per frame"],
        )

    def test_synced_layouts_need_a_prediction_and_a_comparison(self) -> None:
        """Without something to show on both sides there is no synced layout."""
        for paths in (
            {"scene/ground_truth/segmentation"},
            {"scene/prediction/segmentation"},
            {"scene/lidar/solid"},
            set(),
        ):
            with self.subTest(paths=sorted(paths)):
                self.assertIsNone(
                    build_synced_scene_layouts(
                        paths,
                        [],
                        point_color_mode="semantic",
                        camera_frustums_visible=False,
                        timeline="frame",
                    )
                )


if __name__ == "__main__":
    unittest.main()
