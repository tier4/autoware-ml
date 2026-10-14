"""
Filters of the 3D bounding boxes of a sample, by class, attribute, range and point count.
The code is modified based on
https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/transforms_3d.py.
"""

from collections.abc import Sequence

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.transforms.base import BaseTransform


def parse_exclusion_rules(rules: Sequence[Sequence[str]]) -> frozenset[tuple[str, str]]:
    """
    Read box exclusion rules as class and attribute pairs.

    Args:
      rules: Exclusion rules, each a [class_name, attribute] pair.

    Returns:
      frozenset[tuple[str, str]]: The rules as (class_name, attribute) pairs.

    Raises:
      ValueError: If a rule is not a pair.
    """
    for index, rule in enumerate(rules):
        if isinstance(rule, str) or len(rule) != 2:
            raise ValueError(
                f"Exclusion rule {index} must be a [class_name, attribute] pair, got {rule!r}."
            )
    return frozenset((str(rule[0]), str(rule[1])) for rule in rules)


class BBoxesMinPointsFilter(BaseTransform):
    """Filter 3D bounding boxes by minimum number of points and distance of bboxes."""

    _required_keys = ["detection3d_gt_bboxes_3d", "point_cloud_data"]

    def __init__(
        self,
        min_points: int,
        bev_range: Sequence[float],
    ) -> None:
        """
        Initialize the BBoxesMinPointsFilter transform.

        Args:
            min_points (int): The minimum number of points required for a bounding box to be
                kept.
            bev_range (Sequence[float]): The BEV range ([x_min, y_min, x_max, y_max]) of the
                bounding boxes the minimum number of points applies to.
        """
        super().__init__(probability=None)
        self.min_points = min_points
        self.bev_range = torch.tensor(bev_range, dtype=torch.float32)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Drop the boxes in range that hold fewer than ``min_points`` points."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = (
            model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        )
        if not len(detection3d_gt_bboxes_3d):
            return model_gt_sample

        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = (
            model_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        )

        in_range = detection3d_gt_bboxes_3d.in_range_bev(self.bev_range)
        points_in_bboxes = detection3d_gt_bboxes_3d.compute_points_in_bboxes(
            points=point_cloud_data.coords,
        )

        # Boxes outside the range are kept whatever their point count
        keep_bboxes_mask = (points_in_bboxes.sum(dim=1) >= self.min_points) | ~in_range
        detection3d_gt_bboxes_3d.remove_bboxes(keep_bboxes_mask)
        return model_gt_sample


class BBoxesRangeFilter(BaseTransform):
    """Filter 3D bounding boxes whose center lies outside the point cloud range."""

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(
        self,
        point_cloud_range: Sequence[float],
    ) -> None:
        """
        Initialize the BBoxesRangeFilter transform.

        Args:
            point_cloud_range (Sequence[float]): The range ([x_min, y_min, z_min, x_max, y_max,
                z_max]) the bounding box centers have to lie in.
        """
        super().__init__(probability=None)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Drop the boxes whose center lies outside the point cloud range."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = (
            model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        )
        if not len(detection3d_gt_bboxes_3d):
            return model_gt_sample

        in_range_masks = detection3d_gt_bboxes_3d.in_range_3d(self.point_cloud_range)
        detection3d_gt_bboxes_3d.remove_bboxes(in_range_masks)

        return model_gt_sample


class BBoxesLabelNameFilter(BaseTransform):
    """Filter 3D bounding boxes by the name of the class they are mapped to.

    A box keeps the label name it was annotated with, which can be finer than the class it
    trains as (an ambulance trains as a car). The class is read from the label index of the
    box, so a box is kept when its class is one of the kept names, whatever its own name.
    """

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(self, label_names_to_keep: Sequence[str], class_names: Sequence[str]) -> None:
        """Initialize the BBoxesLabelNameFilter transform.

        Args:
            label_names_to_keep: Names of the classes whose boxes are kept.
            class_names: Class names in label index order.
        """
        super().__init__(probability=None)
        unknown = sorted(set(label_names_to_keep) - set(class_names))
        if unknown:
            raise ValueError(f"label_names_to_keep names classes that do not exist: {unknown}.")
        self.label_indices_to_keep = torch.tensor(
            [index for index, name in enumerate(class_names) if name in label_names_to_keep],
            dtype=torch.int64,
        )

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Filter 3D bounding boxes by the class of their label index."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = (
            model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        )
        if not len(detection3d_gt_bboxes_3d):
            return model_gt_sample

        labels = detection3d_gt_bboxes_3d.bbox_labels.to(torch.int64)
        bboxes_to_keep_mask = torch.isin(labels, self.label_indices_to_keep.to(labels.device))

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        detection3d_gt_bboxes_3d.remove_bboxes(bboxes_to_keep_mask)
        return model_gt_sample


class BBoxesAttributeFilter(BaseTransform):
    """
    Drop the 3D bounding boxes whose class and attributes match an exclusion rule.

    Some annotated objects are not detection targets, for example a parked bicycle or a
    motorcycle without a rider. A rule names a class and an attribute. Boxes of that class with
    that attribute are removed, so they are neither trained on nor scored.
    """

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(self, filter_attributes: Sequence[Sequence[str]]) -> None:
        """
        Initialize the BBoxesAttributeFilter transform.

        Args:
          filter_attributes: Exclusion rules, each a pair of class name and attribute name.
        """
        super().__init__(probability=None)
        self.filter_attributes = parse_exclusion_rules(filter_attributes)

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Drop the boxes matching an exclusion rule."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = (
            model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        )
        if not len(detection3d_gt_bboxes_3d) or not self.filter_attributes:
            return model_gt_sample

        bbox_attributes = detection3d_gt_bboxes_3d.bbox_attributes
        if bbox_attributes is None:
            raise ValueError(
                "The attribute filter needs the attributes of every box, the dataset served none."
            )

        bboxes_to_keep_mask = torch.tensor(
            [
                not any(
                    (label_name, attribute) in self.filter_attributes for attribute in attributes
                )
                for label_name, attributes in zip(
                    detection3d_gt_bboxes_3d.bbox_label_names, bbox_attributes, strict=True
                )
            ],
            dtype=torch.bool,
        )
        detection3d_gt_bboxes_3d.remove_bboxes(bboxes_to_keep_mask)
        return model_gt_sample


class BBoxesPhysicalFilter(BaseTransform):
    """Remove 3D bounding boxes that cannot become a valid training target.

    A physically invalid box is not a geometry outlier, it is pipeline garbage that would
    poison the regression losses: non-finite box parameters, non-positive dimensions (the
    box size targets are log-encoded, so a zero or negative extent is not representable),
    or a ground-plane speed beyond the physical bound (the velocity is never range-filtered
    downstream, so an absurd speed silently explodes the velocity loss).

    Note that the bbox parameters are already float32 here, so a float64 value that overflows
    the float32 cast has become non-finite by this point and is still caught.
    """

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(
        self,
        max_absolute_speed: float = 150.0,  # m/s, 540 km/h
    ) -> None:
        """
        Initialize the BBoxesPhysicalFilter transform.

        Args:
            max_absolute_speed (float): The maximum ground-plane speed, in m/s, that a bounding
                box may carry before it is considered non-physical. Defaults to 150.0 m/s
        """
        super().__init__(probability=None)
        self.max_absolute_speed = max_absolute_speed

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Remove non-physical 3D bounding boxes."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = model_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return model_gt_sample

        is_finite_mask = torch.isfinite(detection3d_gt_bboxes_3d.bbox_params).all(dim=1)
        has_positive_dims_mask = (detection3d_gt_bboxes_3d.dims > 0.0).all(dim=1)
        # Only the ground-plane velocity is bounded.
        is_speed_physical_mask = (
            torch.linalg.norm(detection3d_gt_bboxes_3d.velocity[:, :2], dim=1)
            <= self.max_absolute_speed
        )

        keep_bboxes_mask = is_finite_mask & has_positive_dims_mask & is_speed_physical_mask

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        detection3d_gt_bboxes_3d.remove_bboxes(keep_bboxes_mask)
        return model_gt_sample
