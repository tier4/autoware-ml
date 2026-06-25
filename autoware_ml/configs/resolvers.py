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

"""Hydra and OmegaConf resolver registration for bundled Autoware-ML configs."""

from collections.abc import Mapping

from omegaconf import OmegaConf


def strip_tasks_prefix(config_name: str) -> str:
    """Return the user-facing config name without the bundled ``tasks/`` prefix."""
    return str(config_name).removeprefix("tasks/")


def segmentation_class_names(
    class_mapping: Mapping[str, int], num_classes: int, separator: str = "-"
) -> list[str]:
    """Build an ordered class-name list from a ``name -> index`` mapping.

    Inverts the segmentation ``class_mapping`` into per-index names for metric
    keys. When several names share an index the names are joined in definition
    order (for example ``noise-ghost_point``). Indices outside
    ``[0, num_classes)`` (such as the ignore index) are dropped, and any index
    without a name falls back to ``class_<i>``.

    Args:
        class_mapping: Mapping of raw class name to target class index.
        num_classes: Number of target classes.
        separator: String used to join names that share an index.

    Returns:
        List of length ``num_classes`` mapping each index to its name.
    """
    index_to_names: dict[int, list[str]] = {}
    if class_mapping:
        for name, index in class_mapping.items():
            target = int(index)
            if 0 <= target < int(num_classes):
                index_to_names.setdefault(target, []).append(str(name))
    return [
        separator.join(index_to_names[index]) if index in index_to_names else f"class_{index}"
        for index in range(int(num_classes))
    ]


def register_config_resolvers() -> None:
    """Register all custom OmegaConf resolvers required by bundled configs."""
    OmegaConf.register_new_resolver("user_config_name", strip_tasks_prefix, replace=True)
    OmegaConf.register_new_resolver("seg_class_names", segmentation_class_names, replace=True)
