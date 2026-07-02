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

from collections.abc import Iterable, Mapping
from typing import Any, cast

from omegaconf import ListConfig, OmegaConf


def strip_tasks_prefix(config_name: str) -> str:
    """Return the user-facing config name without the bundled ``tasks/`` prefix."""
    return str(config_name).removeprefix("tasks/")


def merge_lists(*lists: Iterable[Any]) -> ListConfig:
    """Concatenate several lists into one.

    Hydra/OmegaConf replaces lists on merge rather than appending, so the joint
    segmentation + detection task cannot accumulate both per-dataset metric
    suites through the defaults list alone. A plain ``[${a}, ${b}]`` list of
    references almost works, but breaks when a referenced suite contains a
    custom resolver that returns a list (the segmentation suite's
    ``${seg_class_names:...}``): OmegaConf cannot copy that list result into the
    new list element. This resolver sidesteps it by fully resolving each item
    (while its source node is still attached to the tree) and returning plain
    containers, e.g.::

        model:
          metrics: ${merge_lists:${det.metrics},${seg.metrics}}

    Args:
        *lists: Any number of lists (or list-like configs) to concatenate, in order.

    Returns:
        A single config list with every element of the inputs, in argument order.
    """
    merged: list[Any] = []
    for lst in lists:
        for item in lst:
            if OmegaConf.is_config(item):
                merged.append(OmegaConf.to_container(item, resolve=True))
            else:
                merged.append(item)
    return cast(ListConfig, OmegaConf.create(merged))


def segmentation_class_names(
    class_mapping: Mapping[str, int] | None, num_classes: int, separator: str = "-"
) -> ListConfig:
    """Build an ordered class-name list from a ``name -> index`` mapping.

    Inverts the segmentation ``class_mapping`` into per-index names for metric
    keys. When several names share an index the names are joined in definition
    order (for example ``noise-ghost_point``). Indices outside
    ``[0, num_classes)`` (such as the ignore index) are dropped, and any index
    without a name falls back to ``class_<i>``.

    Args:
        class_mapping: Mapping of raw class name to target class index. May be
            ``None`` or empty - as produced by ``${oc.select:...class_mapping,
            null}`` when no mapping is defined - in which case every index falls
            back to ``class_<i>``.
        num_classes: Number of target classes.
        separator: String used to join names that share an index.

    Returns:
        Config list of length ``num_classes`` mapping each index to its name.
        Returned as a ``ListConfig`` (not a plain ``list``) so that in-place
        ``OmegaConf.resolve()`` - which ``hydra.utils.instantiate`` runs on the
        model config - can write the resolved value back into the tree. A plain
        list raises ``UnsupportedValueType`` there (the same trap ``merge_lists``
        documents), which broke seg-only configs whose ``metrics`` reference this
        resolver directly.
    """
    index_to_names: dict[int, list[str]] = {}
    if class_mapping:
        for name, index in class_mapping.items():
            target = int(index)
            if 0 <= target < int(num_classes):
                index_to_names.setdefault(target, []).append(str(name))
    names = [
        separator.join(index_to_names[index]) if index in index_to_names else f"class_{index}"
        for index in range(int(num_classes))
    ]
    return cast(ListConfig, OmegaConf.create(names))


def register_config_resolvers() -> None:
    """Register all custom OmegaConf resolvers required by bundled configs."""
    OmegaConf.register_new_resolver("user_config_name", strip_tasks_prefix, replace=True)
    OmegaConf.register_new_resolver("seg_class_names", segmentation_class_names, replace=True)
    OmegaConf.register_new_resolver("merge_lists", merge_lists, replace=True)
