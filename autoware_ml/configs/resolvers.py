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

from omegaconf import DictConfig, ListConfig, OmegaConf


def strip_tasks_prefix(config_name: str) -> str:
    """Return the user-facing config name without the bundled ``tasks/`` prefix.

    Args:
        config_name: Config name as referenced on the command line.

    Returns:
        The name without a leading ``tasks/`` component.
    """
    return str(config_name).removeprefix("tasks/")


def merge_lists(*lists: Iterable[Any]) -> ListConfig:
    """Concatenate several lists into one.

    OmegaConf replaces lists on merge instead of appending, so a config that
    needs the union of several lists requests it explicitly::

        model:
          metrics: ${merge_lists:${det.metrics},${seg.metrics}}

    Every element is fully resolved while its source node is still attached to
    the config tree, so the result carries plain values only.

    Args:
        *lists: Lists or list-like configs to concatenate, in order.

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


def raw_name_to_train_index(
    name_mapping: Mapping[str, str | None], class_names, ignore_index: int = -1
) -> DictConfig:
    """Derive the raw category to training index map for segmentation loading.

    ``name_mapping`` normalizes raw dataset categories to final class names,
    where ``null`` drops a category. ``class_names`` is the ordered list of
    trained classes, so a name's index is its position. A raw category whose
    final name is ``null`` or absent from ``class_names`` maps to
    ``ignore_index``. This way one mapping can also list categories a given
    model does not train.

    Args:
        name_mapping: Raw category name to final class name, ``null`` to drop.
        class_names: Ordered final class names defining the training indices.
        ignore_index: Index assigned to dropped and untrained categories.

    Returns:
        Config mapping of raw category name to training index. A config node
        is required here because in-place resolution writes the value back
        into the tree.
    """
    index_of = {str(name): index for index, name in enumerate(class_names)}
    mapping: dict[str, int] = {}
    for raw, final in dict(name_mapping).items():
        mapping[str(raw)] = (
            int(ignore_index) if final is None else index_of.get(str(final), int(ignore_index))
        )
    return cast(DictConfig, OmegaConf.create(mapping))


def register_config_resolvers() -> None:
    """Register all custom OmegaConf resolvers required by bundled configs."""
    OmegaConf.register_new_resolver("user_config_name", strip_tasks_prefix, replace=True)
    OmegaConf.register_new_resolver("seg_class_mapping", raw_name_to_train_index, replace=True)
    OmegaConf.register_new_resolver("merge_lists", merge_lists, replace=True)
