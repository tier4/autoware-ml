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

from collections.abc import Iterable
from typing import Any, cast

from omegaconf import ListConfig, OmegaConf

from autoware_ml.utils.config.settings import Settings


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
          metrics: ${merge_lists:${det3d.metrics},${seg3d.metrics}}

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


def class_count(class_names: Iterable[Any]) -> int:
    """Number of classes a taxonomy names.

    The taxonomy owns the class list, so the count is read off it rather than restated. A
    model trains as many outputs as its database has classes, and two taxonomies of the same
    task disagree on that number::

        num_classes: ${class_count:${database.taxonomy.detection3d.class_names}}

    Args:
        class_names: Class names of the taxonomy.

    Returns:
        The number of classes.
    """
    return len(list(class_names))


def register_config_resolvers(settings: Settings) -> None:
    """Register all custom OmegaConf resolvers required by bundled configs.

    Args:
        settings: Settings of the checkout the path resolvers read.
    """
    OmegaConf.register_new_resolver("user_config_name", strip_tasks_prefix, replace=True)
    OmegaConf.register_new_resolver("merge_lists", merge_lists, replace=True)
    OmegaConf.register_new_resolver("class_count", class_count, replace=True)
    OmegaConf.register_new_resolver("data_path", settings.resolve_data_path, replace=True)
    OmegaConf.register_new_resolver("cache_path", settings.resolve_cache_path, replace=True)
    OmegaConf.register_new_resolver("working_dir", settings.resolve_working_dir, replace=True)
    OmegaConf.register_new_resolver("scenarios_path", settings.resolve_scenarios_path, replace=True)
