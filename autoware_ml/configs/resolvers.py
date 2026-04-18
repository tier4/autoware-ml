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

from omegaconf import OmegaConf


def strip_tasks_prefix(config_name: str) -> str:
    """Return the user-facing config name without the bundled ``tasks/`` prefix."""
    return str(config_name).removeprefix("tasks/")


def register_config_resolvers() -> None:
    """Register all custom OmegaConf resolvers required by bundled configs."""
    OmegaConf.register_new_resolver("user_config_name", strip_tasks_prefix, replace=True)
