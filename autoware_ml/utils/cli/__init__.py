# Copyright 2025 TIER IV, Inc.
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

"""CLI utility exports.

This package exposes reusable command-line parsing and shell completion helpers
used by the top-level Autoware-ML CLI.
"""

from autoware_ml.utils.cli.helpers import (
    adjust_argv,
    complete_config_value,
    complete_path_value,
    complete_session_command_value,
    complete_session_name_value,
    expand_config_path,
    list_config_names,
    list_tmux_session_names,
    parse_extra_args,
    resolve_config_reference,
    run_lazy_script,
)

__all__ = [
    "adjust_argv",
    "complete_config_value",
    "complete_path_value",
    "complete_session_command_value",
    "complete_session_name_value",
    "expand_config_path",
    "list_config_names",
    "list_tmux_session_names",
    "parse_extra_args",
    "resolve_config_reference",
    "run_lazy_script",
]
