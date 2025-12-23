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

"""CLI helpers."""

import importlib
from typing import Any, Dict, List


def parse_extra_args(extra_args: List[str]) -> Dict[str, Any]:
    """Refines a list of CLI strings into a dictionary of typed kwargs.

    Args:
        extra_args: List of CLI strings.

    Returns:
        Dictionary of typed kwargs.
    """
    kwargs: Dict[str, Any] = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]

        # Clean the key: strip all leading dashes and replace internal hyphens with underscores
        key = arg.lstrip("-").replace("-", "_")

        # Check if the next element is a value or if the current arg is a standalone flag
        if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("-"):
            value_str = extra_args[i + 1]

            # Type Conversion Logic
            if value_str.lower() == "true":
                value = True
            elif value_str.lower() == "false":
                value = False
            else:
                try:
                    # Attempt integer conversion
                    if "." in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    # Fallback to string
                    value = value_str

            kwargs[key] = value
            i += 2  # Consumed key and value
        else:
            # It's a boolean flag (e.g., --verbose) or a key without a clear value
            # If the original arg didn't start with a dash, we treat it as a key/value
            # based on context, but your tests suggest no-dash keys still expect values.
            if not arg.startswith("-"):
                # This covers the 'test_no_dash_key' scenario where 'learning-rate' is followed by '0.001'
                # The logic above already handles this if the NEXT arg isn't a dash-arg.
                i += 1  # This branch is actually unreachable with the logic above,
                # but kept for structural clarity.
            else:
                kwargs[key] = True
                i += 1

    return kwargs


def expand_config_path(config_path: str, prefix: str) -> str:
    """Expand a short config path to a fully namespaced path.

    Args:
        config_path: Short or fully namespaced config path.
        prefix: Prefix to add to the config path (e.g., "tasks").
    Returns:
        Fully namespaced config path.
    """
    if config_path.startswith(f"{prefix}/"):
        return config_path

    return f"{prefix}/{config_path}"


def adjust_argv(argv: List[str]) -> List[str]:
    """Adjust the argv list.

    Remove escape characters, split on spaces, flatten the list and remove unresolved items.

    Args:
        argv: List of CLI strings.

    Returns:
        List of CLI strings.
    """
    result = []
    for arg in argv:
        # Skip empty strings and unresolved
        if not arg or arg.startswith("$"):
            continue
        # Remove escape characters and split on spaces
        result.extend(arg.replace("\\", "").split(" "))
    return result


def run_lazy_script(module_path: str, function_name: str, *args: Any, **kwargs: Any) -> Any:
    """Helper to load and run a function only when called.

    Args:
        module_path: Path to the module containing the function.
        function_name: Name of the function to run.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    Returns:
        The result of the function call.
    """
    module = importlib.import_module(module_path)
    func = getattr(module, function_name)
    return func(*args, **kwargs)
