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

"""CLI helpers.

This module contains shared parsing, completion, and lazy-import helpers for the CLI.
"""

import importlib
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autoware_ml.configs.paths import CONFIGS_ROOT
from autoware_ml.configs.resolvers import register_config_resolvers
from autoware_ml.utils.session import AUTOWARE_ML_SESSION_OPTION, TMUX_BASE_COMMAND

register_config_resolvers()


def parse_extra_args(extra_args: Sequence[str]) -> dict[str, Any]:
    """Refines a list of CLI strings into a dictionary of typed kwargs.

    Args:
        extra_args: List of CLI strings.

    Returns:
        Dictionary of typed kwargs.
    """
    kwargs: dict[str, Any] = {}
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


def resolve_config_reference(config_value: str, prefix: str) -> tuple[str | None, str, list[str]]:
    """Resolve a CLI config value into Hydra config-path/config-name parts.

    Args:
        config_value: Config value from CLI, either a task shorthand or a filesystem path.
        prefix: Prefix for bundled configs (e.g. ``tasks``).

    Returns:
        Tuple of (config_path, config_name, hydra_overrides).
        ``config_path`` is ``None`` for bundled configs.
    """
    expanded_path = Path(config_value).expanduser()
    is_filesystem_path = (
        expanded_path.is_absolute()
        or config_value.startswith(("./", "../", "~/"))
        or expanded_path.suffix in {".yaml", ".yml"}
        or expanded_path.exists()
    )

    if not is_filesystem_path:
        return None, expand_config_path(config_value, prefix), []

    resolved_path = expanded_path.resolve()
    try:
        relative_to_configs = resolved_path.relative_to(CONFIGS_ROOT)
    except ValueError:
        relative_to_configs = None

    if relative_to_configs is not None:
        return None, relative_to_configs.with_suffix("").as_posix(), []

    hydra_searchpath_override = "hydra.searchpath=[pkg://autoware_ml.configs]"
    return str(resolved_path.parent), resolved_path.stem, [hydra_searchpath_override]


def infer_user_config_name(config_value: str, prefix: str) -> str:
    """Infer the user-facing config name from a CLI config reference.

    Args:
        config_value: Config value from CLI, either a bundled config path or a filesystem path.
        prefix: Bundled config namespace prefix such as ``tasks``.

    Returns:
        User-facing config name without the bundled namespace prefix.
    """
    _, resolved_config_name, _ = resolve_config_reference(config_value, prefix)
    return resolved_config_name.removeprefix(f"{prefix}/")


def list_config_names(prefix: str) -> list[str]:
    """List bundled config names without YAML suffixes for shell completion.

    Args:
        prefix: Config namespace prefix such as ``tasks``.

    Returns:
        Sorted list of bundled config names.
    """
    config_root = CONFIGS_ROOT / prefix
    return sorted(
        config_file.relative_to(config_root).with_suffix("").as_posix()
        for config_file in config_root.rglob("*.yaml")
        if config_file.stem != "base"
    )


def complete_config_value(incomplete: str, prefix: str) -> list[str]:
    """Complete bundled config names and filesystem config paths.

    Args:
        incomplete: Current completion prefix entered by the user.
        prefix: Config namespace prefix such as ``tasks``.

    Returns:
        Completion candidates for config values.
    """
    suggestions: list[str] = []

    available_configs = list_config_names(prefix)
    if incomplete.startswith(f"{prefix}/"):
        suggestions.extend(
            f"{prefix}/{config_name}"
            for config_name in available_configs
            if f"{prefix}/{config_name}".startswith(incomplete)
        )
    else:
        suggestions.extend(
            config_name for config_name in available_configs if config_name.startswith(incomplete)
        )

    if _looks_like_path(incomplete):
        suggestions.extend(_complete_filesystem_path(incomplete))

    return sorted(dict.fromkeys(suggestions))


def complete_path_value(
    incomplete: str,
    *,
    directories_only: bool = False,
    file_suffixes: tuple[str, ...] | None = None,
) -> list[str]:
    """Complete filesystem paths.

    Args:
        incomplete: Current shell token.
        directories_only: Only suggest directories.
        file_suffixes: Allowed file suffixes when suggesting files.
    """
    if incomplete and not _looks_like_path(incomplete):
        return []
    return _complete_filesystem_path(
        incomplete,
        directories_only=directories_only,
        file_suffixes=file_suffixes,
    )


def list_tmux_session_names() -> list[str]:
    """List managed session names for shell completion.

    Returns:
        Managed session names, or an empty list when tmux is unavailable.
    """
    if shutil.which("tmux") is None:
        return []

    result = subprocess.run(
        [
            *TMUX_BASE_COMMAND,
            "list-sessions",
            "-F",
            "#{session_name}\t#{" + AUTOWARE_ML_SESSION_OPTION + "}",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    names: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, _, managed = line.partition("\t")
        if managed == "1":
            names.append(name)
    return names


def complete_session_name_value(incomplete: str) -> list[str]:
    """Complete managed session names.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Matching managed session names.
    """
    return [name for name in list_tmux_session_names() if name.startswith(incomplete)]


def complete_session_command_value(command_args: list[str], incomplete: str) -> list[str]:
    """Complete commands forwarded through ``autoware-ml session start``.

    Args:
        command_args: Already parsed forwarded command tokens.
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for the forwarded command line.
    """

    def complete_option_values() -> list[str]:
        if not command_args:
            return []

        root = command_args[0]
        last = command_args[-1]

        if root in {"train", "test", "deploy"} and last == "--config-name":
            return complete_config_value(incomplete, "tasks")
        if root in {"test", "deploy"} and last == "+checkpoint":
            return complete_path_value(incomplete, file_suffixes=(".ckpt",))
        if root == "create-dataset" and last in {"--root-path", "--out-dir"}:
            return complete_path_value(incomplete, directories_only=True)
        if root == "mlflow":
            if len(command_args) >= 2 and command_args[1] == "ui" and last == "--db-path":
                return complete_path_value(incomplete)
            if len(command_args) >= 2 and command_args[1] == "export":
                if last == "--config-name":
                    return complete_config_value(incomplete, "tasks")
                if last == "--db-path":
                    return complete_path_value(incomplete)
                if last == "--export-dir":
                    return complete_path_value(incomplete, directories_only=True)
        return []

    value_suggestions = complete_option_values()
    if value_suggestions:
        return value_suggestions

    available_commands = [
        "train",
        "test",
        "deploy",
        "create-dataset",
        "mlflow",
    ]

    if not command_args:
        return [command for command in available_commands if command.startswith(incomplete)]

    root_command = command_args[0]
    last_arg = command_args[-1]

    if root_command == "mlflow":
        if len(command_args) == 1:
            return [
                subcommand for subcommand in ["ui", "export"] if subcommand.startswith(incomplete)
            ]
        if command_args[1] == "ui":
            return [
                option
                for option in ["--db-path", "--host", "--port"]
                if option.startswith(incomplete)
            ]
        if command_args[1] == "export" and last_arg == "--config-name":
            return complete_config_value(incomplete, "tasks")
        if command_args[1] == "export":
            return [
                option
                for option in [
                    "--db-path",
                    "--experiment-name",
                    "--config-name",
                    "--export-dir",
                    "--override",
                ]
                if option.startswith(incomplete)
            ]

    command_options = {
        "train": ["--config-name"],
        "test": ["--config-name", "+checkpoint"],
        "deploy": ["--config-name", "+checkpoint"],
        "create-dataset": ["--dataset", "--task", "--root-path", "--out-dir"],
        "mlflow": ["ui", "export"],
    }
    return [
        option for option in command_options.get(root_command, []) if option.startswith(incomplete)
    ]


def _looks_like_path(value: str) -> bool:
    """Return whether a completion prefix should be treated as a filesystem path.

    Args:
        value: Current completion prefix.

    Returns:
        ``True`` when the prefix should be handled as a path.
    """
    return value.startswith(("/", "./", "../", "~/"))


def _complete_filesystem_path(
    incomplete: str,
    *,
    directories_only: bool = False,
    file_suffixes: tuple[str, ...] | None = None,
) -> list[str]:
    """Complete filesystem paths relative to the current working directory.

    Args:
        incomplete: Current shell token.
        directories_only: Whether only directories should be suggested.
        file_suffixes: Optional file suffix filter.

    Returns:
        List of completion candidates.
    """
    expanded = Path(incomplete).expanduser() if incomplete else Path(".")
    search_dir = expanded if expanded.is_dir() else expanded.parent
    if not search_dir.exists():
        return []

    path_prefix = _path_display_prefix(incomplete)
    name_prefix = "" if incomplete.endswith("/") else expanded.name
    suggestions: list[str] = []
    for candidate in sorted(search_dir.iterdir()):
        if not candidate.name.startswith(name_prefix):
            continue
        if candidate.is_dir():
            suggestions.append(f"{path_prefix}{candidate.name}/")
        elif not directories_only and (file_suffixes is None or candidate.suffix in file_suffixes):
            suggestions.append(f"{path_prefix}{candidate.name}")

    return suggestions


def _path_display_prefix(incomplete: str) -> str:
    """Preserve the path prefix displayed in completion suggestions.

    Args:
        incomplete: Current shell token.

    Returns:
        Path prefix that should be prepended to suggested names.
    """
    if not incomplete:
        return ""

    expanded = Path(incomplete).expanduser()
    if expanded.is_dir() or incomplete.endswith("/"):
        return incomplete

    separator_index = incomplete.rfind("/")
    if separator_index == -1:
        return ""

    return incomplete[: separator_index + 1]


def adjust_argv(argv: Sequence[str]) -> list[str]:
    """Adjust the argv list.

    Normalize argv while preserving Hydra-style overrides.

    Args:
        argv: List of CLI strings.

    Returns:
        List of CLI strings.
    """
    result: list[str] = []
    for arg in argv:
        if not arg or arg.startswith("$"):
            continue
        if " " not in arg:
            result.append(arg)
            continue
        result.extend(_split_cli_argument(arg))
    return result


def _split_cli_argument(argument: str) -> list[str]:
    """Split a CLI argument string while respecting quotes and bracketed values.

    Args:
        argument: Raw CLI argument string.

    Returns:
        Tokenized argument list.
    """
    tokens: list[str] = []
    current: list[str] = []
    quote_char: str | None = None
    nesting = 0
    escape_next = False

    for char in argument:
        if escape_next:
            current.append(char)
            escape_next = False
            continue

        if char == "\\" and quote_char is not None:
            current.append(char)
            escape_next = True
            continue

        if quote_char is not None:
            current.append(char)
            if char == quote_char:
                quote_char = None
            continue

        if char in {'"', "'"}:
            current.append(char)
            quote_char = char
            continue

        if char in {"[", "{", "("}:
            nesting += 1
            current.append(char)
            continue

        if char in {"]", "}", ")"}:
            nesting = max(0, nesting - 1)
            current.append(char)
            continue

        if char.isspace() and nesting == 0:
            if current:
                tokens.append("".join(current))
                current = []
            continue

        current.append(char)

    if current:
        tokens.append("".join(current))

    return [
        shlex.split(token)[0] if token[:1] in {'"', "'"} and len(shlex.split(token)) == 1 else token
        for token in tokens
    ]


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
