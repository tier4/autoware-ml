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

"""tmux-backed session management for long-running Autoware-ML commands.

This module implements the session lifecycle used by the CLI for starting,
attaching, detaching, and stopping managed tmux sessions.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

AUTOWARE_ML_SESSION_OPTION = "@autoware_ml_managed"


class SessionCommandError(RuntimeError):
    """Signal that a tmux-backed session command could not be completed.

    The CLI converts this error into concise user-facing messages instead of
    exposing raw subprocess tracebacks.
    """


def _require_tmux() -> None:
    """Validate that tmux is available in the current environment.

    Raises:
        SessionCommandError: If ``tmux`` cannot be found on the current PATH.
    """
    if shutil.which("tmux") is None:
        raise SessionCommandError("tmux is not installed. Install tmux to use session commands.")


def _run_tmux(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a tmux command and normalize subprocess failures.

    Args:
        args: tmux command arguments without the ``tmux`` executable.

    Returns:
        Completed tmux subprocess result.

    Raises:
        SessionCommandError: If tmux is unavailable or the command fails.
    """
    _require_tmux()
    try:
        return subprocess.run(["tmux", *args], check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "tmux command failed."
        raise SessionCommandError(message) from exc


def _build_session_command(work_dir: Path, session_name: str, command_args: Sequence[str]) -> str:
    """Build the shell command executed inside the tmux session.

    The managed shell runs the forwarded command in a separate process group so
    ``Ctrl+C`` can detach the tmux client without interrupting training.

    Args:
        work_dir: Working directory for the managed shell.
        session_name: tmux session name.
        command_args: ``autoware-ml`` subcommand arguments.

    Returns:
        Shell command string executed inside the tmux session.
    """
    quoted_command = " ".join(shlex.quote(arg) for arg in ["autoware-ml", *command_args])
    return _build_managed_shell_command(work_dir, session_name, quoted_command)


def _build_raw_session_command(
    work_dir: Path, session_name: str, command_args: Sequence[str]
) -> str:
    """Build the shell command for an arbitrary process executed inside the tmux session.

    Args:
        work_dir: Working directory for the managed shell.
        session_name: tmux session name.
        command_args: Raw command arguments executed without the ``autoware-ml`` prefix.

    Returns:
        Shell command string executed inside the tmux session.
    """
    quoted_command = " ".join(shlex.quote(arg) for arg in command_args)
    return _build_managed_shell_command(work_dir, session_name, quoted_command)


def _build_managed_shell_command(work_dir: Path, session_name: str, quoted_command: str) -> str:
    """Build the managed shell wrapper used by tmux-backed sessions.

    Args:
        work_dir: Working directory for the managed shell.
        session_name: tmux session name.
        quoted_command: Fully quoted command executed by the managed shell.

    Returns:
        Shell wrapper string passed to ``tmux send-keys``.
    """
    child_command = shlex.quote(f"exec {quoted_command}")

    shell_lines = [
        f"cd {shlex.quote(str(work_dir))} || exit 1",
        f"AUTOWARE_ML_SESSION_NAME={shlex.quote(session_name)}",
        f"setsid bash -lc {child_command} &",
        "child_pid=$!",
        'detach_client() { tmux detach-client -s "$AUTOWARE_ML_SESSION_NAME" >/dev/null 2>&1 || true; }',
        'stop_child() { if kill -0 "$child_pid" 2>/dev/null; then kill -- -"$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; fi; }',
        "trap detach_client INT",
        "trap stop_child TERM HUP EXIT",
        "while true; do",
        '  wait "$child_pid"',
        "  exit_code=$?",
        '  if ! kill -0 "$child_pid" 2>/dev/null; then',
        "    break",
        "  fi",
        "done",
        "trap - TERM HUP EXIT",
        'exit "$exit_code"',
    ]
    shell_script = "\n".join(shell_lines)
    return f"bash -lc {shlex.quote(shell_script)}"


def start_session(
    name: str,
    command_args: Sequence[str],
    cwd: str | None = None,
    attach: bool = False,
    raw: bool = False,
) -> None:
    """Start a detached tmux session running a managed command.

    Args:
        name: Session name.
        command_args: Command arguments forwarded into the managed shell.
        cwd: Working directory for the managed shell.
        attach: Whether to attach immediately after session creation.
        raw: Whether to run the forwarded command without the ``autoware-ml`` prefix.

    Raises:
        SessionCommandError: If no command is provided or tmux reports an error.
    """
    if not command_args:
        raise SessionCommandError(
            "A command is required, e.g. autoware-ml session start --name train -- train --config-name ..."
        )

    work_dir = Path(cwd).resolve() if cwd is not None else Path.cwd()
    session_command = (
        _build_raw_session_command(work_dir, name, command_args)
        if raw
        else _build_session_command(work_dir, name, command_args)
    )

    _run_tmux(["new-session", "-d", "-s", name])
    _run_tmux(["set-option", "-t", name, "-q", AUTOWARE_ML_SESSION_OPTION, "1"])
    _run_tmux(["send-keys", "-t", name, session_command, "C-m"])
    if attach:
        attach_session(name)


def attach_session(name: str) -> None:
    """Attach to an existing tmux session.

    Args:
        name: Session name.

    Raises:
        SessionCommandError: If the session cannot be attached.
    """
    _run_tmux(["has-session", "-t", name])
    try:
        subprocess.run(["tmux", "attach-session", "-t", name], check=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if exc.stderr is not None else ""
        if not message:
            message = f"Failed to attach to session '{name}'."
        raise SessionCommandError(message) from exc


def list_sessions() -> str:
    """Return formatted tmux session information.

    Returns:
        Tabular session information. Returns an empty string when no tmux server is running.
    """
    try:
        result = _run_tmux(
            [
                "list-sessions",
                "-F",
                "#{session_name}\t#{session_attached}\t#{session_windows}\t#{session_created_string}\t#{"
                + AUTOWARE_ML_SESSION_OPTION
                + "}",
            ]
        )
    except SessionCommandError as exc:
        if str(exc).startswith("no server running on "):
            return ""
        raise
    rows = [line.split("\t", maxsplit=4) for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return ""

    lines = ["NAME\tSTATUS\tWINDOWS\tCREATED"]
    for session_name, attached, windows, created, managed in rows:
        if managed != "1":
            continue
        status = "attached" if attached == "1" else "detached"
        window_label = "window" if windows == "1" else "windows"
        lines.append(f"{session_name}\t{status}\t{windows} {window_label}\t{created}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def detach_session(name: str) -> None:
    """Detach clients attached to a tmux session.

    Args:
        name: Session name.

    Raises:
        SessionCommandError: If the session does not exist or tmux fails.
    """
    _run_tmux(["has-session", "-t", name])
    _run_tmux(["detach-client", "-s", name])


def stop_session(name: str) -> None:
    """Stop a tmux session.

    Args:
        name: Session name.

    Raises:
        SessionCommandError: If the session does not exist or tmux fails.
    """
    _run_tmux(["kill-session", "-t", name])
