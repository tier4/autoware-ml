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

"""Managed background sessions backed by a private tmux server.

The public feature scope is intentionally narrow:

- start a long-running task in the background
- view live terminal output
- list running sessions
- stop the tracked task

The implementation uses tmux internally, but the CLI is not intended to expose
general-purpose tmux session management.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

AUTOWARE_ML_SESSION_OPTION = "@autoware_ml_managed"
AUTOWARE_ML_CHILD_PGID_OPTION = "@autoware_ml_child_pgid"
TMUX_BASE_COMMAND = ["tmux", "-L", "autoware-ml", "-f", "/dev/null"]
_VIEWER_CLEAR_SCREEN = "\033[H\033[2J"
_VIEWER_HIDE_CURSOR = "\033[?25l"
_VIEWER_SHOW_CURSOR = "\033[?25h"


class SessionCommandError(RuntimeError):
    """Signal that a managed session command could not be completed.

    The CLI converts this error into concise user-facing messages instead of
    exposing raw subprocess tracebacks.
    """


def _matches_tmux_error(error: str | SessionCommandError, *prefixes: str) -> bool:
    """Return whether a tmux error message starts with one of the given prefixes."""
    message = str(error)
    return any(message.startswith(prefix) for prefix in prefixes)


def _is_missing_session_error(error: str | SessionCommandError) -> bool:
    """Return whether tmux reported a missing managed session."""
    return _matches_tmux_error(error, "can't find session:", "no server running on ")


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
        return subprocess.run(
            [*TMUX_BASE_COMMAND, *args], check=True, text=True, capture_output=True
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "tmux command failed."
        raise SessionCommandError(message) from exc


def _build_session_command(work_dir: Path, session_name: str, command_args: Sequence[str]) -> str:
    """Build the shell command executed inside a managed background session.

    The managed shell runs the forwarded command in a separate process group so
    the live viewer can exit without interrupting the task.

    Args:
        work_dir: Working directory for the managed shell.
        session_name: Managed session name.
        command_args: ``autoware-ml`` subcommand arguments.

    Returns:
        Shell command string executed inside the private tmux server.
    """
    quoted_command = " ".join(shlex.quote(arg) for arg in ["autoware-ml", *command_args])
    return _build_managed_shell_command(work_dir, session_name, quoted_command)


def _build_raw_session_command(
    work_dir: Path, session_name: str, command_args: Sequence[str]
) -> str:
    """Build the shell command for an arbitrary process executed in a managed session.

    Args:
        work_dir: Working directory for the managed shell.
        session_name: Managed session name.
        command_args: Raw command arguments executed without the ``autoware-ml`` prefix.

    Returns:
        Shell command string executed inside the private tmux server.
    """
    quoted_command = " ".join(shlex.quote(arg) for arg in command_args)
    return _build_managed_shell_command(work_dir, session_name, quoted_command)


def _build_managed_shell_command(work_dir: Path, session_name: str, quoted_command: str) -> str:
    """Build the shell wrapper used by managed background sessions.

    Args:
        work_dir: Working directory for the managed shell.
        session_name: Managed session name.
        quoted_command: Fully quoted command executed by the managed shell.

    Returns:
        Shell wrapper string passed to ``tmux new-session``.
    """
    child_command = shlex.quote(f"exec {quoted_command}")

    tmux_shell_command = " ".join(shlex.quote(token) for token in TMUX_BASE_COMMAND)
    shell_lines = [
        f"cd {shlex.quote(str(work_dir))} || exit 1",
        f"AUTOWARE_ML_SESSION_NAME={shlex.quote(session_name)}",
        f"setsid bash -lc {child_command} &",
        "child_pid=$!",
        tmux_shell_command
        + ' set-option -t "$AUTOWARE_ML_SESSION_NAME" -q '
        + AUTOWARE_ML_CHILD_PGID_OPTION
        + ' "$child_pid" >/dev/null 2>&1 || true',
        "detach_client() { "
        + tmux_shell_command
        + ' detach-client -s "$AUTOWARE_ML_SESSION_NAME" >/dev/null 2>&1 || true; }',
        "trap detach_client INT",
        "cleanup_session() { "
        + tmux_shell_command
        + ' set-option -t "$AUTOWARE_ML_SESSION_NAME" -qu '
        + AUTOWARE_ML_CHILD_PGID_OPTION
        + " >/dev/null 2>&1 || true; }",
        "trap cleanup_session EXIT",
        "while true; do",
        '  wait "$child_pid"',
        "  exit_code=$?",
        '  if ! kill -0 "$child_pid" 2>/dev/null; then',
        "    break",
        "  fi",
        "done",
        "trap - EXIT",
        'exit "$exit_code"',
    ]
    shell_script = "\n".join(shell_lines)
    return f"bash -lc {shlex.quote(shell_script)}"


def _read_session_option(name: str, option: str) -> str:
    """Return the value of a managed-session tmux option.

    Args:
        name: Session name.
        option: tmux session option name.

    Returns:
        Option value, or an empty string when the option is unset.

    Raises:
        SessionCommandError: If the session does not exist or tmux fails.
    """
    result = _run_tmux(["show-options", "-t", name, "-q", "-v", option])
    return result.stdout.strip()


def _stop_child_process_group(process_group_id: int, timeout_seconds: float = 10.0) -> None:
    """Stop a managed child process group gracefully, then forcefully if needed.

    Args:
        process_group_id: Process group ID recorded for the managed child.
        timeout_seconds: Grace period before escalating from ``SIGTERM`` to ``SIGKILL``.
    """
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return


def _kill_session_if_present(name: str) -> None:
    """Kill a managed session unless it has already disappeared.

    Args:
        name: Session name.

    Raises:
        SessionCommandError: If tmux reports an unexpected failure.
    """
    try:
        _run_tmux(["kill-session", "-t", name])
    except SessionCommandError as exc:
        if _is_missing_session_error(exc):
            return
        raise


def start_session(
    name: str,
    command_args: Sequence[str],
    cwd: str | None = None,
    attach: bool = False,
    raw: bool = False,
) -> None:
    """Start a detached managed session running a background task.

    Args:
        name: Managed session name.
        command_args: Command arguments forwarded into the managed shell.
        cwd: Working directory for the managed shell.
        attach: Whether to open the live viewer immediately after startup.
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

    _run_tmux(["new-session", "-d", "-s", name, session_command])
    _run_tmux(["set-option", "-t", name, "-q", "status", "off"])
    _run_tmux(["set-option", "-t", name, "-q", AUTOWARE_ML_SESSION_OPTION, "1"])
    if attach:
        attach_session(name)


def attach_session(name: str) -> None:
    """Render a live terminal view of an existing managed session.

    Args:
        name: Session name.

    Raises:
        SessionCommandError: If the session cannot be viewed.
    """
    stdout = sys.stdout
    last_frame: str | None = None
    has_rendered_frame = False
    viewer_size: tuple[int, int] | None = None
    try:
        _run_tmux(["has-session", "-t", name])
        stdout.write(_VIEWER_HIDE_CURSOR)
        stdout.flush()
        while True:
            columns, rows = shutil.get_terminal_size(fallback=(80, 24))
            current_size = (columns, rows)
            if current_size != viewer_size:
                _run_tmux(["resize-window", "-t", name, "-x", str(columns), "-y", str(rows)])
                viewer_size = current_size
            frame = _run_tmux(["capture-pane", "-p", "-e", "-J", "-t", name]).stdout
            if frame != last_frame:
                stdout.write(_VIEWER_CLEAR_SCREEN)
                stdout.write(frame)
                if not frame.endswith("\n"):
                    stdout.write("\n")
                stdout.flush()
                last_frame = frame
                has_rendered_frame = True
            time.sleep(0.1)
    except KeyboardInterrupt:
        return
    except SessionCommandError as exc:
        if _is_missing_session_error(exc) and has_rendered_frame:
            return
        raise
    finally:
        stdout.write(_VIEWER_SHOW_CURSOR)
        stdout.flush()


def list_sessions() -> str:
    """Return formatted information about managed background sessions.

    Returns:
        Tabular session information. Returns an empty string when no managed
        sessions are running.
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
        if _matches_tmux_error(exc, "no server running on "):
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
    """Disconnect raw tmux clients attached to a managed session.

    Most users do not need this command because ``autoware-ml session attach``
    uses a read-only live viewer instead of a tmux client.

    Args:
        name: Session name.

    Raises:
        SessionCommandError: If the session does not exist or tmux fails.
    """
    _run_tmux(["has-session", "-t", name])
    _run_tmux(["detach-client", "-s", name])


def stop_session(name: str) -> None:
    """Stop the tracked task and close the managed session.

    Args:
        name: Session name.

    Raises:
        SessionCommandError: If the session does not exist or tmux fails.
    """
    _run_tmux(["has-session", "-t", name])
    child_process_group = _read_session_option(name, AUTOWARE_ML_CHILD_PGID_OPTION)
    if child_process_group:
        _stop_child_process_group(int(child_process_group))
    _kill_session_if_present(name)
