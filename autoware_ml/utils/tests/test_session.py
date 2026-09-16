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

"""Unit tests for managed background session helpers."""

from io import StringIO
import os
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import call, patch

import pytest

from autoware_ml.utils.session import (
    AUTOWARE_ML_CHILD_PGID_OPTION,
    SessionCommandError,
    _build_session_command,
    _kill_session_if_present,
    _build_raw_session_command,
    attach_session,
    detach_session,
    list_sessions,
    start_session,
    stop_session,
)

SAMPLE_CONFIG_NAME = "calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2"
SAMPLE_SESSION_NAME = "calibration-status-train"


class TestListSessions:
    def test_formats_session_rows(self) -> None:
        with patch(
            "autoware_ml.utils.session._run_tmux",
            return_value=CompletedProcess(
                args=["tmux"],
                returncode=0,
                stdout="default\t0\t1\tTue Mar 17\t1\ntrain\t1\t2\tTue Mar 17\t1\n",
                stderr="",
            ),
        ):
            output = list_sessions()

        assert output == (
            "NAME\tSTATUS\tWINDOWS\tCREATED\n"
            "default\tdetached\t1 window\tTue Mar 17\n"
            "train\tattached\t2 windows\tTue Mar 17"
        )

    def test_returns_empty_string_when_no_server_is_running(self) -> None:
        with patch(
            "autoware_ml.utils.session._run_tmux",
            side_effect=SessionCommandError("no server running on /tmp/tmux-1000/default"),
        ):
            output = list_sessions()

        assert output == ""


class TestStartSession:
    def test_builds_managed_shell_command(self) -> None:
        command = _build_session_command(
            Path("/workspace"),
            "default",
            ["train", "--config-name", SAMPLE_CONFIG_NAME],
        )

        assert command.startswith("bash -lc ")
        assert "cd /workspace || exit 1" in command
        assert "AUTOWARE_ML_SESSION_NAME=default" in command
        assert "setsid bash -lc" in command
        assert f"exec autoware-ml train --config-name {SAMPLE_CONFIG_NAME}" in command
        assert AUTOWARE_ML_CHILD_PGID_OPTION in command
        assert "-L autoware-ml -f /dev/null" in command
        assert "trap detach_client INT" in command
        assert "trap cleanup_session EXIT" in command
        assert "trap stop_child TERM HUP EXIT" not in command

    def test_starts_tmux_session_before_sending_command(self) -> None:
        with patch("autoware_ml.utils.session._run_tmux") as run_tmux_mock:
            start_session(
                name="default",
                command_args=["train", "--config-name", SAMPLE_CONFIG_NAME],
                cwd="/workspace",
            )

        expected_command = _build_session_command(
            Path("/workspace"),
            "default",
            ["train", "--config-name", SAMPLE_CONFIG_NAME],
        )
        run_tmux_mock.assert_has_calls(
            [
                call(["new-session", "-d", "-s", "default", expected_command]),
                call(["set-option", "-t", "default", "-q", "status", "off"]),
                call(["set-option", "-t", "default", "-q", "@autoware_ml_managed", "1"]),
            ]
        )

    def test_builds_managed_raw_command(self) -> None:
        command = _build_raw_session_command(
            Path("/workspace"),
            "docs",
            ["zensical", "serve"],
        )

        assert command.startswith("bash -lc ")
        assert "exec zensical serve" in command
        assert "AUTOWARE_ML_SESSION_NAME=docs" in command

    def test_starts_raw_tmux_session_before_sending_command(self) -> None:
        with patch("autoware_ml.utils.session._run_tmux") as run_tmux_mock:
            start_session(
                name="docs",
                command_args=["zensical", "serve"],
                cwd="/workspace",
                raw=True,
            )

        expected_command = _build_raw_session_command(
            Path("/workspace"),
            "docs",
            ["zensical", "serve"],
        )
        run_tmux_mock.assert_has_calls(
            [
                call(["new-session", "-d", "-s", "docs", expected_command]),
                call(["set-option", "-t", "docs", "-q", "status", "off"]),
                call(["set-option", "-t", "docs", "-q", "@autoware_ml_managed", "1"]),
            ]
        )

    def test_filters_unmanaged_sessions_from_listing(self) -> None:
        with patch(
            "autoware_ml.utils.session._run_tmux",
            return_value=CompletedProcess(
                args=["tmux"],
                returncode=0,
                stdout="default\t0\t1\tTue Mar 17\t1\nscratch\t0\t1\tTue Mar 17\t\n",
                stderr="",
            ),
        ):
            output = list_sessions()

        assert output == "NAME\tSTATUS\tWINDOWS\tCREATED\ndefault\tdetached\t1 window\tTue Mar 17"

    def test_requires_forwarded_command(self) -> None:
        with pytest.raises(SessionCommandError, match="A command is required"):
            start_session(name="default", command_args=[], cwd=str(Path.cwd()))


class TestDetachSession:
    def test_detaches_existing_session(self) -> None:
        with patch("autoware_ml.utils.session._run_tmux") as run_tmux_mock:
            detach_session("default")

        run_tmux_mock.assert_has_calls(
            [
                call(["has-session", "-t", "default"]),
                call(["detach-client", "-s", "default"]),
            ]
        )


class TestAttachSession:
    def test_raises_when_session_is_missing_before_viewer_starts(self) -> None:
        with (
            patch("autoware_ml.utils.session._run_tmux") as run_tmux_mock,
            patch("autoware_ml.utils.session.sys.stdout", new_callable=StringIO) as stdout_mock,
            pytest.raises(SessionCommandError, match="can't find session: default"),
        ):
            run_tmux_mock.side_effect = [SessionCommandError("can't find session: default")]
            attach_session("default")

        run_tmux_mock.assert_called_once_with(["has-session", "-t", "default"])
        assert stdout_mock.getvalue() == "\033[?25h"

    def test_renders_live_session_frame_until_interrupted(self) -> None:
        with (
            patch("autoware_ml.utils.session._run_tmux") as run_tmux_mock,
            patch(
                "autoware_ml.utils.session.shutil.get_terminal_size",
                return_value=os.terminal_size((120, 40)),
            ),
            patch("autoware_ml.utils.session.time.sleep", side_effect=KeyboardInterrupt),
            patch("autoware_ml.utils.session.sys.stdout", new_callable=StringIO) as stdout_mock,
        ):
            run_tmux_mock.side_effect = [
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="progress 42%\n", stderr=""),
            ]
            attach_session("default")

        run_tmux_mock.assert_has_calls(
            [
                call(["has-session", "-t", "default"]),
                call(["resize-window", "-t", "default", "-x", "120", "-y", "40"]),
                call(["capture-pane", "-p", "-e", "-J", "-t", "default"]),
            ]
        )
        output = stdout_mock.getvalue()
        assert "\033[?25l" in output
        assert "\033[H\033[2J" in output
        assert "progress 42%" in output
        assert output.endswith("\033[?25h")

    def test_returns_cleanly_when_session_disappears(self) -> None:
        with (
            patch("autoware_ml.utils.session._run_tmux") as run_tmux_mock,
            patch(
                "autoware_ml.utils.session.shutil.get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
            patch("autoware_ml.utils.session.sys.stdout", new_callable=StringIO) as stdout_mock,
        ):
            run_tmux_mock.side_effect = [
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="progress 42%\n", stderr=""),
                SessionCommandError("can't find session: default"),
            ]
            attach_session("default")

        run_tmux_mock.assert_has_calls(
            [
                call(["has-session", "-t", "default"]),
                call(["resize-window", "-t", "default", "-x", "80", "-y", "24"]),
                call(["capture-pane", "-p", "-e", "-J", "-t", "default"]),
            ]
        )
        assert stdout_mock.getvalue().startswith("\033[?25l")
        assert "progress 42%" in stdout_mock.getvalue()
        assert stdout_mock.getvalue().endswith("\033[?25h")

    def test_resizes_only_when_terminal_size_changes(self) -> None:
        with (
            patch("autoware_ml.utils.session._run_tmux") as run_tmux_mock,
            patch(
                "autoware_ml.utils.session.shutil.get_terminal_size",
                side_effect=[
                    os.terminal_size((80, 24)),
                    os.terminal_size((80, 24)),
                ],
            ),
            patch("autoware_ml.utils.session.time.sleep", side_effect=[None, KeyboardInterrupt]),
            patch("autoware_ml.utils.session.sys.stdout", new_callable=StringIO),
        ):
            run_tmux_mock.side_effect = [
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="frame 1\n", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="frame 2\n", stderr=""),
            ]
            attach_session("default")

        run_tmux_mock.assert_has_calls(
            [
                call(["has-session", "-t", "default"]),
                call(["resize-window", "-t", "default", "-x", "80", "-y", "24"]),
                call(["capture-pane", "-p", "-e", "-J", "-t", "default"]),
                call(["capture-pane", "-p", "-e", "-J", "-t", "default"]),
            ]
        )


class TestStopSession:
    def test_stops_recorded_child_process_group_before_killing_session(self) -> None:
        with (
            patch("autoware_ml.utils.session._run_tmux") as run_tmux_mock,
            patch(
                "autoware_ml.utils.session._stop_child_process_group"
            ) as stop_child_process_group_mock,
        ):
            run_tmux_mock.side_effect = [
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="12345\n", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
            ]

            stop_session("default")

        run_tmux_mock.assert_has_calls(
            [
                call(["has-session", "-t", "default"]),
                call(["show-options", "-t", "default", "-q", "-v", AUTOWARE_ML_CHILD_PGID_OPTION]),
                call(["kill-session", "-t", "default"]),
            ]
        )
        stop_child_process_group_mock.assert_called_once_with(12345)

    def test_kills_session_even_when_no_child_process_group_is_recorded(self) -> None:
        with (
            patch("autoware_ml.utils.session._run_tmux") as run_tmux_mock,
            patch(
                "autoware_ml.utils.session._stop_child_process_group"
            ) as stop_child_process_group_mock,
        ):
            run_tmux_mock.side_effect = [
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="\n", stderr=""),
                CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
            ]

            stop_session("default")

        stop_child_process_group_mock.assert_not_called()
        run_tmux_mock.assert_has_calls(
            [
                call(["has-session", "-t", "default"]),
                call(["show-options", "-t", "default", "-q", "-v", AUTOWARE_ML_CHILD_PGID_OPTION]),
                call(["kill-session", "-t", "default"]),
            ]
        )


class TestKillSessionIfPresent:
    def test_ignores_missing_session(self) -> None:
        with patch(
            "autoware_ml.utils.session._run_tmux",
            side_effect=SessionCommandError("can't find session: default"),
        ):
            _kill_session_if_present("default")

    def test_raises_unexpected_tmux_errors(self) -> None:
        with (
            patch(
                "autoware_ml.utils.session._run_tmux",
                side_effect=SessionCommandError("permission denied"),
            ),
            pytest.raises(SessionCommandError, match="permission denied"),
        ):
            _kill_session_if_present("default")
