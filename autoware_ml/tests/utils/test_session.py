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

"""Unit tests for tmux-backed session helpers."""

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import call, patch

import pytest

from autoware_ml.utils.session import (
    SessionCommandError,
    _build_session_command,
    _build_raw_session_command,
    detach_session,
    list_sessions,
    start_session,
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
        assert "trap detach_client INT" in command
        assert "trap stop_child TERM HUP EXIT" in command

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
                call(["new-session", "-d", "-s", "default"]),
                call(["set-option", "-t", "default", "-q", "@autoware_ml_managed", "1"]),
                call(["send-keys", "-t", "default", expected_command, "C-m"]),
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
                call(["new-session", "-d", "-s", "docs"]),
                call(["set-option", "-t", "docs", "-q", "@autoware_ml_managed", "1"]),
                call(["send-keys", "-t", "docs", expected_command, "C-m"]),
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
