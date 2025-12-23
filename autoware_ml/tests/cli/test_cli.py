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

"""Unit tests for CLI utilities."""

from autoware_ml.utils.cli import adjust_argv, expand_config_path, parse_extra_args


class TestParseExtraArgs:
    """Tests for parse_extra_args."""

    def test_parse_extra_args(self) -> None:
        """Test that parse_extra_args returns a dictionary of typed kwargs."""
        extra_args = ["--model-name", "resnet18", "--batch-size", "128", "--epochs", "100"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"model_name": "resnet18", "batch_size": 128, "epochs": 100}

    def test_parse_extra_args_with_boolean_flag(self) -> None:
        """Test that parse_extra_args returns a dictionary of typed kwargs with boolean flags."""
        extra_args = ["--verbose", "--debug"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"verbose": True, "debug": True}

    def test_parse_extra_args_with_numeric_value(self) -> None:
        """Test that parse_extra_args returns a dictionary of typed kwargs with numeric values."""
        extra_args = ["--batch-size", "128", "--epochs", "100"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"batch_size": 128, "epochs": 100}

    def test_parse_extra_args_with_float_value(self) -> None:
        """Test that parse_extra_args returns a dictionary of typed kwargs with float values."""
        extra_args = ["--learning-rate", "0.001"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"learning_rate": 0.001}

    def test_parse_extra_args_with_string_value(self) -> None:
        """Test that parse_extra_args returns a dictionary of typed kwargs with string values."""
        extra_args = ["--model-name", "resnet18"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"model_name": "resnet18"}

    def test_single_dash_key(self) -> None:
        """Test that parse_extra_args returns a dictionary of typed kwargs with single dash key."""
        extra_args = ["-learning-rate", "0.001"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"learning_rate": 0.001}

    def test_multiple_dash_key(self) -> None:
        """Test that parse_extra_args returns a dictionary of typed kwargs with multiple dash key."""
        extra_args = ["---model-name", "resnet18"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"model_name": "resnet18"}

    def test_no_dash_key(self) -> None:
        """Test that parse_extra_args returns a dictionary of typed kwargs with no dash key."""
        extra_args = ["learning-rate", "0.001"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"learning_rate": 0.001}

    def test_mixed_dash_key(self) -> None:
        """Test that parse_extra_args returns a dictionary of typed kwargs with mixed dash key."""
        extra_args = ["--model-name", "resnet18", "learning-rate", "0.001", "---batch-size", "128"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"model_name": "resnet18", "learning_rate": 0.001, "batch_size": 128}


class TestExpandConfigPath:
    """Tests for expand_config_path."""

    def test_expand_short_path(self) -> None:
        """Test that short config path is expanded with prefix."""
        result = expand_config_path("calibration_status/resnet18_nuscenes", "tasks")
        assert result == "tasks/calibration_status/resnet18_nuscenes"

    def test_keep_existing_prefix(self) -> None:
        """Test that paths with existing prefix are returned unchanged."""
        result = expand_config_path("tasks/calibration_status/resnet18_nuscenes", "tasks")
        assert result == "tasks/calibration_status/resnet18_nuscenes"


class TestAdjustArgv:
    """Tests for adjust_argv."""

    def test_clean_argv_input(self) -> None:
        """Test that clean argv input is returned unchanged."""
        argv = ["--config-name", "calibration_status/resnet18_nuscenes", "--batch-size", "32"]
        result = adjust_argv(argv)
        assert result == [
            "--config-name",
            "calibration_status/resnet18_nuscenes",
            "--batch-size",
            "32",
        ]

    def test_argv_with_spaces(self) -> None:
        """Test that argv with spaces is split correctly."""
        argv = ["arg1 arg2", "arg3"]
        result = adjust_argv(argv)
        assert result == ["arg1", "arg2", "arg3"]

    def test_argv_with_escape_characters(self) -> None:
        """Test that escape characters are removed."""
        argv = ["arg1\\ arg2"]
        result = adjust_argv(argv)
        assert result == ["arg1", "arg2"]

    def test_argv_with_unresolved_variable(self) -> None:
        """Test that unresolved variables starting with $ are filtered out."""
        argv = ["arg1", "${input:arguments}", "arg2"]
        result = adjust_argv(argv)
        assert result == ["arg1", "arg2"]
