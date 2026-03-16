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

from pathlib import Path

from autoware_ml.utils.cli import (
    adjust_argv,
    complete_config_value,
    expand_config_path,
    parse_extra_args,
    resolve_config_reference,
)
from autoware_ml.utils.cli import helpers


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
        result = expand_config_path(
            "calibration_status/calibration_status_classifier/resnet18_nuscenes", "tasks"
        )
        assert result == "tasks/calibration_status/calibration_status_classifier/resnet18_nuscenes"

    def test_keep_existing_prefix(self) -> None:
        """Test that paths with existing prefix are returned unchanged."""
        result = expand_config_path(
            "tasks/calibration_status/calibration_status_classifier/resnet18_nuscenes", "tasks"
        )
        assert result == "tasks/calibration_status/calibration_status_classifier/resnet18_nuscenes"


class TestResolveConfigReference:
    """Tests for resolve_config_reference."""

    def test_resolve_bundled_config_name(self) -> None:
        """Task shorthands should be expanded under the bundled config prefix."""
        config_path, config_name, hydra_overrides = resolve_config_reference(
            "calibration_status/calibration_status_classifier/resnet18_nuscenes", "tasks"
        )
        assert config_path is None
        assert (
            config_name
            == "tasks/calibration_status/calibration_status_classifier/resnet18_nuscenes"
        )
        assert hydra_overrides == []

    def test_resolve_packaged_yaml_path(self) -> None:
        """Packaged config paths should resolve back to the bundled Hydra namespace."""
        config_path, config_name, hydra_overrides = resolve_config_reference(
            "autoware_ml/configs/tasks/calibration_status/calibration_status_classifier/resnet18_t4dataset.yaml",
            "tasks",
        )

        assert config_path is None
        assert (
            config_name
            == "tasks/calibration_status/calibration_status_classifier/resnet18_t4dataset"
        )
        assert hydra_overrides == []

    def test_resolve_relative_yaml_path(self, tmp_path: Path, monkeypatch) -> None:
        """Relative YAML paths should be converted into Hydra config-path/config-name."""
        config_file = tmp_path / "custom_train.yaml"
        config_file.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        config_path, config_name, hydra_overrides = resolve_config_reference(
            "./custom_train.yaml", "tasks"
        )

        assert config_path == str(tmp_path)
        assert config_name == "custom_train"
        assert hydra_overrides == ["hydra.searchpath=[pkg://autoware_ml.configs]"]

    def test_resolve_absolute_yaml_path(self, tmp_path: Path) -> None:
        """Absolute YAML paths should be converted into Hydra config-path/config-name."""
        config_file = tmp_path / "custom_train.yaml"
        config_file.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")

        config_path, config_name, hydra_overrides = resolve_config_reference(
            str(config_file), "tasks"
        )

        assert config_path == str(tmp_path)
        assert config_name == "custom_train"
        assert hydra_overrides == ["hydra.searchpath=[pkg://autoware_ml.configs]"]


class TestCompleteConfigValue:
    """Tests for config completion helpers."""

    def test_complete_bundled_configs(self, tmp_path: Path, monkeypatch) -> None:
        """Bundled task config names should be suggested without YAML suffixes."""
        config_root = tmp_path / "tasks" / "calibration_status" / "calibration_status_classifier"
        config_root.mkdir(parents=True)
        (config_root / "resnet18_nuscenes.yaml").write_text("", encoding="utf-8")
        (config_root / "resnet18_t4dataset.yaml").write_text("", encoding="utf-8")
        monkeypatch.setattr(helpers, "CONFIGS_ROOT", tmp_path)

        completions = complete_config_value(
            "calibration_status/calibration_status_classifier/resnet18_n", "tasks"
        )

        assert completions == [
            "calibration_status/calibration_status_classifier/resnet18_nuscenes",
        ]

    def test_complete_prefixed_bundled_configs(self, tmp_path: Path, monkeypatch) -> None:
        """Bundled configs should also complete when the tasks/ prefix is included."""
        config_root = tmp_path / "tasks" / "calibration_status" / "calibration_status_classifier"
        config_root.mkdir(parents=True)
        (config_root / "resnet18_nuscenes.yaml").write_text("", encoding="utf-8")
        monkeypatch.setattr(helpers, "CONFIGS_ROOT", tmp_path)

        completions = complete_config_value(
            "tasks/calibration_status/calibration_status_classifier/resnet18_n", "tasks"
        )

        assert completions == [
            "tasks/calibration_status/calibration_status_classifier/resnet18_nuscenes"
        ]

    def test_complete_filesystem_yaml_paths(self, tmp_path: Path, monkeypatch) -> None:
        """Filesystem completion should suggest YAML files for path-like input."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "custom_a.yaml").write_text("", encoding="utf-8")
        (config_dir / "custom_b.yml").write_text("", encoding="utf-8")
        (config_dir / "notes.txt").write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        completions = complete_config_value("./configs/cus", "tasks")

        assert completions == ["./configs/custom_a.yaml", "./configs/custom_b.yml"]

    def test_no_filesystem_completion_without_path_prefix(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Empty completion input should not enumerate the current directory."""
        config_root = tmp_path / "tasks" / "calibration_status" / "calibration_status_classifier"
        config_root.mkdir(parents=True)
        (config_root / "resnet18_nuscenes.yaml").write_text("", encoding="utf-8")
        (tmp_path / "local.yaml").write_text("", encoding="utf-8")
        monkeypatch.setattr(helpers, "CONFIGS_ROOT", tmp_path)
        monkeypatch.chdir(tmp_path)

        completions = complete_config_value("", "tasks")

        assert "calibration_status/calibration_status_classifier/resnet18_nuscenes" in completions
        assert "./local.yaml" not in completions


class TestAdjustArgv:
    """Tests for adjust_argv."""

    def test_clean_argv_input(self) -> None:
        """Test that clean argv input is returned unchanged."""
        argv = [
            "--config-name",
            "calibration_status/calibration_status_classifier/resnet18_nuscenes",
            "--batch-size",
            "32",
        ]
        result = adjust_argv(argv)
        assert result == [
            "--config-name",
            "calibration_status/calibration_status_classifier/resnet18_nuscenes",
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
