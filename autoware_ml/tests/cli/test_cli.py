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
from subprocess import CompletedProcess
from unittest.mock import patch

from typer.testing import CliRunner

from autoware_ml.cli.cli import app
from autoware_ml.cli import cli
from autoware_ml.utils.cli import (
    adjust_argv,
    complete_config_value,
    complete_path_value,
    complete_session_command_value,
    complete_session_name_value,
    expand_config_path,
    parse_extra_args,
    resolve_config_reference,
)
from autoware_ml.utils.cli import helpers

SAMPLE_CONFIG_NAME = "calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2"
SAMPLE_CONFIG_PATH = f"tasks/{SAMPLE_CONFIG_NAME}"
SAMPLE_SESSION_NAME = "calibration-status-train"


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
        result = expand_config_path(SAMPLE_CONFIG_NAME, "tasks")
        assert result == SAMPLE_CONFIG_PATH

    def test_keep_existing_prefix(self) -> None:
        """Test that paths with existing prefix are returned unchanged."""
        result = expand_config_path(SAMPLE_CONFIG_PATH, "tasks")
        assert result == SAMPLE_CONFIG_PATH


class TestResolveConfigReference:
    """Tests for resolve_config_reference."""

    def test_resolve_bundled_config_name(self) -> None:
        """Task shorthands should be expanded under the bundled config prefix."""
        config_path, config_name, hydra_overrides = resolve_config_reference(
            SAMPLE_CONFIG_NAME, "tasks"
        )
        assert config_path is None
        assert config_name == SAMPLE_CONFIG_PATH
        assert hydra_overrides == []

    def test_resolve_packaged_yaml_path(self) -> None:
        """Packaged config paths should resolve back to the bundled Hydra namespace."""
        config_path, config_name, hydra_overrides = resolve_config_reference(
            "autoware_ml/configs/tasks/calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2.yaml",
            "tasks",
        )

        assert config_path is None
        assert (
            config_name
            == "tasks/calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2"
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
        (config_root / "base.yaml").write_text("", encoding="utf-8")
        (config_root / "resnet18_t4dataset_j6gen2.yaml").write_text("", encoding="utf-8")
        monkeypatch.setattr(helpers, "CONFIGS_ROOT", tmp_path)

        completions = complete_config_value(
            "calibration_status/calibration_status_classifier/resnet18_t", "tasks"
        )

        assert completions == [SAMPLE_CONFIG_NAME]
        assert "calibration_status/calibration_status_classifier/base" not in completions

    def test_complete_prefixed_bundled_configs(self, tmp_path: Path, monkeypatch) -> None:
        """Bundled configs should also complete when the tasks/ prefix is included."""
        config_root = tmp_path / "tasks" / "calibration_status" / "calibration_status_classifier"
        config_root.mkdir(parents=True)
        (config_root / "resnet18_t4dataset_j6gen2.yaml").write_text("", encoding="utf-8")
        monkeypatch.setattr(helpers, "CONFIGS_ROOT", tmp_path)

        completions = complete_config_value(
            "tasks/calibration_status/calibration_status_classifier/resnet18_t", "tasks"
        )

        assert completions == [SAMPLE_CONFIG_PATH]

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
        (config_root / "resnet18_t4dataset_j6gen2.yaml").write_text("", encoding="utf-8")
        (tmp_path / "local.yaml").write_text("", encoding="utf-8")
        monkeypatch.setattr(helpers, "CONFIGS_ROOT", tmp_path)
        monkeypatch.chdir(tmp_path)

        completions = complete_config_value("", "tasks")

        assert SAMPLE_CONFIG_NAME in completions
        assert "./local.yaml" not in completions


class TestAdjustArgv:
    """Tests for adjust_argv."""

    def test_clean_argv_input(self) -> None:
        """Test that clean argv input is returned unchanged."""
        argv = [
            "--config-name",
            SAMPLE_CONFIG_NAME,
            "--batch-size",
            "32",
        ]
        result = adjust_argv(argv)
        assert result == [
            "--config-name",
            SAMPLE_CONFIG_NAME,
            "--batch-size",
            "32",
        ]


class TestCliCommands:
    """Tests for top-level CLI commands."""

    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_deploy_requires_checkpoint(self) -> None:
        """Deploy should reject invocations without an explicit checkpoint."""
        result = self.runner.invoke(
            app,
            [
                "deploy",
                "--config-name",
                SAMPLE_CONFIG_NAME,
            ],
        )

        assert result.exit_code != 0
        assert "Missing option" in result.output or "+checkpoint" in result.output

    def test_test_requires_checkpoint(self) -> None:
        """Test command should require an explicit checkpoint."""
        result = self.runner.invoke(
            app,
            [
                "test",
                "--config-name",
                SAMPLE_CONFIG_NAME,
            ],
        )

        assert result.exit_code != 0
        assert "Missing option" in result.output or "+checkpoint" in result.output

    def test_test_runs_script(self) -> None:
        """Test command should dispatch to the evaluation script."""
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "test",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "+checkpoint=model.ckpt",
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with("autoware_ml.scripts.test", "main")

    def test_mlflow_ui_runs_script(self) -> None:
        """MLflow UI should dispatch through the mlflow command group."""
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(app, ["mlflow", "ui", "--port", "6000"])

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.mlflow",
            "run_mlflow_ui",
            host="0.0.0.0",
            port=6000,
            db_path="mlruns/mlflow.db",
        )

    def test_mlflow_export_runs_script(self) -> None:
        """MLflow export should dispatch the extraction helper."""
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "mlflow",
                    "export",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.mlflow",
            "export_experiment_from_db",
            db_path="mlruns/mlflow.db",
            experiment_name=None,
            config_name=SAMPLE_CONFIG_NAME,
            export_dir=None,
            override=False,
        )

    def test_create_dataset_runs_runner(self) -> None:
        """Dataset generation should dispatch through the dataset tooling package."""
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "create-dataset",
                    "--dataset",
                    "nuscenes",
                    "--task",
                    "calibration_status",
                    "--root-path",
                    "/data/nuscenes",
                    "--out-dir",
                    "/tmp/out",
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.create_dataset",
            "main",
            dataset="nuscenes",
            tasks=["calibration_status"],
            root_path="/data/nuscenes",
            out_dir="/tmp/out",
        )

    def test_session_start_runs_script(self) -> None:
        """Session start should dispatch to the managed-session helper."""
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "session",
                    "start",
                    "--name",
                    SAMPLE_SESSION_NAME,
                    "--cwd",
                    "/workspace",
                    "--",
                    "train",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                ],
            )

        assert result.exit_code == 0
        assert (
            f"View live output with: autoware-ml session attach --name {SAMPLE_SESSION_NAME}"
            in result.output
        )
        assert "Press Ctrl+C in the viewer to return without stopping the task." in result.output
        assert (
            f"Stop the task with: autoware-ml session stop --name {SAMPLE_SESSION_NAME}"
            in result.output
        )
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.session",
            "start_session",
            name=SAMPLE_SESSION_NAME,
            command_args=["train", "--config-name", SAMPLE_CONFIG_NAME],
            cwd="/workspace",
            attach=False,
            raw=False,
        )

    def test_session_start_runs_raw_script(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "session",
                    "start",
                    "--name",
                    "docs",
                    "--raw",
                    "--cwd",
                    "/workspace",
                    "--",
                    "zensical",
                    "serve",
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.session",
            "start_session",
            name="docs",
            command_args=["zensical", "serve"],
            cwd="/workspace",
            attach=False,
            raw=True,
        )

    def test_session_attach_runs_script(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(app, ["session", "attach", "--name", SAMPLE_SESSION_NAME])

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.session",
            "attach_session",
            name=SAMPLE_SESSION_NAME,
        )

    def test_session_detach_runs_script(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(app, ["session", "detach", "--name", SAMPLE_SESSION_NAME])

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.session",
            "detach_session",
            name=SAMPLE_SESSION_NAME,
        )

    def test_session_ls_runs_script(self) -> None:
        with patch(
            "autoware_ml.cli.cli.run_lazy_script",
            return_value=f"{SAMPLE_SESSION_NAME}\t0\t1\tTue Mar 17",
        ) as run_lazy_script_mock:
            result = self.runner.invoke(app, ["session", "ls"])

        assert result.exit_code == 0
        assert SAMPLE_SESSION_NAME in result.output
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.session",
            "list_sessions",
        )

    def test_session_ls_handles_no_sessions(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script", return_value="") as run_lazy_script_mock:
            result = self.runner.invoke(app, ["session", "ls"])

        assert result.exit_code == 0
        assert result.output == ""
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.session",
            "list_sessions",
        )

    def test_session_stop_runs_script(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(app, ["session", "stop", "--name", SAMPLE_SESSION_NAME])

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.session",
            "stop_session",
            name=SAMPLE_SESSION_NAME,
        )

    def test_session_start_reports_clean_error(self) -> None:
        with patch(
            "autoware_ml.cli.cli.run_lazy_script",
            side_effect=RuntimeError(
                "A command is required, e.g. autoware-ml session start --name train -- train --config-name ..."
            ),
        ):
            result = self.runner.invoke(app, ["session", "start", "--name", SAMPLE_SESSION_NAME])

        assert result.exit_code == 1
        assert "A command is required" in result.output

    def test_argv_with_spaces(self) -> None:
        """Test that combined argv items are split into tokens."""
        argv = ["arg1 arg2", "arg3"]
        result = adjust_argv(argv)
        assert result == ["arg1", "arg2", "arg3"]

    def test_argv_with_escape_characters(self) -> None:
        """Test that quoted combined argv items are handled correctly."""
        argv = ['--message "hello world"']
        result = adjust_argv(argv)
        assert result == ["--message", "hello world"]

    def test_argv_with_unresolved_variable(self) -> None:
        """Test that unresolved variables starting with $ are filtered out."""
        argv = ["arg1", "${input:arguments}", "arg2"]
        result = adjust_argv(argv)
        assert result == ["arg1", "arg2"]

    def test_argv_preserves_hydra_list_override(self) -> None:
        """Test that Hydra overrides with spaces remain intact."""
        argv = ["trainer.devices=[0, 1]", "trainer.strategy=ddp"]
        result = adjust_argv(argv)
        assert result == ["trainer.devices=[0, 1]", "trainer.strategy=ddp"]

    def test_argv_splits_vscode_debug_argument_string(self) -> None:
        """Test that VS Code prompt-style combined arguments are tokenized."""
        argv = ["trainer.devices=[0, 1] trainer.strategy=ddp"]
        result = adjust_argv(argv)
        assert result == ["trainer.devices=[0, 1]", "trainer.strategy=ddp"]


class TestResolveHydraArgv:
    def test_does_not_inject_run_config_name_override(self) -> None:
        hydra_argv = cli.resolve_hydra_argv(
            SAMPLE_CONFIG_NAME,
            "tasks",
        )

        assert not any(arg.startswith("run_config_name=") for arg in hydra_argv)

    def test_adds_config_path_for_external_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "custom_train.yaml"
        config_file.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")

        hydra_argv = cli.resolve_hydra_argv(str(config_file), "tasks")

        assert hydra_argv[:4] == [
            "--config-name",
            "custom_train",
            "--config-path",
            str(tmp_path),
        ]
        assert "hydra.searchpath=[pkg://autoware_ml.configs]" in hydra_argv
        assert not any(arg.startswith("run_config_name=") for arg in hydra_argv)

    def test_preserves_explicit_hydra_searchpath_override(self, tmp_path: Path) -> None:
        config_file = tmp_path / "custom_train.yaml"
        config_file.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")

        hydra_argv = cli.resolve_hydra_argv(
            str(config_file),
            "tasks",
            extra_args=["hydra.searchpath=[file:///tmp/custom]"],
        )

        assert hydra_argv.count("hydra.searchpath=[file:///tmp/custom]") == 1
        assert "hydra.searchpath=[pkg://autoware_ml.configs]" not in hydra_argv


class TestSessionCompletion:
    def test_suggests_root_commands(self) -> None:
        suggestions = complete_session_command_value([], "tr")
        assert suggestions == ["train"]

    def test_suggests_config_names_after_flag(self) -> None:
        suggestions = complete_session_command_value(
            ["train", "--config-name"],
            "calibration_status/calibration_status_classifier/resnet18_t",
        )
        assert SAMPLE_CONFIG_NAME in suggestions

    def test_suggests_command_options(self) -> None:
        suggestions = complete_session_command_value(["deploy"], "--c")
        assert suggestions == ["--config-name"]

    def test_suggests_checkpoint_paths_after_checkpoint_flag(self, tmp_path: Path) -> None:
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        (checkpoints_dir / "best.ckpt").write_text("ckpt")
        (checkpoints_dir / "notes.txt").write_text("txt")

        suggestions = complete_session_command_value(
            ["deploy", "+checkpoint"], f"{checkpoints_dir}/b"
        )

        assert suggestions == [f"{checkpoints_dir}/best.ckpt"]

    def test_suggests_create_dataset_directory_values(self, tmp_path: Path) -> None:
        dataset_root = tmp_path / "dataset_root"
        dataset_root.mkdir()
        (tmp_path / "artifact.txt").write_text("txt")

        suggestions = complete_session_command_value(
            ["create-dataset", "--root-path"], f"{tmp_path}/"
        )

        assert suggestions == [f"{tmp_path}/dataset_root/"]

    def test_suggests_mlflow_ui_options(self) -> None:
        suggestions = complete_session_command_value(["mlflow", "ui"], "--")
        assert suggestions == ["--db-path", "--host", "--port"]

    def test_suggests_mlflow_ui_db_path_values(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mlruns.db"
        db_path.write_text("db")

        suggestions = complete_session_command_value(["mlflow", "ui", "--db-path"], f"{tmp_path}/")

        assert suggestions == [f"{tmp_path}/mlruns.db"]

    def test_suggests_mlflow_export_options(self) -> None:
        suggestions = complete_session_command_value(["mlflow", "export"], "--")
        assert suggestions == [
            "--db-path",
            "--experiment-name",
            "--config-name",
            "--export-dir",
            "--override",
        ]

    def test_suggests_mlflow_export_config_names_after_flag(self) -> None:
        suggestions = complete_session_command_value(
            ["mlflow", "export", "--config-name"],
            "calibration_status/calibration_status_classifier/resnet18_t",
        )

        assert SAMPLE_CONFIG_NAME in suggestions

    def test_suggests_mlflow_export_directory_values(self, tmp_path: Path) -> None:
        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()

        suggestions = complete_session_command_value(
            ["mlflow", "export", "--export-dir"], f"{tmp_path}/"
        )

        assert suggestions == [f"{tmp_path}/exports/"]


class TestMlflowCliCommands:
    def test_mlflow_export_forwards_override_flag(self) -> None:
        runner = CliRunner()

        with patch.object(cli, "run_lazy_script") as run_lazy_script_mock:
            result = runner.invoke(
                app,
                [
                    "mlflow",
                    "export",
                    "--db-path",
                    "mlruns/mlflow.db",
                    "--experiment-name",
                    "exp",
                    "--override",
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.mlflow",
            "export_experiment_from_db",
            db_path="mlruns/mlflow.db",
            experiment_name="exp",
            config_name=None,
            export_dir=None,
            override=True,
        )


class TestCliMain:
    def test_main_runs_app(self) -> None:
        with patch.object(cli, "app") as app_mock:
            cli.main()

        app_mock.assert_called_once_with()


class TestCliCallback:
    def test_main_callback_configures_logging(self) -> None:
        with patch.object(cli, "setup_logging") as setup_logging_mock:
            cli.main_callback()

        setup_logging_mock.assert_called_once_with()


class TestPathCompletion:
    def test_completes_checkpoint_paths(self, tmp_path: Path) -> None:
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        checkpoint_path = checkpoints_dir / "best.ckpt"
        checkpoint_path.write_text("ckpt")
        (checkpoints_dir / "notes.txt").write_text("txt")

        suggestions = complete_path_value(f"{checkpoints_dir}/b", file_suffixes=(".ckpt",))

        assert suggestions == [f"{checkpoints_dir}/best.ckpt"]

    def test_completes_directories_only(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (tmp_path / "artifact.txt").write_text("txt")

        suggestions = complete_path_value(f"{tmp_path}/", directories_only=True)

        assert suggestions == [f"{tmp_path}/logs/"]


class TestSessionNameCompletion:
    def test_completes_managed_session_names(self) -> None:
        with patch(
            "autoware_ml.utils.cli.helpers.list_tmux_session_names",
            return_value=["default", SAMPLE_SESSION_NAME],
        ):
            suggestions = complete_session_name_value("cal")

        assert suggestions == [SAMPLE_SESSION_NAME]

    def test_returns_empty_when_tmux_is_not_available(self) -> None:
        with patch("autoware_ml.utils.cli.helpers.shutil.which", return_value=None):
            suggestions = complete_session_name_value("fr")

        assert suggestions == []

    def test_returns_empty_when_tmux_server_is_not_running(self) -> None:
        with patch("autoware_ml.utils.cli.helpers.shutil.which", return_value="/usr/bin/tmux"):
            with patch(
                "autoware_ml.utils.cli.helpers.subprocess.run",
                return_value=CompletedProcess(
                    args=[
                        "tmux",
                        "-L",
                        "autoware-ml",
                        "-f",
                        "/dev/null",
                        "list-sessions",
                        "-F",
                        "#{session_name}\t#{@autoware_ml_managed}",
                    ],
                    returncode=1,
                    stdout="",
                    stderr="no server running on /tmp/tmux-1000/default",
                ),
            ):
                suggestions = complete_session_name_value("fr")

        assert suggestions == []

    def test_filters_unmanaged_session_names(self) -> None:
        with patch("autoware_ml.utils.cli.helpers.shutil.which", return_value="/usr/bin/tmux"):
            with patch(
                "autoware_ml.utils.cli.helpers.subprocess.run",
                return_value=CompletedProcess(
                    args=[
                        "tmux",
                        "-L",
                        "autoware-ml",
                        "-f",
                        "/dev/null",
                        "list-sessions",
                        "-F",
                        "#{session_name}\t#{@autoware_ml_managed}",
                    ],
                    returncode=0,
                    stdout="default\t1\nscratch\t\ncalibration-status\t1\n",
                    stderr="",
                ),
            ):
                suggestions = complete_session_name_value("cal")

        assert suggestions == ["calibration-status"]
