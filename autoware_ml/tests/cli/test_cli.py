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

import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

import __main__
import autoware_ml.cli.cli as cli
import autoware_ml.cli.runtime as cli_runtime
import autoware_ml.utils.cli.helpers as helpers
from autoware_ml.cli.cli import app
from autoware_ml.utils.cli.helpers import (
    adjust_argv,
    complete_config_value,
    complete_path_value,
    complete_session_command_value,
    complete_session_name_value,
    expand_config_path,
    parse_extra_args,
    resolve_config_reference,
)
from autoware_ml.utils.session import SessionCommandError

SAMPLE_CONFIG_NAME = "calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2"
SAMPLE_CONFIG_PATH = f"tasks/{SAMPLE_CONFIG_NAME}"
SAMPLE_SESSION_NAME = "calibration-status-train"


class TestParseExtraArgs:
    """Tests for parse_extra_args."""

    def test_parse_extra_args(self) -> None:
        extra_args = ["--model-name", "resnet18", "--batch-size", "128", "--epochs", "100"]
        kwargs = parse_extra_args(extra_args)
        assert kwargs == {"model_name": "resnet18", "batch_size": 128, "epochs": 100}

    def test_parse_extra_args_with_boolean_flag(self) -> None:
        kwargs = parse_extra_args(["--verbose", "--debug"])
        assert kwargs == {"verbose": True, "debug": True}

    def test_parse_extra_args_with_float_value(self) -> None:
        kwargs = parse_extra_args(["--learning-rate", "0.001"])
        assert kwargs == {"learning_rate": 0.001}

    def test_parse_extra_args_with_negative_numeric_values(self) -> None:
        kwargs = parse_extra_args(["--offset", "-5", "--scale", "-1.5e-2"])
        assert kwargs == {"offset": -5, "scale": -1.5e-2}


class TestExpandConfigPath:
    """Tests for expand_config_path."""

    def test_expand_short_path(self) -> None:
        result = expand_config_path(SAMPLE_CONFIG_NAME, "tasks")
        assert result == SAMPLE_CONFIG_PATH

    def test_keep_existing_prefix(self) -> None:
        result = expand_config_path(SAMPLE_CONFIG_PATH, "tasks")
        assert result == SAMPLE_CONFIG_PATH


class TestResolveConfigReference:
    """Tests for resolve_config_reference."""

    def test_resolve_bundled_config_name(self) -> None:
        config_path, config_name, hydra_overrides = resolve_config_reference(
            SAMPLE_CONFIG_NAME, "tasks"
        )
        assert config_path is None
        assert config_name == SAMPLE_CONFIG_PATH
        assert hydra_overrides == []

    def test_resolve_packaged_yaml_path(self) -> None:
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

    def test_complete_filesystem_yaml_paths(self, tmp_path: Path, monkeypatch) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "custom_a.yaml").write_text("", encoding="utf-8")
        (config_dir / "notes.txt").write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        completions = complete_config_value("./configs/cus", "tasks")

        assert completions == ["./configs/custom_a.yaml"]


class TestAdjustArgv:
    """Tests for adjust_argv."""

    def test_cli_syntax(self) -> None:
        # Normal argv array - Hydra overrides with brackets pass through unchanged.
        argv = ["--config-name", SAMPLE_CONFIG_NAME, "trainer.devices=[0,1]", "model.lr=0.001"]
        assert adjust_argv(argv) == argv

    def test_vscode_syntax(self) -> None:
        # VS Code passes all args as a single space-delimited string.
        # Brackets must be preserved as a single token.
        result = adjust_argv(
            [f"--config-name {SAMPLE_CONFIG_NAME} trainer.devices=[0,1] model.lr=0.001"]
        )
        assert result == [
            "--config-name",
            SAMPLE_CONFIG_NAME,
            "trainer.devices=[0,1]",
            "model.lr=0.001",
        ]


class TestCliCommands:
    """Tests for top-level CLI commands."""

    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_train_dispatches_to_runtime_module(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "train",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "+trainer.strategy=ddp",
                    "trainer.devices=2",
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            cli.CLI_RUNTIME_MODULE,
            "run_hydra_entrypoint",
            entrypoint_module=cli.TRAIN_ENTRYPOINT_MODULE,
            config_name=SAMPLE_CONFIG_NAME,
            stage="train",
            extra_args=["+trainer.strategy=ddp", "trainer.devices=2"],
            hydra_overrides=[],
            resume_checkpoint=None,
            new_run=False,
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_train_runs_with_weights(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                ["train", "--config-name", SAMPLE_CONFIG_NAME, "--weights", "seg.ckpt"],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            cli.CLI_RUNTIME_MODULE,
            "run_hydra_entrypoint",
            entrypoint_module=cli.TRAIN_ENTRYPOINT_MODULE,
            config_name=SAMPLE_CONFIG_NAME,
            stage="train",
            extra_args=[],
            hydra_overrides=["+weights=[seg.ckpt]"],
            resume_checkpoint=None,
            new_run=False,
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_train_runs_with_resume_checkpoint(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "last.ckpt"
        checkpoint_path.touch()

        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "train",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "--resume-checkpoint",
                    str(checkpoint_path),
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            cli.CLI_RUNTIME_MODULE,
            "run_hydra_entrypoint",
            entrypoint_module=cli.TRAIN_ENTRYPOINT_MODULE,
            config_name=SAMPLE_CONFIG_NAME,
            stage="train",
            extra_args=[],
            hydra_overrides=[f"+resume_checkpoint={checkpoint_path.resolve()}"],
            resume_checkpoint=str(checkpoint_path.resolve()),
            new_run=False,
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_train_resolves_relative_resume_checkpoint(self, tmp_path: Path, monkeypatch) -> None:
        checkpoint_path = tmp_path / "last.ckpt"
        checkpoint_path.touch()
        monkeypatch.chdir(tmp_path)

        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                ["train", "--config-name", SAMPLE_CONFIG_NAME, "--resume-checkpoint", "last.ckpt"],
            )

        assert result.exit_code == 0
        resume_kwarg = run_lazy_script_mock.call_args.kwargs["resume_checkpoint"]
        assert resume_kwarg == str(checkpoint_path.resolve())

    def test_train_rejects_missing_resume_checkpoint(self) -> None:
        result = self.runner.invoke(
            app,
            [
                "train",
                "--config-name",
                SAMPLE_CONFIG_NAME,
                "--resume-checkpoint",
                "/nonexistent/last.ckpt",
            ],
        )
        assert result.exit_code != 0

    def test_train_rejects_new_run_without_resume_checkpoint(self) -> None:
        result = self.runner.invoke(
            app,
            ["train", "--config-name", SAMPLE_CONFIG_NAME, "--new-run"],
        )
        assert result.exit_code != 0

    def test_train_rejects_weights_and_resume_checkpoint_together(self) -> None:
        result = self.runner.invoke(
            app,
            [
                "train",
                "--config-name",
                SAMPLE_CONFIG_NAME,
                "--weights",
                "seg.ckpt",
                "--resume-checkpoint",
                "last.ckpt",
            ],
        )
        assert result.exit_code != 0

    def test_deploy_requires_weights(self) -> None:
        result = self.runner.invoke(app, ["deploy", "--config-name", SAMPLE_CONFIG_NAME])
        assert result.exit_code != 0
        assert "--weights" in result.output

    def test_deploy_runs_with_single_weights(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                ["deploy", "--config-name", SAMPLE_CONFIG_NAME, "--weights", "model.ckpt"],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            cli.CLI_RUNTIME_MODULE,
            "run_hydra_entrypoint",
            entrypoint_module=cli.DEPLOY_ENTRYPOINT_MODULE,
            config_name=SAMPLE_CONFIG_NAME,
            stage="deploy",
            extra_args=[],
            hydra_overrides=["+weights=[model.ckpt]"],
            checkpoints=["model.ckpt"],
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_deploy_runs_with_multiple_weights(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "deploy",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "--weights",
                    "seg.ckpt",
                    "--weights",
                    "det.ckpt",
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            cli.CLI_RUNTIME_MODULE,
            "run_hydra_entrypoint",
            entrypoint_module=cli.DEPLOY_ENTRYPOINT_MODULE,
            config_name=SAMPLE_CONFIG_NAME,
            stage="deploy",
            extra_args=[],
            hydra_overrides=["+weights=[seg.ckpt,det.ckpt]"],
            checkpoints=["seg.ckpt", "det.ckpt"],
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_test_requires_weights(self) -> None:
        result = self.runner.invoke(app, ["test", "--config-name", SAMPLE_CONFIG_NAME])
        assert result.exit_code != 0
        assert "--weights" in result.output

    def test_test_runs_with_weights(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "test",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "--weights",
                    "seg.ckpt",
                    "--weights",
                    "det.ckpt",
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            cli.CLI_RUNTIME_MODULE,
            "run_hydra_entrypoint",
            entrypoint_module=cli.TEST_ENTRYPOINT_MODULE,
            config_name=SAMPLE_CONFIG_NAME,
            stage="test",
            extra_args=[],
            # Test forces a single device by default.
            hydra_overrides=["+weights=[seg.ckpt,det.ckpt]", "++trainer.devices=1"],
            checkpoint="det.ckpt",
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_test_forces_single_device_over_many_devices(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "test",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "--weights",
                    "seg.ckpt",
                    "trainer.devices=4",
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            cli.CLI_RUNTIME_MODULE,
            "run_hydra_entrypoint",
            entrypoint_module=cli.TEST_ENTRYPOINT_MODULE,
            config_name=SAMPLE_CONFIG_NAME,
            stage="test",
            extra_args=["trainer.devices=4"],
            # The forcing override is appended after extra args, so runtime applies it last
            # and a single device wins over the user's trainer.devices=4.
            hydra_overrides=["+weights=[seg.ckpt]", "++trainer.devices=1"],
            checkpoint="seg.ckpt",
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_test_use_config_devices_keeps_config(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "test",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "--weights",
                    "seg.ckpt",
                    "--use-config-devices",
                    "trainer.devices=4",
                ],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            cli.CLI_RUNTIME_MODULE,
            "run_hydra_entrypoint",
            entrypoint_module=cli.TEST_ENTRYPOINT_MODULE,
            config_name=SAMPLE_CONFIG_NAME,
            stage="test",
            extra_args=["trainer.devices=4"],
            # No forcing override: trainer.devices from config / extra args is honored.
            hydra_overrides=["+weights=[seg.ckpt]"],
            checkpoint="seg.ckpt",
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_test_single_device_override_wins_in_composed_config(self) -> None:
        # End-to-end: the override order the CLI + runtime produce (user devices first,
        # forcing override last) resolves to a single device in the real config.
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
            cfg = compose(
                config_name=f"tasks/{SAMPLE_CONFIG_NAME}",
                overrides=["trainer.devices=4", "++trainer.devices=1"],
            )
        assert cfg.trainer.devices == 1

    def test_mlflow_ui_runs_script(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(app, ["mlflow", "ui", "--port", "6000"])

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.mlflow_wrapper",
            "run_mlflow_ui",
            host="0.0.0.0",
            port=6000,
            db_path="mlruns/mlflow.db",
        )

    def test_mlflow_export_runs_script(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app, ["mlflow", "export", "--config-name", SAMPLE_CONFIG_NAME]
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.mlflow_wrapper",
            "export_experiment_from_db",
            db_path="mlruns/mlflow.db",
            experiment_name=None,
            config_name=SAMPLE_CONFIG_NAME,
            export_dir=None,
            override=False,
        )

    def test_create_dataset_runs_runner(self) -> None:
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
        run_lazy_script_mock.assert_called_once_with(
            "autoware_ml.scripts.session",
            "start_session",
            name=SAMPLE_SESSION_NAME,
            command_args=["train", "--config-name", SAMPLE_CONFIG_NAME],
            cwd="/workspace",
            attach=False,
            raw=False,
        )

    def test_session_start_reports_clean_error(self) -> None:
        with patch(
            "autoware_ml.cli.cli.run_lazy_script",
            side_effect=SessionCommandError("A command is required"),
        ):
            result = self.runner.invoke(app, ["session", "start", "--name", SAMPLE_SESSION_NAME])

        assert result.exit_code == 1
        assert "A command is required" in result.output


class TestCliRuntime:
    def test_run_hydra_entrypoint_uses_runtime_script_argv_for_distributed_launches(self) -> None:
        def assert_runtime_context(*args, **kwargs) -> None:
            assert cli_runtime.sys.argv[0] == "autoware_ml.scripts.train"
            assert __main__.__spec__ is not None
            assert __main__.__spec__.name == "autoware_ml.scripts.train"

        with (
            patch(
                "autoware_ml.cli.runtime.prepare_runtime_environment",
                return_value={
                    "AUTOWARE_ML_RUN_ID": "run-1",
                    "AUTOWARE_ML_HYDRA_RUN_DIR": "/tmp/hydra",
                },
            ),
            patch(
                "autoware_ml.cli.runtime.run_lazy_script",
                side_effect=assert_runtime_context,
            ) as run_lazy_script_mock,
        ):
            cli_runtime.run_hydra_entrypoint(
                cli.TRAIN_ENTRYPOINT_MODULE,
                SAMPLE_CONFIG_NAME,
                "train",
                extra_args=["+trainer.strategy=ddp", "trainer.devices=2"],
            )

        run_lazy_script_mock.assert_called_once_with("autoware_ml.scripts.train", "main")
        assert "+trainer.strategy=ddp" in cli_runtime.sys.argv
        assert "trainer.devices=2" in cli_runtime.sys.argv

    def test_prepare_runtime_environment_uses_deploy_config_experiment_for_multi_checkpoint(
        self,
        tmp_path: Path,
    ) -> None:
        config_name = "multi/ptv3/voxel012"
        experiment_name = "multi_ptv3_voxel012"
        cfg = OmegaConf.create({"logger": {"tracking_uri": "sqlite:///mlruns/mlflow.db"}})

        with (
            patch(
                "autoware_ml.cli.runtime.resolve_config_reference",
                return_value=(None, f"tasks/{config_name}", []),
            ),
            patch("autoware_ml.cli.runtime.initialize_config_module", return_value=nullcontext()),
            patch("autoware_ml.cli.runtime.compose", return_value=cfg),
            patch(
                "autoware_ml.cli.runtime.resolve_deploy_lineage",
                return_value=(
                    experiment_name,
                    None,
                    [{"run_id": "seg-run"}, {"run_id": "det-run"}],
                ),
            ) as resolve_deploy_lineage_mock,
            patch(
                "autoware_ml.cli.runtime.prepare_run_context",
                return_value=SimpleNamespace(run_id="deploy-run", hydra_dir=tmp_path / "hydra"),
            ) as prepare_run_context_mock,
        ):
            env_updates = cli_runtime.prepare_runtime_environment(
                config_name,
                "tasks",
                "deploy",
                checkpoints=["seg.ckpt", "det.ckpt"],
            )

        resolve_deploy_lineage_mock.assert_called_once_with(
            config_name,
            [Path("seg.ckpt"), Path("det.ckpt")],
        )
        prepare_run_context_mock.assert_called_once()
        assert prepare_run_context_mock.call_args.kwargs["experiment_name"] == experiment_name
        assert prepare_run_context_mock.call_args.kwargs["parent_run_id"] is None
        assert prepare_run_context_mock.call_args.kwargs["extra_tags"]["source_run_ids"] == (
            "seg-run,det-run"
        )
        assert env_updates["AUTOWARE_ML_RUN_ID"] == "deploy-run"

    @contextmanager
    def _patched_logger_config(self, config_name: str):
        cfg = OmegaConf.create({"logger": {"tracking_uri": "sqlite:///mlruns/mlflow.db"}})
        with (
            patch(
                "autoware_ml.cli.runtime.resolve_config_reference",
                return_value=(None, f"tasks/{config_name}", []),
            ),
            patch("autoware_ml.cli.runtime.initialize_config_module", return_value=nullcontext()),
            patch("autoware_ml.cli.runtime.compose", return_value=cfg),
        ):
            yield

    def test_prepare_runtime_environment_reuses_source_run_for_resume(self, tmp_path: Path) -> None:
        config_name = "multi/ptv3/voxel012"
        with (
            self._patched_logger_config(config_name),
            patch(
                "autoware_ml.cli.runtime.load_run_metadata",
                return_value={"run_id": "source-run", "config_name": config_name},
            ),
            patch(
                "autoware_ml.cli.runtime.load_run_context",
                return_value=SimpleNamespace(run_id="source-run", hydra_dir=tmp_path / "hydra"),
            ) as load_run_context_mock,
            patch("autoware_ml.cli.runtime.prepare_run_context") as prepare_run_context_mock,
        ):
            env_updates = cli_runtime.prepare_runtime_environment(
                config_name,
                "tasks",
                "train",
                resume_checkpoint=str(tmp_path / "checkpoints" / "last.ckpt"),
            )

        prepare_run_context_mock.assert_not_called()
        load_run_context_mock.assert_called_once()
        assert env_updates["AUTOWARE_ML_RUN_ID"] == "source-run"
        assert env_updates["AUTOWARE_ML_HYDRA_RUN_DIR"] == str(tmp_path / "hydra")

    def test_prepare_runtime_environment_resume_new_run_creates_fresh_run(
        self, tmp_path: Path
    ) -> None:
        config_name = "multi/ptv3/voxel012"
        with (
            self._patched_logger_config(config_name),
            patch(
                "autoware_ml.cli.runtime.prepare_run_context",
                return_value=SimpleNamespace(run_id="fork-run", hydra_dir=tmp_path / "hydra"),
            ) as prepare_run_context_mock,
        ):
            env_updates = cli_runtime.prepare_runtime_environment(
                config_name,
                "tasks",
                "train",
                resume_checkpoint=str(tmp_path / "last.ckpt"),
                new_run=True,
            )

        prepare_run_context_mock.assert_called_once()
        assert env_updates["AUTOWARE_ML_RUN_ID"] == "fork-run"

    def test_prepare_runtime_environment_resume_without_metadata_fails(
        self, tmp_path: Path
    ) -> None:
        config_name = "multi/ptv3/voxel012"
        with (
            self._patched_logger_config(config_name),
            patch("autoware_ml.cli.runtime.load_run_metadata", return_value=None),
            pytest.raises(ValueError, match="--new-run"),
        ):
            cli_runtime.prepare_runtime_environment(
                config_name,
                "tasks",
                "train",
                resume_checkpoint=str(tmp_path / "last.ckpt"),
            )

    def test_prepare_runtime_environment_resume_rejects_config_mismatch(
        self, tmp_path: Path
    ) -> None:
        with (
            self._patched_logger_config("multi/ptv3/voxel012"),
            patch(
                "autoware_ml.cli.runtime.load_run_metadata",
                return_value={"run_id": "source-run", "config_name": "detection3d/other"},
            ),
            pytest.raises(ValueError, match="detection3d/other"),
        ):
            cli_runtime.prepare_runtime_environment(
                "multi/ptv3/voxel012",
                "tasks",
                "train",
                resume_checkpoint=str(tmp_path / "last.ckpt"),
            )


class TestResolveHydraArgv:
    def test_does_not_inject_run_config_name_override(self) -> None:
        hydra_argv = cli_runtime.resolve_hydra_argv(SAMPLE_CONFIG_NAME, "tasks")
        assert not any(arg.startswith("run_config_name=") for arg in hydra_argv)

    def test_adds_config_path_for_external_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "custom_train.yaml"
        config_file.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")

        hydra_argv = cli_runtime.resolve_hydra_argv(str(config_file), "tasks")

        assert hydra_argv[:4] == [
            "--config-name",
            "custom_train",
            "--config-path",
            str(tmp_path),
        ]
        assert "hydra.searchpath=[pkg://autoware_ml.configs]" in hydra_argv

    def test_preserves_explicit_hydra_searchpath_override(self, tmp_path: Path) -> None:
        config_file = tmp_path / "custom_train.yaml"
        config_file.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")

        hydra_argv = cli_runtime.resolve_hydra_argv(
            str(config_file),
            "tasks",
            extra_args=["hydra.searchpath=[file:///tmp/custom]"],
        )

        assert hydra_argv.count("hydra.searchpath=[file:///tmp/custom]") == 1
        assert "hydra.searchpath=[pkg://autoware_ml.configs]" not in hydra_argv

    def test_resolves_runtime_entrypoint_script_path(self) -> None:
        runtime_argv = cli_runtime.resolve_hydra_entrypoint_argv(
            "autoware_ml.scripts.train",
            SAMPLE_CONFIG_NAME,
            "tasks",
        )
        assert runtime_argv[0] == "autoware_ml.scripts.train"
        assert runtime_argv[1:] == cli_runtime.resolve_hydra_argv(SAMPLE_CONFIG_NAME, "tasks")


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

    def test_suggests_checkpoint_paths_after_weights_flag(self, tmp_path: Path) -> None:
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        (checkpoints_dir / "best.ckpt").write_text("ckpt")
        (checkpoints_dir / "notes.txt").write_text("txt")

        suggestions = complete_session_command_value(
            ["deploy", "--weights"], f"{checkpoints_dir}/b"
        )
        assert suggestions == [f"{checkpoints_dir}/best.ckpt"]


class TestShellCompletion:
    def test_train_suggests_options_on_empty_token(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [],
            env={
                "_AUTOWARE_ML_COMPLETE": "complete_bash",
                "COMP_WORDS": "autoware-ml train ",
                "COMP_CWORD": "2",
            },
        )
        assert result.exit_code == 0
        assert "--config-name" in result.stdout.splitlines()


class TestPathCompletion:
    def test_completes_checkpoint_paths(self, tmp_path: Path) -> None:
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        (checkpoints_dir / "best.ckpt").write_text("ckpt")
        (checkpoints_dir / "notes.txt").write_text("txt")

        suggestions = complete_path_value(f"{checkpoints_dir}/b", file_suffixes=(".ckpt",))
        assert suggestions == [f"{checkpoints_dir}/best.ckpt"]

    def test_completes_directories_only(self, tmp_path: Path) -> None:
        (tmp_path / "logs").mkdir()
        (tmp_path / "artifact.txt").write_text("txt")

        suggestions = complete_path_value(f"{tmp_path}/", directories_only=True)
        assert suggestions == [f"{tmp_path}/logs/"]


class TestSessionNameCompletion:
    def test_completes_managed_session_names(self) -> None:
        with patch(
            "autoware_ml.utils.cli.helpers.list_tmux_session_names",
            return_value=["default", SAMPLE_SESSION_NAME],
        ):
            assert complete_session_name_value("cal") == [SAMPLE_SESSION_NAME]

    def test_returns_empty_when_tmux_unavailable(self) -> None:
        with patch("autoware_ml.utils.cli.helpers.shutil.which", return_value=None):
            assert complete_session_name_value("fr") == []

    def test_filters_unmanaged_session_names(self) -> None:
        with patch("autoware_ml.utils.cli.helpers.shutil.which", return_value="/usr/bin/tmux"):
            with patch(
                "autoware_ml.utils.cli.helpers.subprocess.run",
                return_value=CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="default\t1\nscratch\t\ncalibration-status\t1\n",
                    stderr="",
                ),
            ):
                assert complete_session_name_value("cal") == ["calibration-status"]


class TestCliStartup:
    def test_importing_cli_does_not_import_mlflow_or_hydra(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import autoware_ml.cli.cli\nimport sys\n"
                "print('mlflow' in sys.modules)\nprint('hydra' in sys.modules)\n",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        assert result.stdout.splitlines() == ["False", "False"]
