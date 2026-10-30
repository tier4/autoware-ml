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

import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer.main
from click.shell_completion import get_completion_class
from omegaconf import OmegaConf
from typer.testing import CliRunner

import __main__
import autoware_ml.cli.cli as cli
import autoware_ml.cli.runtime as cli_runtime
import autoware_ml.utils.cli.helpers as helpers
from autoware_ml.cli.cli import app
from autoware_ml.configs.paths import CONFIGS_ROOT
from autoware_ml.configs.tests.settings import write_settings
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
from autoware_ml.utils.mlflow_store import compose_config_reference
from autoware_ml.utils.session import SessionCommandError

SAMPLE_CONFIG_NAME = (
    "detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base"
)
SAMPLE_CONFIG_PATH = f"tasks/{SAMPLE_CONFIG_NAME}"
SAMPLE_SESSION_NAME = "centerpoint-train"
SAMPLE_EXPERIMENT_GROUP_NAME = "ptv3_det3d_seg3d/T4Dataset-j6gen2_base-v1.0.0/j6gen2_base"
SAMPLE_EXPERIMENT_NAME = f"{SAMPLE_EXPERIMENT_GROUP_NAME}/voxel012"
SAMPLE_EXPERIMENT_UID = "2026.10.15.12.00.00"


def _write_external_root(tmp_path: Path, monkeypatch, config_name: str, content: str) -> Path:
    """Write one config into an external config root and register the root."""
    root = tmp_path / "configs"
    config_file = root / f"{config_name}.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(content, encoding="utf-8")
    write_settings(tmp_path, monkeypatch, AUTOWARE_ML_CONFIGS_PATH=str(root))
    return root


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
        assert config_path == str(CONFIGS_ROOT)
        assert config_name == SAMPLE_CONFIG_PATH
        assert hydra_overrides == ["hydra.searchpath=[]"]

    def test_resolve_packaged_yaml_path(self) -> None:
        config_path, config_name, hydra_overrides = resolve_config_reference(
            "autoware_ml/configs/tasks/detection3d/centerpoint/t4dataset/"
            "voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base.yaml",
            "tasks",
        )

        assert config_path == str(CONFIGS_ROOT)
        assert config_name == SAMPLE_CONFIG_PATH
        assert hydra_overrides == ["hydra.searchpath=[]"]

    def test_resolve_external_config_name(self, tmp_path: Path, monkeypatch) -> None:
        root = _write_external_root(tmp_path, monkeypatch, "tasks/custom/train", "")

        config_path, config_name, hydra_overrides = resolve_config_reference(
            "custom/train", "tasks"
        )

        assert config_path == str(root)
        assert config_name == "tasks/custom/train"
        assert hydra_overrides == [f"hydra.searchpath=[file://{CONFIGS_ROOT}]"]

    def test_resolve_external_yaml_path(self, tmp_path: Path, monkeypatch) -> None:
        root = _write_external_root(tmp_path, monkeypatch, "tasks/custom/train", "")
        monkeypatch.chdir(root)

        config_path, config_name, hydra_overrides = resolve_config_reference(
            "./tasks/custom/train.yaml", "tasks"
        )

        assert config_path == str(root)
        assert config_name == "tasks/custom/train"
        assert hydra_overrides == [f"hydra.searchpath=[file://{CONFIGS_ROOT}]"]

    def test_bundled_config_searches_external_roots(self, tmp_path: Path, monkeypatch) -> None:
        root = _write_external_root(tmp_path, monkeypatch, "tasks/custom/train", "")

        _, _, hydra_overrides = resolve_config_reference(SAMPLE_CONFIG_NAME, "tasks")

        assert hydra_overrides == [f"hydra.searchpath=[file://{root}]"]

    def test_reject_yaml_outside_config_roots(self, tmp_path: Path) -> None:
        config_file = tmp_path / "custom_train.yaml"
        config_file.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="inside no config root"):
            resolve_config_reference(str(config_file), "tasks")

    def test_reject_missing_config(self) -> None:
        with pytest.raises(FileNotFoundError, match="tasks/custom/missing.yaml"):
            resolve_config_reference("custom/missing", "tasks")

    def test_reject_a_missing_external_config_root(self, tmp_path: Path, monkeypatch) -> None:
        write_settings(tmp_path, monkeypatch, AUTOWARE_ML_CONFIGS_PATH=str(tmp_path / "missing"))

        with pytest.raises(NotADirectoryError, match="is no directory"):
            resolve_config_reference(SAMPLE_CONFIG_NAME, "tasks")

    def test_reject_external_config_shadowing_a_bundled_one(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_external_root(tmp_path, monkeypatch, SAMPLE_CONFIG_PATH, "")

        with pytest.raises(ValueError, match="more than one config root"):
            resolve_config_reference(SAMPLE_CONFIG_NAME, "tasks")

    def test_compose_external_config_on_bundled_defaults(self, tmp_path: Path, monkeypatch) -> None:
        bundled_cfg = compose_config_reference(SAMPLE_CONFIG_NAME, "tasks")
        _write_external_root(
            tmp_path,
            monkeypatch,
            "tasks/detection3d/centerpoint/t4dataset/custom_t4dataset",
            "# @package _global_\n"
            "defaults:\n"
            f"  - {Path(SAMPLE_CONFIG_NAME).name}\n"
            "  - _self_\n"
            "\n"
            "trainer:\n"
            "  max_epochs: 3\n",
        )

        cfg = compose_config_reference(
            "detection3d/centerpoint/t4dataset/custom_t4dataset", "tasks"
        )

        assert cfg.trainer.max_epochs == 3
        assert cfg.model == bundled_cfg.model


class TestCompleteConfigValue:
    """Tests for config completion helpers."""

    def test_complete_bundled_configs(self, tmp_path: Path, monkeypatch) -> None:
        config_root = tmp_path / "tasks" / "detection3d" / "centerpoint" / "t4dataset"
        config_root.mkdir(parents=True)
        (config_root / "base.yaml").write_text("", encoding="utf-8")
        (config_root / f"{Path(SAMPLE_CONFIG_NAME).name}.yaml").write_text("", encoding="utf-8")
        monkeypatch.setattr(helpers, "CONFIGS_ROOT", tmp_path)

        completions = complete_config_value("detection3d/centerpoint/t4dataset/voxel024", "tasks")

        assert completions == [SAMPLE_CONFIG_NAME]
        assert "detection3d/centerpoint/t4dataset/base" not in completions

    def test_complete_external_root_configs(self, tmp_path: Path, monkeypatch) -> None:
        _write_external_root(tmp_path, monkeypatch, "tasks/custom/train", "")

        completions = complete_config_value("custom/tr", "tasks")

        assert completions == ["custom/train"]

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
                ["train", "--config-name", SAMPLE_CONFIG_NAME, "--weights", "seg3d.ckpt"],
            )

        assert result.exit_code == 0
        run_lazy_script_mock.assert_called_once_with(
            cli.CLI_RUNTIME_MODULE,
            "run_hydra_entrypoint",
            entrypoint_module=cli.TRAIN_ENTRYPOINT_MODULE,
            config_name=SAMPLE_CONFIG_NAME,
            stage="train",
            extra_args=[],
            hydra_overrides=["+weights=[seg3d.ckpt]"],
            resume_checkpoint=None,
            new_run=False,
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_train_forwards_skip_mismatched_weights(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "train",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "--weights",
                    "seg3d.ckpt",
                    "--skip-mismatched-weights",
                ],
            )

        assert result.exit_code == 0
        hydra_overrides = run_lazy_script_mock.call_args.kwargs["hydra_overrides"]
        assert hydra_overrides == ["+weights=[seg3d.ckpt]", "+skip_mismatched_weights=true"]

    def test_train_rejects_skip_mismatched_weights_without_weights(self) -> None:
        result = self.runner.invoke(
            app,
            ["train", "--config-name", SAMPLE_CONFIG_NAME, "--skip-mismatched-weights"],
        )

        assert result.exit_code != 0
        assert "--skip-mismatched-weights requires --weights" in result.output

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
                "seg3d.ckpt",
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

    def test_deploy_forwards_release_override(self) -> None:
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "deploy",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "--weights",
                    "model.ckpt",
                    "--release",
                    "v1.2.3",
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
            hydra_overrides=["+weights=[model.ckpt]", "+release=v1.2.3"],
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
                    "seg3d.ckpt",
                    "--weights",
                    "det3d.ckpt",
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
            hydra_overrides=["+weights=[seg3d.ckpt,det3d.ckpt]"],
            checkpoints=["seg3d.ckpt", "det3d.ckpt"],
            config_prefix=cli.TASK_CONFIG_PREFIX,
        )

    def test_visualize_previews_data_without_weights(self) -> None:
        """
        Input: ``visualize`` with a config name and the noop backend but no weights.
        Expected: the CLI runs the visualize entrypoint without a runtime stage and passes the
            backend override, and no weights override.
        Check: the entrypoint module, the stage and the Hydra overrides of the dispatched call.
        """
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                ["visualize", "--config-name", SAMPLE_CONFIG_NAME, "--backend", "noop"],
            )

        assert result.exit_code == 0
        kwargs = run_lazy_script_mock.call_args.kwargs
        assert kwargs["entrypoint_module"] == cli.VISUALIZE_ENTRYPOINT_MODULE
        assert kwargs["stage"] is None
        assert "+visualization.backend=noop" in kwargs["hydra_overrides"]
        assert not any(override.startswith("+weights=") for override in kwargs["hydra_overrides"])

    def test_visualize_merges_repeated_weights(self) -> None:
        """
        Input: ``visualize`` with two ``--weights`` options and ``--split val``.
        Expected: the two checkpoints are merged into one weights list override, like for
            ``test``, and the split reaches the visualization group.
        Check: the weights and split Hydra overrides of the dispatched call.
        """
        with patch("autoware_ml.cli.cli.run_lazy_script") as run_lazy_script_mock:
            result = self.runner.invoke(
                app,
                [
                    "visualize",
                    "--config-name",
                    SAMPLE_CONFIG_NAME,
                    "--weights",
                    "seg3d.ckpt",
                    "--weights",
                    "det3d.ckpt",
                    "--split",
                    "val",
                ],
            )

        assert result.exit_code == 0
        hydra_overrides = run_lazy_script_mock.call_args.kwargs["hydra_overrides"]
        assert "+weights=[seg3d.ckpt,det3d.ckpt]" in hydra_overrides
        assert "+visualization.split=val" in hydra_overrides

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
                    "seg3d.ckpt",
                    "--weights",
                    "det3d.ckpt",
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
            hydra_overrides=["+weights=[seg3d.ckpt,det3d.ckpt]", "++trainer.devices=1"],
            checkpoint="det3d.ckpt",
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
                    "seg3d.ckpt",
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
            hydra_overrides=["+weights=[seg3d.ckpt]", "++trainer.devices=1"],
            checkpoint="seg3d.ckpt",
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
                    "seg3d.ckpt",
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
            hydra_overrides=["+weights=[seg3d.ckpt]"],
            checkpoint="seg3d.ckpt",
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

    def test_session_start_reports_clean_error(self, capsys: pytest.CaptureFixture) -> None:
        argv = ["autoware-ml", "session", "start", "--name", SAMPLE_SESSION_NAME]
        with patch(
            "autoware_ml.cli.cli.run_lazy_script",
            side_effect=SessionCommandError("A command is required"),
        ):
            with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exit_info:
                cli.main()

        assert exit_info.value.code == 1
        assert "A command is required" in capsys.readouterr().err


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

        with (
            self._patched_logger_config(config_name),
            patch(
                "autoware_ml.cli.runtime.resolve_deploy_lineage",
                return_value=(None, [{"run_id": "seg-run"}, {"run_id": "det-run"}]),
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
                checkpoints=["seg3d.ckpt", "det3d.ckpt"],
            )

        resolve_deploy_lineage_mock.assert_called_once_with(
            [Path("seg3d.ckpt"), Path("det3d.ckpt")]
        )
        prepare_run_context_mock.assert_called_once()
        assert prepare_run_context_mock.call_args.kwargs["experiment_name"] == (
            SAMPLE_EXPERIMENT_NAME
        )
        assert prepare_run_context_mock.call_args.kwargs["run_name"] == (
            f"deploy_voxel012_{SAMPLE_EXPERIMENT_UID}"
        )
        assert prepare_run_context_mock.call_args.kwargs["parent_run_id"] is None
        assert prepare_run_context_mock.call_args.kwargs["extra_tags"]["source_run_ids"] == (
            "seg-run,det-run"
        )
        assert env_updates["AUTOWARE_ML_RUN_ID"] == "deploy-run"

    @contextmanager
    def _patched_logger_config(self, config_name: str, logger_enabled: bool = True):
        """Compose to the experiment bookkeeping of ``default_task.yaml`` plus the logger."""
        cfg = OmegaConf.create(
            {
                "experiment_group_name": SAMPLE_EXPERIMENT_GROUP_NAME,
                "experiment_name": "voxel012",
                "experiment_uid": SAMPLE_EXPERIMENT_UID,
                "logger": (
                    {"tracking_uri": "sqlite:///mlruns/mlflow.db"} if logger_enabled else None
                ),
            }
        )
        with (
            patch(
                "autoware_ml.cli.runtime.resolve_config_reference",
                return_value=(str(CONFIGS_ROOT), f"tasks/{config_name}", ["hydra.searchpath=[]"]),
            ),
            patch("autoware_ml.cli.runtime.initialize_config_dir", return_value=nullcontext()),
            patch("autoware_ml.cli.runtime.compose", return_value=cfg),
        ):
            yield

    def test_prepare_runtime_environment_names_the_run_like_the_entrypoint_builder(
        self, tmp_path: Path
    ) -> None:
        """
        The CLI creates the run before the entrypoint starts. It must use the experiment and
        run names the entrypoint's builder derives from the same config, otherwise ``aml train``
        and ``python -m autoware_ml.scripts.train`` would name their runs differently.
        """
        config_name = "multi/ptv3/voxel012"
        with (
            self._patched_logger_config(config_name),
            patch(
                "autoware_ml.cli.runtime.prepare_run_context",
                return_value=SimpleNamespace(run_id="train-run", hydra_dir=tmp_path / "hydra"),
            ) as prepare_run_context_mock,
        ):
            env_updates = cli_runtime.prepare_runtime_environment(config_name, "tasks", "train")

        kwargs = prepare_run_context_mock.call_args.kwargs
        assert prepare_run_context_mock.call_args.args[1] == config_name
        assert kwargs["experiment_name"] == SAMPLE_EXPERIMENT_NAME
        assert kwargs["run_name"] == f"train_voxel012_{SAMPLE_EXPERIMENT_UID}"
        assert kwargs["parent_run_id"] is None
        assert env_updates["AUTOWARE_ML_RUN_ID"] == "train-run"

    def test_prepare_runtime_environment_without_logger_keys_hydra_dir_on_the_experiment(
        self,
    ) -> None:
        config_name = "multi/ptv3/voxel012"
        with self._patched_logger_config(config_name, logger_enabled=False):
            env_updates = cli_runtime.prepare_runtime_environment(config_name, "tasks", "train")

        assert env_updates["AUTOWARE_ML_RUN_ID"] is None
        hydra_dir = Path(env_updates["AUTOWARE_ML_HYDRA_RUN_DIR"])
        assert hydra_dir.parent.parent.parent == (
            Path(cli_runtime.generate_hydra_run_dir(SAMPLE_EXPERIMENT_NAME)).parent.parent.parent
        )
        assert SAMPLE_EXPERIMENT_NAME in hydra_dir.as_posix()
        assert "_hydra" in hydra_dir.parts

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
            patch("autoware_ml.cli.runtime.mark_run_running") as mark_run_running_mock,
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
        mark_run_running_mock.assert_called_once_with("sqlite:///mlruns/mlflow.db", "source-run")
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

    def test_prepare_runtime_environment_resume_rejects_incomplete_metadata(
        self, tmp_path: Path
    ) -> None:
        config_name = "multi/ptv3/voxel012"
        with (
            self._patched_logger_config(config_name),
            patch(
                "autoware_ml.cli.runtime.load_run_metadata",
                return_value={"experiment_name": "multi_ptv3_voxel012"},
            ),
            pytest.raises(ValueError, match="missing.*run_id.*config_name"),
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

    def test_adds_config_path_of_external_root(self, tmp_path: Path, monkeypatch) -> None:
        root = _write_external_root(tmp_path, monkeypatch, "tasks/custom/train", "")

        hydra_argv = cli_runtime.resolve_hydra_argv("custom/train", "tasks")

        assert hydra_argv == [
            "--config-name",
            "tasks/custom/train",
            "--config-path",
            str(root),
            f"hydra.searchpath=[file://{CONFIGS_ROOT}]",
        ]

    def test_preserves_explicit_hydra_searchpath_override(self) -> None:
        hydra_argv = cli_runtime.resolve_hydra_argv(
            SAMPLE_CONFIG_NAME,
            "tasks",
            extra_args=["hydra.searchpath=[file:///tmp/custom]"],
        )

        assert hydra_argv.count("hydra.searchpath=[file:///tmp/custom]") == 1
        assert "hydra.searchpath=[]" not in hydra_argv

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
            "detection3d/centerpoint/t4dataset/voxel024",
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
    @staticmethod
    def _completions(words: str, cword: str) -> list[str]:
        """Completion values the shell would receive for a partially typed command."""
        command = typer.main.get_command(app)
        with patch.dict(os.environ, {"COMP_WORDS": words, "COMP_CWORD": cword}):
            completion = get_completion_class("bash")(
                command, {}, "autoware-ml", "_AUTOWARE_ML_COMPLETE"
            )
            args, incomplete = completion.get_completion_args()
            return [item.value for item in completion.get_completions(args, incomplete)]

    def test_train_suggests_its_options_on_a_dash(self) -> None:
        assert "--config-name" in self._completions("autoware-ml train -", "2")

    def test_train_narrows_the_options_on_a_prefix(self) -> None:
        assert self._completions("autoware-ml train --con", "2") == ["--config-name"]

    def test_the_root_suggests_its_commands(self) -> None:
        assert "train" in self._completions("autoware-ml ", "1")


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
            assert complete_session_name_value("cen") == [SAMPLE_SESSION_NAME]

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
