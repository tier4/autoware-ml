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

"""Main CLI entry point for Autoware-ML commands.

This module defines the Typer application, command groups, and shell
completion helpers used by the ``autoware-ml`` executable.
"""

import logging
from importlib.metadata import version

import click
import typer
from click.core import ParameterSource
from click.shell_completion import CompletionItem
from typer.core import TyperCommand
from typing_extensions import Annotated

from autoware_ml.utils.cli.helpers import (
    complete_config_value,
    complete_path_value,
    complete_session_command_value,
    complete_session_name_value,
    parse_extra_args,
    run_lazy_script,
)

app = typer.Typer(
    name="autoware-ml",
    help="Autoware-ML - Machine learning framework for Autoware",
    no_args_is_help=True,
    add_completion=True,
)
mlflow_app = typer.Typer(
    name="mlflow",
    help="MLflow utilities",
    no_args_is_help=True,
)
session_app = typer.Typer(
    name="session",
    help="Managed background task sessions",
    no_args_is_help=True,
)

TASK_CONFIG_PREFIX = "tasks"
TRAIN_ENTRYPOINT_MODULE = "autoware_ml.scripts.train"
DEPLOY_ENTRYPOINT_MODULE = "autoware_ml.scripts.deploy"
TEST_ENTRYPOINT_MODULE = "autoware_ml.scripts.test"
CLI_RUNTIME_MODULE = "autoware_ml.cli.runtime"


class OptionFirstTyperCommand(TyperCommand):
    """Suggest command options even when completion starts on an empty token."""

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        results = super().shell_complete(ctx, incomplete)
        if incomplete:
            return results

        seen = {item.value for item in results}
        for param in self.get_params(ctx):
            if (
                not isinstance(param, click.Option)
                or param.hidden
                or (
                    not param.multiple
                    and ctx.get_parameter_source(param.name) is ParameterSource.COMMANDLINE
                )
            ):
                continue
            for option_name in [*param.opts, *param.secondary_opts]:
                if option_name in seen:
                    continue
                results.append(CompletionItem(option_name, help=param.help))
                seen.add(option_name)
        return results


def setup_logging(level: str = "INFO") -> None:
    """Configure process-wide logging for CLI execution.

    Args:
        level: Root logging level name.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


@app.callback(invoke_without_command=True)
def main_callback(
    version_flag: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit"),
    ] = False,
) -> None:
    """Handle top-level CLI options before subcommand execution.

    Args:
        version_flag: Whether to print the installed package version and exit.
    """
    setup_logging()
    if version_flag:
        typer.echo(f"autoware-ml {version('autoware-ml')}")
        raise typer.Exit()


def complete_task_config(incomplete: str) -> list[str]:
    """Complete task config names and config file paths.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for bundled task configs and YAML config paths.
    """
    return complete_config_value(incomplete, TASK_CONFIG_PREFIX)


def complete_checkpoint_path(incomplete: str) -> list[str]:
    """Complete checkpoint file paths.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates limited to checkpoint files.
    """
    return complete_path_value(incomplete, file_suffixes=(".ckpt",))


def complete_directory_path(incomplete: str) -> list[str]:
    """Complete directory paths.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates limited to directories.
    """
    return complete_path_value(incomplete, directories_only=True)


def complete_any_path(incomplete: str) -> list[str]:
    """Complete generic filesystem paths.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for files and directories.
    """
    return complete_path_value(incomplete)


def complete_session_command(ctx: click.Context, incomplete: str) -> list[str]:
    """Complete commands forwarded through ``session start``.

    Args:
        ctx: Typer shell-completion context with parsed parameters.
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for the forwarded command line.
    """
    command_args = list(ctx.params.get("command_args", ()))
    return complete_session_command_value(command_args, incomplete)


def complete_session_name(incomplete: str) -> list[str]:
    """Complete managed session names.

    Args:
        incomplete: Current completion prefix entered by the user.

    Returns:
        Completion candidates for managed session names.
    """
    return complete_session_name_value(incomplete)


@app.command(
    name="train",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(
    ctx: typer.Context,
    config_name: Annotated[
        str,
        typer.Option(
            "--config-name",
            help="Config name or YAML config path",
            autocompletion=complete_task_config,
        ),
    ],
) -> None:
    """Run model training through the Hydra-backed training entrypoint.

    Args:
        ctx: Typer context containing additional Hydra overrides.
        config_name: Config name or config file path to train.
    """
    run_lazy_script(
        CLI_RUNTIME_MODULE,
        "run_hydra_entrypoint",
        entrypoint_module=TRAIN_ENTRYPOINT_MODULE,
        config_name=config_name,
        stage="train",
        extra_args=ctx.args,
        config_prefix=TASK_CONFIG_PREFIX,
    )


@app.command(
    name="deploy",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def deploy(
    ctx: typer.Context,
    config_name: Annotated[
        str,
        typer.Option(
            "--config-name",
            help="Config name or YAML config path",
            autocompletion=complete_task_config,
        ),
    ],
    checkpoint: Annotated[
        str,
        typer.Option(
            "+checkpoint",
            help="Checkpoint path",
            autocompletion=complete_checkpoint_path,
        ),
    ],
) -> None:
    """Export a trained model through the deployment entrypoint.

    Args:
        ctx: Typer context containing additional Hydra overrides.
        config_name: Config name or config file path to deploy.
        checkpoint: Path to the checkpoint used for export.
    """
    hydra_overrides = [f"+checkpoint={checkpoint}"]
    run_lazy_script(
        CLI_RUNTIME_MODULE,
        "run_hydra_entrypoint",
        entrypoint_module=DEPLOY_ENTRYPOINT_MODULE,
        config_name=config_name,
        stage="deploy",
        extra_args=ctx.args,
        hydra_overrides=hydra_overrides,
        checkpoint=checkpoint,
        config_prefix=TASK_CONFIG_PREFIX,
    )


@app.command(
    name="test",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def test(
    ctx: typer.Context,
    config_name: Annotated[
        str,
        typer.Option(
            "--config-name",
            help="Config name or YAML config path",
            autocompletion=complete_task_config,
        ),
    ],
    checkpoint: Annotated[
        str,
        typer.Option(
            "+checkpoint",
            help="Checkpoint path",
            autocompletion=complete_checkpoint_path,
        ),
    ],
) -> None:
    """Run model evaluation through the Hydra-backed test entrypoint.

    Args:
        ctx: Typer context containing additional Hydra overrides.
        config_name: Config name or config file path to evaluate.
        checkpoint: Path to the checkpoint used for evaluation.
    """
    hydra_overrides = [f"+checkpoint={checkpoint}"]
    run_lazy_script(
        CLI_RUNTIME_MODULE,
        "run_hydra_entrypoint",
        entrypoint_module=TEST_ENTRYPOINT_MODULE,
        config_name=config_name,
        stage="test",
        extra_args=ctx.args,
        hydra_overrides=hydra_overrides,
        checkpoint=checkpoint,
        config_prefix=TASK_CONFIG_PREFIX,
    )


@mlflow_app.command(name="ui", cls=OptionFirstTyperCommand)
def mlflow_ui(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to listen on")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 5000,
    db_path: Annotated[
        str,
        typer.Option(
            "--db-path",
            help="Path to SQLite backend store file",
            autocompletion=complete_any_path,
        ),
    ] = "mlruns/mlflow.db",
) -> None:
    """Launch MLflow UI against a selected backend store.

    Args:
        host: Host interface used by the MLflow UI server.
        port: TCP port used by the MLflow UI server.
        db_path: Path to the SQLite backend store file.
    """
    run_lazy_script(
        "autoware_ml.scripts.mlflow_wrapper",
        "run_mlflow_ui",
        host=host,
        port=port,
        db_path=db_path,
    )


@mlflow_app.command(name="export", cls=OptionFirstTyperCommand)
def mlflow_export(
    db_path: Annotated[
        str,
        typer.Option(
            "--db-path",
            help="Path to SQLite backend store file",
            autocompletion=complete_any_path,
        ),
    ] = "mlruns/mlflow.db",
    config_name: Annotated[
        str | None,
        typer.Option(
            "--config-name",
            help="Export the experiment matching this task config path",
            autocompletion=complete_task_config,
        ),
    ] = None,
    experiment_name: Annotated[
        str | None, typer.Option("--experiment-name", help="Export only this MLflow experiment")
    ] = None,
    export_dir: Annotated[
        str | None,
        typer.Option(
            "--export-dir",
            help="Directory for the extracted experiment store",
            autocompletion=complete_directory_path,
        ),
    ] = None,
    override: Annotated[
        bool,
        typer.Option("--override", help="Allow replacing an existing exported MLflow store"),
    ] = False,
) -> None:
    """Export one MLflow experiment into an isolated tracking store.

    Args:
        config_name: User-facing config name used to derive the experiment name.
        experiment_name: Explicit MLflow experiment name to export.
        export_dir: Output directory for the exported tracking store.
        db_path: Path to the source SQLite backend store file.
        override: Whether to overwrite an existing export directory.
    """
    run_lazy_script(
        "autoware_ml.scripts.mlflow_wrapper",
        "export_experiment_from_db",
        db_path=db_path,
        experiment_name=experiment_name,
        config_name=config_name,
        export_dir=export_dir,
        override=override,
    )


@app.command(
    name="create-dataset",
    cls=OptionFirstTyperCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def create_dataset(
    ctx: typer.Context,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset name (e.g., nuscenes, nuscenes_mini)"),
    ],
    task: Annotated[list[str], typer.Option("--task", help="Task name (can be repeated)")],
    root_path: Annotated[
        str,
        typer.Option(
            "--root-path",
            help="Root path of the dataset",
            autocompletion=complete_directory_path,
        ),
    ],
    out_dir: Annotated[
        str,
        typer.Option(
            "--out-dir",
            help="Output directory for info files",
            autocompletion=complete_directory_path,
        ),
    ],
) -> None:
    """Generate dataset info files with specified tasks.

    Requires dataset name and at least one task.
    """

    run_lazy_script(
        "autoware_ml.scripts.create_dataset",
        "main",
        dataset=dataset,
        tasks=task,
        root_path=root_path,
        out_dir=out_dir,
        **parse_extra_args(ctx.args),
    )


@session_app.command(
    name="start",
    cls=OptionFirstTyperCommand,
)
def session_start(
    name: Annotated[str, typer.Option("--name", "-n", help="Session name")],
    cwd: Annotated[
        str | None,
        typer.Option(
            "--cwd",
            help="Working directory for the session command",
            autocompletion=complete_directory_path,
        ),
    ] = None,
    attach: Annotated[
        bool, typer.Option("--attach", help="Open the live viewer immediately after starting")
    ] = False,
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Run the forwarded command as-is instead of prefixing it with autoware-ml",
        ),
    ] = False,
    command_args: Annotated[
        list[str] | None,
        typer.Argument(
            help="Command to run in the managed background session. Pass it after '--', e.g. -- train --config-name ...",
            autocompletion=complete_session_command,
        ),
    ] = None,
) -> None:
    """Start a detached managed session for a background task.

    Args:
        name: Managed session name.
        cwd: Working directory used when launching the session command.
        attach: Whether to open the live viewer immediately after startup.
        raw: Whether to execute the forwarded command directly instead of
            prefixing it with ``autoware-ml``.
        command_args: Command tokens forwarded to the managed shell.
    """
    run_lazy_script(
        "autoware_ml.scripts.session",
        "start_session",
        name=name,
        command_args=command_args or [],
        cwd=cwd,
        attach=attach,
        raw=raw,
    )
    if not attach:
        typer.echo(f"Started session '{name}'.")
        typer.echo(f"View live output with: autoware-ml session attach --name {name}")
        typer.echo("Press Ctrl+C in the viewer to return without stopping the task.")
        typer.echo(f"Stop the task with: autoware-ml session stop --name {name}")


@session_app.command(name="attach", cls=OptionFirstTyperCommand)
def session_attach(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Session name", autocompletion=complete_session_name),
    ],
) -> None:
    """Render a live terminal view of a managed session.

    Args:
        name: Name of the session to view.
    """
    run_lazy_script("autoware_ml.scripts.session", "attach_session", name=name)


@session_app.command(name="detach", cls=OptionFirstTyperCommand)
def session_detach(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Session name", autocompletion=complete_session_name),
    ],
) -> None:
    """Disconnect raw tmux clients from a managed session.

    Args:
        name: Name of the session whose tmux clients should be detached.
    """
    run_lazy_script("autoware_ml.scripts.session", "detach_session", name=name)


@session_app.command(name="ls", cls=OptionFirstTyperCommand)
def session_ls() -> None:
    """List background sessions managed by ``autoware-ml``.

    The command prints formatted session information and exits quietly when no
    managed sessions are currently running.
    """
    output = run_lazy_script("autoware_ml.scripts.session", "list_sessions")
    if output:
        typer.echo(output)


@session_app.command(name="stop", cls=OptionFirstTyperCommand)
def session_stop(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Session name", autocompletion=complete_session_name),
    ],
) -> None:
    """Stop a managed background task and close its session.

    Args:
        name: Name of the session to stop.
    """
    run_lazy_script("autoware_ml.scripts.session", "stop_session", name=name)


def main() -> None:
    """Run the top-level Typer application.

    This wrapper keeps the installed entrypoint and ``python -m`` execution
    path aligned on the same CLI startup logic.
    """
    app()


app.add_typer(mlflow_app, name="mlflow")
app.add_typer(session_app, name="session")


if __name__ == "__main__":
    main()
