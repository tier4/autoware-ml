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

"""Main CLI entry point for Autoware-ML commands."""

import sys
from importlib.metadata import version

import typer
from typing_extensions import Annotated

from autoware_ml.utils.cli import adjust_argv, expand_config_path, parse_extra_args, run_lazy_script

app = typer.Typer(
    name="autoware-ml",
    help="Autoware-ML - Machine learning framework for Autoware",
    no_args_is_help=True,
    add_completion=True,
)


@app.callback(invoke_without_command=True)
def main_callback(
    version_flag: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit"),
    ] = False,
) -> None:
    """Autoware-ML - Machine learning framework for Autoware."""
    if version_flag:
        try:
            package_version = version("autoware-ml")
            typer.echo(f"autoware-ml {package_version}")
        except Exception:
            typer.echo("autoware-ml (version not available)")
        raise typer.Exit()


@app.command(
    name="train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(
    config_name: Annotated[str, typer.Option("--config-name", help="Config name")],
) -> None:
    """Train models using PyTorch Lightning.

    Requires config name, all other arguments are passed to Hydra.
    """
    assert config_name is not None, "Config name is required"
    expanded_config_name = expand_config_path(config_name, "tasks")
    sys.argv = [sys.argv[0]] + adjust_argv(sys.argv[2:])
    sys.argv[sys.argv.index("--config-name") + 1] = expanded_config_name

    run_lazy_script("autoware_ml.scripts.train", "main")


@app.command(
    name="deploy",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def deploy(
    config_name: Annotated[str, typer.Option("--config-name", help="Config name")],
    checkpoint: Annotated[str, typer.Option("+checkpoint", help="Checkpoint path")],
) -> None:
    """Export models to ONNX and TensorRT.

    Requires config name and checkpoint path, all other arguments are passed to Hydra.
    """
    assert config_name is not None, "Config name is required"
    assert checkpoint is not None, "Checkpoint path is required"
    expanded_config_name = expand_config_path(config_name, "tasks")
    sys.argv = [sys.argv[0]] + adjust_argv(sys.argv[2:])
    sys.argv[sys.argv.index("--config-name") + 1] = expanded_config_name

    run_lazy_script("autoware_ml.scripts.deploy", "main")


@app.command(name="mlflow-ui")
def mlflow_ui(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 5000,
    db_path: Annotated[
        str | None, typer.Option("--db-path", help="Path to SQLite backend store file")
    ] = None,
) -> None:
    """Launch MLflow UI."""
    run_lazy_script("autoware_ml.scripts.mlflow_ui", "run_mlflow_ui", port=port, db_path=db_path)


@app.command(
    name="create-dataset",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def create_dataset(
    ctx: typer.Context,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset name (e.g., nuscenes, nuscenes_mini)"),
    ],
    task: Annotated[list[str], typer.Option("--task", help="Task name (can be repeated)")],
    root_path: Annotated[str, typer.Option("--root-path", help="Root path of the dataset")],
    out_dir: Annotated[str, typer.Option("--out-dir", help="Output directory for info files")],
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


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
