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

"""MLflow helper functions used by the CLI."""

import logging
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from autoware_ml.utils.cli.helpers import infer_user_config_name
from autoware_ml.utils.mlflow_helpers import MLFLOW_PARENT_RUN_ID, artifact_uri_to_path

logger = logging.getLogger(__name__)


def normalize_experiment_name(experiment_name: str | None, config_name: str | None) -> str | None:
    """Resolve the target experiment name.

    Args:
        experiment_name: Explicit MLflow experiment name.
        config_name: User-facing config name used as shorthand.

    Returns:
        Resolved experiment name, or ``None`` when neither input is provided.
    """
    if experiment_name is not None and config_name is not None:
        raise ValueError("Use either --experiment-name or --config-name, not both.")
    if config_name is not None:
        return infer_user_config_name(config_name, "tasks").replace("/", "_")
    return experiment_name


def build_backend_uri(db_path: Path, *, create_parent: bool = False) -> str:
    """Convert a SQLite database path into an MLflow backend URI.

    Args:
        db_path: Path to the SQLite database file.
        create_parent: Whether to create the parent directory when missing.

    Returns:
        MLflow backend URI string.
    """
    resolved = db_path if db_path.is_absolute() else (Path.cwd() / db_path)
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved}"


def resolve_db_path(db_path: str) -> Path:
    """Resolve the global MLflow DB path.

    Args:
        db_path: User-provided database path.

    Returns:
        Absolute path to the MLflow SQLite database.
    """
    resolved = Path(db_path)
    return resolved if resolved.is_absolute() else (Path.cwd() / resolved)


def resolve_export_dir(export_dir: str | None) -> Path | None:
    """Resolve the optional export directory path.

    Args:
        export_dir: User-provided export directory path.

    Returns:
        Absolute export directory path, or ``None`` when not provided.
    """
    if export_dir is None:
        return None
    resolved = Path(export_dir)
    return resolved if resolved.is_absolute() else (Path.cwd() / resolved)


def prepare_export_output_dir(output_dir: Path, override: bool) -> Path:
    """Prepare an output directory for experiment export.

    Args:
        output_dir: Destination directory for the isolated MLflow store.
        override: Whether existing export outputs may be overwritten.

    Returns:
        Target DB path.

    Raises:
        FileExistsError: If the target DB already exists and ``override`` is ``False``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_db_path = output_dir / "mlflow.db"
    if target_db_path.exists() and not override:
        raise FileExistsError(
            f"Export target already exists at '{target_db_path}'. "
            "Use --override to replace the exported MLflow store."
        )

    if override:
        artifacts_dir = output_dir / "artifacts"
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
        for sqlite_path in output_dir.glob("mlflow.db*"):
            if sqlite_path.is_file():
                sqlite_path.unlink()
    return target_db_path


def is_port_free(host: str, port: int) -> bool:
    """Return whether a TCP port is available.

    Args:
        host: Host interface to test.
        port: TCP port number.

    Returns:
        ``True`` when the port is available.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def resolve_port(host: str, port: int) -> int:
    """Validate that the requested port is available.

    Args:
        host: Host interface to bind.
        port: Requested TCP port.

    Returns:
        Requested port when it is free.
    """
    if not is_port_free(host, port):
        raise RuntimeError(f"Port {port} is occupied. Try a different port with --port option.")
    return port


def create_export_run_skeleton(
    source_client: MlflowClient,
    target_client: MlflowClient,
    source_run_id: str,
    target_experiment_id: str,
    run_id_map: dict[str, str],
) -> str:
    """Create an empty exported run and keep a mapping from source IDs to target IDs.

    Args:
        source_client: MLflow client for the source tracking store.
        target_client: MLflow client for the target tracking store.
        source_run_id: Source MLflow run ID.
        target_experiment_id: Target experiment ID in the exported store.
        run_id_map: Mapping from source run IDs to exported run IDs.

    Returns:
        Target run ID created in the exported store.
    """
    source_run = source_client.get_run(source_run_id)
    source_tags = {
        key: value
        for key, value in source_run.data.tags.items()
        if key not in {"mlflow.runId", MLFLOW_PARENT_RUN_ID}
    }
    source_parent_run_id = source_run.data.tags.get(MLFLOW_PARENT_RUN_ID)
    if source_parent_run_id is not None and source_parent_run_id in run_id_map:
        source_tags[MLFLOW_PARENT_RUN_ID] = run_id_map[source_parent_run_id]
    run_name = source_run.data.tags.get("mlflow.runName")
    created_run = target_client.create_run(
        experiment_id=target_experiment_id,
        start_time=source_run.info.start_time,
        tags=source_tags,
        run_name=run_name,
    )
    run_id_map[source_run_id] = created_run.info.run_id
    return created_run.info.run_id


def populate_exported_run(
    source_client: MlflowClient,
    target_client: MlflowClient,
    source_run_id: str,
    target_run_id: str,
) -> None:
    """Populate a pre-created exported run.

    Args:
        source_client: MLflow client for the source tracking store.
        target_client: MLflow client for the target tracking store.
        source_run_id: Source MLflow run ID.
        target_run_id: Target MLflow run ID.
    """
    source_run = source_client.get_run(source_run_id)

    if source_run.data.params:
        target_client.log_batch(
            target_run_id,
            params=[
                mlflow.entities.Param(key=key, value=value)
                for key, value in source_run.data.params.items()
            ],
        )

    for metric_key in source_run.data.metrics:
        metric_history = source_client.get_metric_history(source_run_id, metric_key)
        if not metric_history:
            continue
        target_client.log_batch(target_run_id, metrics=metric_history)

    target_client.set_terminated(
        target_run_id,
        status=source_run.info.status,
        end_time=source_run.info.end_time,
    )

    source_artifact_dir = artifact_uri_to_path(source_run.info.artifact_uri)
    target_artifact_dir = artifact_uri_to_path(
        target_client.get_run(target_run_id).info.artifact_uri
    )
    if source_artifact_dir.exists():
        target_artifact_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_artifact_dir, target_artifact_dir, dirs_exist_ok=True)


def export_experiment_to_store(
    source_tracking_uri: str,
    experiment_name: str,
    export_dir: Path | None,
    override: bool = False,
) -> Path:
    """Export one experiment from the global store into an isolated SQLite store.

    Args:
        source_tracking_uri: Tracking URI of the source MLflow store.
        experiment_name: Experiment name to export.
        export_dir: Optional output directory for the exported store.
        override: Whether to overwrite an existing export in ``export_dir``.

    Returns:
        Path to the exported SQLite database.
    """
    source_client = MlflowClient(tracking_uri=source_tracking_uri)
    source_experiment = source_client.get_experiment_by_name(experiment_name)
    if source_experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' was not found in the source tracking store."
        )

    output_dir = export_dir or Path(tempfile.mkdtemp(prefix="autoware-ml-mlflow-export-"))
    target_db_path = prepare_export_output_dir(output_dir, override=override)
    target_tracking_uri = build_backend_uri(target_db_path)

    logger.info(f"Exporting experiment '{experiment_name}' to {output_dir}")

    target_client = MlflowClient(tracking_uri=target_tracking_uri)
    target_experiment_id = target_client.create_experiment(
        name=source_experiment.name,
        artifact_location=(output_dir / "artifacts" / source_experiment.name).resolve().as_uri(),
    )

    runs = source_client.search_runs(
        experiment_ids=[source_experiment.experiment_id],
        run_view_type=ViewType.ALL,
        max_results=50000,
    )
    run_id_map: dict[str, str] = {}
    sorted_runs = sorted(
        runs,
        key=lambda run: (run.data.tags.get(MLFLOW_PARENT_RUN_ID) is not None, run.info.start_time),
    )
    for run in sorted_runs:
        create_export_run_skeleton(
            source_client,
            target_client,
            run.info.run_id,
            target_experiment_id,
            run_id_map,
        )
    for run in sorted_runs:
        populate_exported_run(
            source_client,
            target_client,
            run.info.run_id,
            run_id_map[run.info.run_id],
        )

    logger.info(f"Exported database: {target_db_path}")
    return target_db_path


def launch_mlflow_ui(host: str, port: int, backend_uri: str) -> None:
    """Run MLflow UI against the given backend store.

    Args:
        host: Host interface used by the MLflow UI server.
        port: TCP port used by the MLflow UI server.
        backend_uri: MLflow backend store URI.
    """
    resolved_port = resolve_port(host, port)
    logger.info(f"Backend store: {backend_uri}")
    logger.info(f"Starting MLflow UI on port {resolved_port} ...")
    subprocess.run(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            backend_uri,
            "--host",
            host,
            "--port",
            str(resolved_port),
        ],
        check=True,
    )


def export_experiment_from_db(
    db_path: str,
    experiment_name: str | None = None,
    config_name: str | None = None,
    export_dir: str | None = None,
    override: bool = False,
) -> Path:
    """Export one experiment from the global MLflow DB into an isolated store.

    Args:
        db_path: Path to the source SQLite backend store.
        experiment_name: Explicit MLflow experiment name.
        config_name: User-facing config name used as shorthand.
        export_dir: Optional output directory for the exported store.
        override: Whether to overwrite an existing export in ``export_dir``.

    Returns:
        Path to the exported SQLite database.

    Raises:
        FileNotFoundError: If the source MLflow database does not exist.
    """
    target_experiment_name = normalize_experiment_name(experiment_name, config_name)
    if target_experiment_name is None:
        raise ValueError("Either --experiment-name or --config-name must be provided.")

    source_db_path = resolve_db_path(db_path)
    if not source_db_path.exists():
        raise FileNotFoundError(f"Source MLflow database does not exist: {source_db_path}")

    source_tracking_uri = build_backend_uri(source_db_path, create_parent=False)
    resolved_export_dir = resolve_export_dir(export_dir)
    return export_experiment_to_store(
        source_tracking_uri,
        target_experiment_name,
        resolved_export_dir,
        override=override,
    )


def run_mlflow_ui(
    host: str,
    port: int,
    db_path: str,
) -> None:
    """Run MLflow UI for the global tracking store.

    Args:
        host: Host interface used by the MLflow UI server.
        port: TCP port used by the MLflow UI server.
        db_path: Path to the SQLite backend store.
    """
    resolved_db_path = resolve_db_path(db_path)
    if not resolved_db_path.exists():
        logger.info(
            "MLflow database does not exist at '%s'. A new SQLite store will be created.",
            resolved_db_path,
        )
    backend_uri = build_backend_uri(resolved_db_path, create_parent=True)
    launch_mlflow_ui(host, port, backend_uri)
