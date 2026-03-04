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

"""Launch MLflow UI with sensible defaults and port handling."""

import argparse
import logging
import socket
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start MLflow UI.")
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to the SQLite backend store file.",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host for MLflow UI.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for MLflow UI.",
    )
    return parser.parse_args()


def build_backend_uri(db_path: str) -> str:
    db_path = Path(db_path)
    resolved = db_path if db_path.is_absolute() else (Path.cwd() / db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved}"


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def resolve_port(host: str, port: int) -> int:
    if not is_port_free(host, port):
        logger.error(f"Port {port} is occupied. Try a different port with --port option.")
        sys.exit(1)
    return port


def run_mlflow_ui(host: str, port: int, db_path: str) -> None:
    """Run MLflow UI with the specified port and database path.

    Args:
        host: Host for MLflow UI.
        port: Port for MLflow UI.
        db_path: Path to SQLite backend store file.
    """
    backend_uri = build_backend_uri(db_path)
    resolved_port = resolve_port(host, port)

    logger.info(f"Backend store: {backend_uri}")
    logger.info(f"Starting MLflow UI on port {resolved_port} ...")

    command = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_uri,
        "--host",
        host,
        "--port",
        str(resolved_port),
    ]

    subprocess.run(command, check=True)


def main() -> None:
    """Main entry point for command-line usage."""
    args = parse_args()
    run_mlflow_ui(host=args.host, port=args.port, db_path=args.db_path)


if __name__ == "__main__":
    main()
