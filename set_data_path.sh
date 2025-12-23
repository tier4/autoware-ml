#!/usr/bin/env bash

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

set -e

VAR_NAME="AUTOWARE_ML_DATA_PATH"
NEW_PATH="$1"
BASHRC_FILE="$HOME/.bashrc"

show_help() {
    cat <<EOF
Usage: $0 <path>

Set the root directory path for datasets in your environment.

IMPORTANT: The path should be the root directory containing datasets, not a path
to a specific dataset. Dataset-specific directories (e.g., nuscenes) are added
in configuration files.

Examples:
  Correct:   $0 /data/datasets
  Incorrect: $0 /data/datasets/nuscenes

The script will:
  - Add or update ${VAR_NAME} in ${BASHRC_FILE}
  - Skip execution if running inside a Docker container

EOF
}

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

if grep -qE '/docker|/lxc' /proc/1/cgroup 2>/dev/null ||
    [ -f /.dockerenv ]; then
    echo "Info: Running in a container environment. Skipping .bashrc modification."
    exit 0
fi

if [ -z "$NEW_PATH" ]; then
    echo "Error: No path provided."
    echo ""
    show_help
    exit 1
fi

NEW_PATH="${NEW_PATH%/}"

sed -i "/^export ${VAR_NAME}=.*/d" "$BASHRC_FILE"

echo "export ${VAR_NAME}=\"$NEW_PATH\"" >>"$BASHRC_FILE"

echo "Successfully set ${VAR_NAME}=\"$NEW_PATH\" in ${BASHRC_FILE}"
echo ""
echo "Note: This path should be the root directory for datasets (e.g., /data/datasets),"
echo "      not a path to a specific dataset. Dataset-specific directories are"
echo "      configured separately in configuration files."
