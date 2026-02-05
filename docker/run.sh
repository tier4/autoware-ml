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

# shellcheck disable=SC2086,SC2124

set -e

# Define terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
NC='\033[0m' # No Color

SCRIPT_DIR=$(readlink -f "$(dirname "$0")")
WORKSPACE_ROOT="$SCRIPT_DIR/.."

# Default values
option_headless=false
option_detached=false
option_pull_latest_image=false
option_exec=false
option_stop=false
DATA_PATH=""
WORKSPACE=""
MEMORY_CONFIG=""
MODE=""
CONTAINER=""
CONTAINER_NAME=""
USER_ENV=""

# Function to print help message
print_help() {
    echo -e "\n------------------------------------------------------------"
    echo -e "${RED}Usage:${NC} run.sh [OPTIONS]"
    echo -e "Options:"
    echo -e "  ${GREEN}--help/-h${NC}            Display this help message"
    echo -e "  ${GREEN}--data-path${NC}          Specify the path to mount data files into /autoware-ml-data (overrides AUTOWARE_ML_DATA_PATH if set)"
    echo -e "  ${GREEN}--headless${NC}           Run Autoware-ML in headless mode (default: false)"
    echo -e "  ${GREEN}--detached${NC}           Run Autoware-ML in detached mode (default: false)"
    echo -e "  ${GREEN}--pull-latest-image${NC}  Pull the latest image before starting the container"
    echo -e "  ${GREEN}--exec${NC}               Enter an existing running container instead of creating a new one"
    echo -e "  ${GREEN}--stop${NC}               Stop a running container"
    echo ""
}

# Parse arguments
parse_arguments() {
    while [ "$1" != "" ]; do
        case "$1" in
        --help | -h)
            print_help
            exit 1
            ;;
        --headless)
            option_headless=true
            ;;
        --detached)
            option_detached=true
            ;;
        --pull-latest-image)
            option_pull_latest_image=true
            ;;
        --exec)
            option_exec=true
            ;;
        --stop)
            option_stop=true
            ;;
        --data-path)
            DATA_PATH="$2"
            shift
            ;;
        --*)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
        -*)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
        esac
        shift
    done
}

# Set the docker image and workspace variables
set_variables() {
    # Set data path
    if [ "$DATA_PATH" != "" ]; then
        DATA="-v ${DATA_PATH}:/autoware-ml-data:rw"
    elif [ -n "$AUTOWARE_ML_DATA_PATH" ]; then
        DATA="-v ${AUTOWARE_ML_DATA_PATH}:/autoware-ml-data:rw"
    else
        echo -e "${ORANGE}Neither --data-path nor AUTOWARE_ML_DATA_PATH is set. Not mounting any data directory.${NC}"
    fi

    IMAGE="ghcr.io/tier4/autoware-ml:latest"
    WORKSPACE="-v ${WORKSPACE_ROOT}:/workspace -e PYTHONPATH=/workspace"
    MEMORY_CONFIG="--ipc=host --ulimit memlock=-1 --ulimit stack=67108864" # 64MB

    # Set container name based on USER environment variable
    if [ -n "$USER" ]; then
        CONTAINER_NAME="autoware-ml-${USER}"
    else
        CONTAINER_NAME="autoware-ml"
    fi
    CONTAINER="--name ${CONTAINER_NAME}"
}

# Set X display variables
set_x_display() {
    MOUNT_X=""
    if [ "$option_headless" = "false" ]; then
        MOUNT_X="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix"
        xhost + >/dev/null
    fi
}

set_detached_mode() {
    if [ "$option_detached" = "true" ]; then
        MODE="-d"
    fi
}

# Set user configuration to match host user
set_user_config() {
    HOST_UID=$(id -u)
    HOST_GID=$(id -g)
    USER_ENV="-e HOST_UID=${HOST_UID} -e HOST_GID=${HOST_GID} -e XDG_RUNTIME_DIR=/tmp/xdg"
}

# Execute into an existing running container
exec_container() {
    # Check if container exists and is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}Error: Container '${CONTAINER_NAME}' is not running.${NC}"
        echo -e "${ORANGE}For usage instructions, run:${NC}"
        echo -e "  ${ORANGE}./docker/run.sh --help${NC}"
        exit 1
    fi

    echo -e "${GREEN}-----------------------------------------------------------------${NC}"
    echo -e "${GREEN}Entering Autoware-ML container${NC}"
    echo -e "${GREEN}CONTAINER:${NC} ${CONTAINER_NAME}"
    echo -e "${GREEN}-----------------------------------------------------------------${NC}"

    # Use entrypoint to switch to correct user and start bash
    docker exec -it -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "${CONTAINER_NAME}" /entrypoint.sh bash
}

# Stop a running container
stop_container() {
    # Check if container exists and is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}Error: Container '${CONTAINER_NAME}' is not running.${NC}"
        exit 1
    fi

    echo -e "${GREEN}-----------------------------------------------------------------${NC}"
    echo -e "${GREEN}Stopping Autoware-ML container${NC}"
    echo -e "${GREEN}CONTAINER:${NC} ${CONTAINER_NAME}"
    echo -e "${GREEN}-----------------------------------------------------------------${NC}"

    docker stop "${CONTAINER_NAME}"
    echo -e "${GREEN}Container stopped successfully.${NC}"
}

# Run a new container
run() {
    set_x_display
    set_detached_mode
    set_user_config

    echo -e "${GREEN}-----------------------------------------------------------------${NC}"
    echo -e "${GREEN}Launching Autoware-ML${NC}"
    echo -e "${GREEN}IMAGE:${NC} ${IMAGE}"
    echo -e "${GREEN}-----------------------------------------------------------------${NC}"

    if [ "$option_pull_latest_image" = "true" ]; then
        docker pull "${IMAGE}"
    fi

    # Launch the container
    set -x
    docker run -it --rm --net=host --gpus all ${MODE} ${MEMORY_CONFIG} ${MOUNT_X} \
        -e XAUTHORITY=${XAUTHORITY} -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e TZ="$(cat /etc/timezone)" \
        ${USER_ENV} \
        ${WORKSPACE} ${DATA} ${CONTAINER} ${IMAGE}
}

# Main script execution
main() {
    parse_arguments "$@"
    set_variables

    if [ "$option_stop" = "true" ]; then
        stop_container
    elif [ "$option_exec" = "true" ]; then
        exec_container
    else
        run
    fi
}

# Execute the main script
main "$@"
