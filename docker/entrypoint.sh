#!/bin/bash
# Entrypoint script to create user matching host UID/GID and switch to it using gosu
set -e

HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}
COMPLETION_DIR_NAME=".bash_completions"
COMPLETION_FILE_NAME="autoware-ml.sh"

run_in_pixi_env() {
    cd /workspace
    # Keep the existing prompt customization instead of letting pixi rewrite PS1.
    # Docker shells use the contributor environment by default.
    eval "$(pixi shell-hook --manifest-path /workspace --environment dev --change-ps1 false)"
    exec "$@"
}

# If not running as root, activate the contributor pixi environment and execute the command.
if [ "$(id -u)" != "0" ]; then
    run_in_pixi_env "$@"
fi

# Create group and user if they don't exist
if ! getent group "${HOST_GID}" >/dev/null 2>&1; then
    groupadd -g "${HOST_GID}" "autoware-ml"
fi

if ! getent passwd "${HOST_UID}" >/dev/null 2>&1; then
    useradd -u "${HOST_UID}" -g "${HOST_GID}" -m -s /bin/bash "autoware-ml"
    echo "autoware-ml ALL=(root) NOPASSWD:ALL" >/etc/sudoers.d/autoware-ml
    chmod 0440 /etc/sudoers.d/autoware-ml
    echo "User 'autoware-ml' was created with UID: ${HOST_UID} and GID: ${HOST_GID}."
fi

# Get actual username and home directory for the UID
USERNAME=$(getent passwd "${HOST_UID}" | cut -d: -f1)
USER_HOME=$(getent passwd "${HOST_UID}" | cut -d: -f6)
COMPLETION_DIR="${USER_HOME}/${COMPLETION_DIR_NAME}"
COMPLETION_PATH="${COMPLETION_DIR}/${COMPLETION_FILE_NAME}"
BASHRC_PATH="${USER_HOME}/.bashrc"
PIXI_ENV_DIR="${PIXI_HOME:-/opt/pixi}/envs"

ensure_completion_installed() {
    local username="$1"
    local completion_dir="$2"
    local completion_path="$3"
    local bashrc_path="$4"

    if [ ! -f "${completion_path}" ]; then
        install -d -o "${HOST_UID}" -g "${HOST_GID}" "${completion_dir}"
        gosu "${username}" bash -lc "cd /workspace && pixi run --manifest-path /workspace --environment dev autoware-ml --show-completion bash > '${completion_path}'"
    fi

    if ! grep -Fqx "source '${completion_path}'" "${bashrc_path}"; then
        printf "\nsource '%s'\n" "${completion_path}" >>"${bashrc_path}"
    fi
}

ensure_pixi_env_dir_writable() {
    install -d -o "${HOST_UID}" -g "${HOST_GID}" "${PIXI_ENV_DIR}"
}

# Create XDG runtime directory (passed via XDG_RUNTIME_DIR env var)
mkdir -p /tmp/xdg
chown "${HOST_UID}:${HOST_GID}" /tmp/xdg
chmod 700 /tmp/xdg

# Export necessary environment variables for gosu
export HOME="${USER_HOME}"
export USER="${USERNAME}"

ensure_pixi_env_dir_writable
ensure_completion_installed "${USERNAME}" "${COMPLETION_DIR}" "${COMPLETION_PATH}" "${BASHRC_PATH}"

# Switch to user and execute commands inside the contributor pixi environment.
# $(...) and $@ must be expanded by the inner bash, not the outer shell — single quotes are intentional.
# shellcheck disable=SC2016
exec gosu "${USERNAME}" bash -lc \
    'cd /workspace && eval "$(pixi shell-hook --manifest-path /workspace --environment dev --change-ps1 false)" && exec "$@"' \
    bash "$@"
