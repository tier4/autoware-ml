#!/bin/bash
# Entrypoint script to create user matching host UID/GID and switch to it using gosu
set -e

HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}
COMPLETION_DIR_NAME=".bash_completions"
COMPLETION_FILE_NAME="autoware-ml.sh"

# If not running as root, just exec the command
if [ "$(id -u)" != "0" ]; then
    cd /workspace
    exec "$@"
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

ensure_completion_installed() {
    local username="$1"
    local completion_dir="$2"
    local completion_path="$3"
    local bashrc_path="$4"

    if [ ! -f "${completion_path}" ]; then
        install -d -o "${HOST_UID}" -g "${HOST_GID}" "${completion_dir}"
        gosu "${username}" bash -lc "autoware-ml --show-completion bash > '${completion_path}'"
    fi

    if ! grep -Fqx "source '${completion_path}'" "${bashrc_path}"; then
        printf "\nsource '%s'\n" "${completion_path}" >>"${bashrc_path}"
    fi
}

# Create XDG runtime directory (passed via XDG_RUNTIME_DIR env var)
mkdir -p /tmp/xdg
chown "${HOST_UID}:${HOST_GID}" /tmp/xdg
chmod 700 /tmp/xdg

# Export necessary environment variables for gosu
export HOME="${USER_HOME}"
export USER="${USERNAME}"

ensure_completion_installed "${USERNAME}" "${COMPLETION_DIR}" "${COMPLETION_PATH}" "${BASHRC_PATH}"

# Switch to user and exec command
cd /workspace
exec gosu "${USERNAME}" "$@"
