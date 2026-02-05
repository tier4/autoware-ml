#!/bin/bash
# Entrypoint script to create user matching host UID/GID and switch to it using gosu
set -e

HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}

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
    gosu "${HOST_UID}" autoware-ml --install-completion bash
    echo "User 'autoware-ml' was created with UID: ${HOST_UID} and GID: ${HOST_GID}."
fi

# Get actual username and home directory for the UID
USERNAME=$(getent passwd "${HOST_UID}" | cut -d: -f1)
USER_HOME=$(getent passwd "${HOST_UID}" | cut -d: -f6)

# Create XDG runtime directory (passed via XDG_RUNTIME_DIR env var)
mkdir -p /tmp/xdg
chown "${HOST_UID}:${HOST_GID}" /tmp/xdg
chmod 700 /tmp/xdg

# Export necessary environment variables for gosu
export HOME="${USER_HOME}"
export USER="${USERNAME}"

# Switch to user and exec command
cd /workspace
exec gosu "${USERNAME}" "$@"
