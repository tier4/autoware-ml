#!/bin/bash
# Entrypoint script to remap the fixed autoware-ml user to the host UID/GID
# and switch to it using gosu.
set -euo pipefail

USERNAME=autoware-ml
USER_HOME="/home/${USERNAME}"
HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}

run_in_autoware_ml_env() {
    cd /workspace
    # shellcheck disable=SC1091
    source /etc/profile.d/autoware-ml.sh
    exec "$@"
}

if [ "$(id -u)" != "0" ]; then
    run_in_autoware_ml_env "$@"
fi

CURRENT_UID=$(id -u "${USERNAME}")
CURRENT_GID=$(id -g "${USERNAME}")

if [ "${CURRENT_GID}" != "${HOST_GID}" ]; then
    groupmod -o -g "${HOST_GID}" "${USERNAME}"
    usermod -g "${HOST_GID}" "${USERNAME}"
fi

if [ "${CURRENT_UID}" != "${HOST_UID}" ]; then
    usermod -o -u "${HOST_UID}" "${USERNAME}"
fi

mkdir -p /tmp/xdg /ccache
chown -R "${HOST_UID}:${HOST_GID}" "${USER_HOME}" /ccache /tmp/xdg
chmod 700 /tmp/xdg

if ! mountpoint -q /workspace; then
    chown -R "${HOST_UID}:${HOST_GID}" /workspace
fi

export HOME="${USER_HOME}"
export USER="${USERNAME}"

# $(...) and $@ must be expanded by the inner bash, not the outer shell.
# shellcheck disable=SC2016,SC1091
exec gosu "${USERNAME}" bash -lc \
    'cd /workspace && source /etc/profile.d/autoware-ml.sh && exec "$@"' \
    bash "$@"
