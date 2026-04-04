#!/bin/bash
# Entrypoint script to create a user matching the host UID/GID and switch to it
# using gosu.
set -euo pipefail

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

if ! getent group "${HOST_GID}" >/dev/null 2>&1; then
    groupadd -g "${HOST_GID}" "autoware-ml"
fi

if ! getent passwd "${HOST_UID}" >/dev/null 2>&1; then
    useradd -u "${HOST_UID}" -g "${HOST_GID}" -m -s /bin/bash "autoware-ml"
    echo "autoware-ml ALL=(root) NOPASSWD:ALL" >/etc/sudoers.d/autoware-ml
    chmod 0440 /etc/sudoers.d/autoware-ml
    echo "User 'autoware-ml' was created with UID: ${HOST_UID} and GID: ${HOST_GID}."
fi

USERNAME=$(getent passwd "${HOST_UID}" | cut -d: -f1)
USER_HOME=$(getent passwd "${HOST_UID}" | cut -d: -f6)

mkdir -p /tmp/xdg
chown "${HOST_UID}:${HOST_GID}" /tmp/xdg
chmod 700 /tmp/xdg

export HOME="${USER_HOME}"
export USER="${USERNAME}"

# $(...) and $@ must be expanded by the inner bash, not the outer shell.
# shellcheck disable=SC2016,SC1091
exec gosu "${USERNAME}" bash -lc \
    'cd /workspace && source /etc/profile.d/autoware-ml.sh && exec "$@"' \
    bash "$@"
