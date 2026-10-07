# Selects the locked pixi environment the image is built around, and with it the
# `env-<PIXI_ENV>` stage that provides the sparse convolution backend of the PTv3
# family. `dev` (the default) and `default` take spconv from the lockfile;
# `rulebook` has no spconv and compiles `rulebook_torch` from source instead.
# See docs/getting-started/installation.md.
ARG PIXI_ENV=dev

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PIXI_ENV
ARG PIXI_VERSION=0.80.0
ARG USERNAME=autoware-ml
ARG USER_UID=1000
ARG USER_GID=1000

# Core environment variables
ENV AUTOWARE_ML_DATA_PATH="/workspace/data" \
  CUDA_HOME="/usr/local/cuda" \
  FORCE_CUDA="1" \
  LC_ALL=C.UTF-8 \
  PIXI_HOME="/opt/pixi" \
  PYTHONPATH=/workspace \
  SHELL=/bin/bash \
  PATH="/opt/pixi/bin:${PATH}" \
  TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0+PTX" \
  TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Prompt customization
ENV GIT_PS1_DESCRIBE_STYLE=contains \
  GIT_PS1_SHOWCOLORHINTS=1 \
  GIT_PS1_SHOWDIRTYSTATE=1 \
  GIT_PS1_SHOWSTASHSTATE=1 \
  GIT_PS1_SHOWUNTRACKEDFILES=1 \
  GIT_PS1_SHOWUPSTREAM=verbose \
  PROMPT_COMMAND="__git_ps1 '"'${VIRTUAL_ENV:+($(basename "$VIRTUAL_ENV")) }'"\[\033[01;33m\](container) \[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]' '$ '"

# Install APT dependencies
# hadolint ignore=DL3008
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  bash-completion \
  build-essential \
  ca-certificates \
  curl \
  git \
  gosu \
  sudo \
  vim \
  && rm -rf /var/lib/apt/lists/* \
  && gosu nobody true

# Drop the inherited Ubuntu user/group from the base image so the default
# host UID/GID (1000:1000) remains available for autoware-ml remapping, then
# create the fixed container user. The entrypoint remaps this existing user to
# the host UID/GID at runtime for bind-mounted workspaces.
RUN if id -u ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi \
  && if getent group ubuntu >/dev/null 2>&1; then groupdel ubuntu; fi \
  && groupadd --gid "${USER_GID}" "${USERNAME}" \
  && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" \
  && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
  && chmod 0440 /etc/sudoers.d/${USERNAME} \
  && mkdir -p /tmp/xdg \
  && chown "${USERNAME}:${USERNAME}" /tmp/xdg \
  && chmod 700 /tmp/xdg

# Create workspace, cache, and pixi directories, then install a pinned pixi
# release artifact with checksum verification. Project environments live
# outside the bind-mounted workspace.
RUN mkdir -p /workspace /ccache /etc/pixi /opt/pixi/bin /opt/pixi/envs \
  && if [ "$(dpkg --print-architecture)" != "amd64" ]; then echo "Unsupported architecture: $(dpkg --print-architecture)" >&2; exit 1; fi \
  && PIXI_ARCHIVE="pixi-x86_64-unknown-linux-musl.tar.gz" \
  && PIXI_BASE_URL="https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}" \
  && curl -fsSLo "/tmp/${PIXI_ARCHIVE}" "${PIXI_BASE_URL}/${PIXI_ARCHIVE}" \
  && curl -fsSLo "/tmp/${PIXI_ARCHIVE}.sha256" "${PIXI_BASE_URL}/${PIXI_ARCHIVE}.sha256" \
  && PIXI_SHA256="$(awk '{print $1}' "/tmp/${PIXI_ARCHIVE}.sha256")" \
  && echo "${PIXI_SHA256}  /tmp/${PIXI_ARCHIVE}" | sha256sum -c - \
  && tar -xzf "/tmp/${PIXI_ARCHIVE}" -C /opt/pixi/bin pixi \
  && chmod +x /opt/pixi/bin/pixi \
  && rm -f "/tmp/${PIXI_ARCHIVE}" "/tmp/${PIXI_ARCHIVE}.sha256" \
  && printf 'detached-environments = "/opt/pixi/envs"\n[shell]\nchange-ps1 = false\n' > /etc/pixi/config.toml

# Resolve and install the locked environment first for better layer
# reuse. Docker build stages do not expose the host NVIDIA driver, so pixi
# cannot detect the `__cuda` virtual package there. Override it to match the
# repository's pinned CUDA stack during image build.
COPY LICENSE README.md pyproject.toml setup.py pixi.lock /workspace/
WORKDIR /workspace
RUN CONDA_OVERRIDE_CUDA=12.8 /opt/pixi/bin/pixi install --locked --environment "${PIXI_ENV}" && \
    PIXI_HOME=/opt/pixi PIXI_BIN_DIR=/opt/pixi/bin /opt/pixi/bin/pixi shell-hook --manifest-path /workspace --environment "${PIXI_ENV}" --change-ps1 false > /etc/profile.d/autoware-ml.sh

# Sparse convolution backend. The `FROM env-${PIXI_ENV}` below picks the stage
# matching the pixi environment; spconv ships as a wheel in the lockfile, so the
# `default` and `dev` environments need nothing on top of the base stage.
FROM base AS env-default

FROM base AS env-dev

# The `rulebook` environment has no spconv. Its submanifold convolutions come
# from `rulebook_torch`, a CMake build of the `spconv-replacement` repository
# against the pinned torch and the CUDA toolkit of this base image. That
# repository is private, so its sources arrive through the named build context
# `rulebook`, which `docker/build.sh --rulebook <path>` wires to a local
# checkout. CUTLASS is public and is fetched here at the pinned commit.
FROM base AS env-rulebook

ARG PIXI_ENV
ARG CUTLASS_REF=6c68991985ca8b09594ac6fd43abbfd5830c4140
ARG RULEBOOK_ARCHS="90;120"
ARG RULEBOOK_DTYPES="f16;bf16"

COPY --from=rulebook . /opt/rulebook/src

# One layer: the CUTLASS checkout and the build tree are worth several hundred
# megabytes and are of no use once the extension is installed. The module is
# built in place at python/rulebook_torch/_C*.so, and `pip install` ships only
# the Python sources, so the compiled module is copied in afterwards.
RUN mkdir -p /etc/autoware-ml \
  && { git -C /opt/rulebook/src rev-parse HEAD || echo unknown; } > /etc/autoware-ml/rulebook.commit \
  && rm -rf /opt/rulebook/src/.git /opt/rulebook/src/third_party/cutlass \
  && mkdir -p /opt/rulebook/src/third_party/cutlass \
  && git -C /opt/rulebook/src/third_party/cutlass init -q . \
  && git -C /opt/rulebook/src/third_party/cutlass remote add origin https://github.com/NVIDIA/cutlass.git \
  && git -C /opt/rulebook/src/third_party/cutlass fetch -q --depth 1 --filter=blob:none origin "${CUTLASS_REF}" \
  && git -C /opt/rulebook/src/third_party/cutlass checkout -q --detach FETCH_HEAD \
  && CONDA_OVERRIDE_CUDA=12.8 /opt/pixi/bin/pixi run --manifest-path /workspace --environment "${PIXI_ENV}" bash -c '\
       set -euo pipefail; \
       cmake -S /opt/rulebook/src -B /opt/rulebook/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DRULEBOOK_BUILD_TORCH=ON \
         -DRULEBOOK_BUILD_TESTS=OFF \
         -DRULEBOOK_ARCHS="${RULEBOOK_ARCHS}" \
         -DRULEBOOK_DTYPES="${RULEBOOK_DTYPES}" \
         -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
         -DCUDAToolkit_ROOT="${CUDA_HOME}" \
         -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
         -DPython3_EXECUTABLE="$(command -v python)"; \
       cmake --build /opt/rulebook/build --target rulebook_torch -j "$(nproc)"; \
       python -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/rulebook/src/python' \
  && SITE_PACKAGES="$(CONDA_OVERRIDE_CUDA=12.8 /opt/pixi/bin/pixi run --manifest-path /workspace --environment "${PIXI_ENV}" python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" | tail -n 1)" \
  && cp /opt/rulebook/src/python/rulebook_torch/_C*.so "${SITE_PACKAGES}/rulebook_torch/" \
  && chmod -R a+rX "${SITE_PACKAGES}/rulebook_torch" \
  && rm -rf /opt/rulebook

FROM env-${PIXI_ENV} AS final

ARG PIXI_ENV
ARG USERNAME=autoware-ml

# Install the project itself in editable mode inside the environment
COPY . /workspace/
RUN /opt/pixi/bin/pixi run --environment "${PIXI_ENV}" install-project \
  && mkdir -p /usr/share/bash-completion/completions \
  && /opt/pixi/bin/pixi run --environment "${PIXI_ENV}" autoware-ml --show-completion bash \
      > /usr/share/bash-completion/completions/autoware-ml \
  && chown -R "${USERNAME}:${USERNAME}" /workspace /ccache

# Copy and set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh

# Metadata
LABEL maintainer="TIER IV, Inc." \
  description="Autoware-ML - Machine learning framework for Autoware." \
  version="0.0.0" \
  org.opencontainers.image.source="https://github.com/tier4/autoware-ml" \
  org.opencontainers.image.vendor="TIER IV, Inc."

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
