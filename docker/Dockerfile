FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PIXI_VERSION=0.66.0

# Core environment variables
ENV AUTOWARE_ML_DATA_PATH="/workspace/data" \
  CUDA_HOME="/usr/local/cuda" \
  FORCE_CUDA="1" \
  LC_ALL=C.UTF-8 \
  PYTHONPATH=/workspace \
  SHELL=/bin/bash \
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

# Create workspace, cache, and pixi directories, then install pixi with the
# standard upstream installer. Project environments live outside the
# bind-mounted workspace.
RUN mkdir -p /workspace /ccache /etc/pixi /opt/pixi/bin /opt/pixi/envs \
  && export PIXI_HOME=/opt/pixi PIXI_BIN_DIR=/opt/pixi/bin PIXI_NO_PATH_UPDATE=1 PIXI_VERSION="${PIXI_VERSION}" \
  && curl -fsSL https://pixi.sh/install.sh | sh \
  && printf 'detached-environments = "/opt/pixi/envs"\n[shell]\nchange-ps1 = false\n' > /etc/pixi/config.toml

# Resolve and install the locked environment first for better layer
# reuse. Docker build stages do not expose the host NVIDIA driver, so pixi
# cannot detect the `__cuda` virtual package there. Override it to match the
# repository's pinned CUDA stack during image build.
COPY LICENSE README.md pyproject.toml setup.py pixi.lock /workspace/
WORKDIR /workspace
RUN CONDA_OVERRIDE_CUDA=12.8 /opt/pixi/bin/pixi install --locked --environment dev && \
    /opt/pixi/bin/pixi shell-hook --manifest-path /workspace --environment dev --change-ps1 false > /etc/profile.d/autoware-ml.sh

# Install the project itself in editable mode inside the environment
COPY . /workspace/
RUN /opt/pixi/bin/pixi run --environment dev install-project \
  && mkdir -p /usr/share/bash-completion/completions \
  && /opt/pixi/bin/pixi run --environment dev autoware-ml --show-completion bash \
      > /usr/share/bash-completion/completions/autoware-ml \
  && chown -R 1000:1000 /workspace /ccache

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
