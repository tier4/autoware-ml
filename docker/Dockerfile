FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ARG USERNAME=autoware-ml
ARG USER_UID=1000
ARG USER_GID=$USER_UID

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Core environment variables
ENV AUTOWARE_ML_DATA_PATH="/workspace/data" \
  CUDA_HOME="/usr/local/cuda" \
  FORCE_CUDA="1" \
  LC_ALL=C.UTF-8 \
  PATH=$PATH:/home/$USERNAME/.local/bin \
  PYTHONPATH=/workspace \
  SHELL=/bin/bash \
  TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0+PTX" \
  TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
  XDG_RUNTIME_DIR=/tmp/xdg/$USER_UID

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
  git \
  ninja-build \
  sudo \
  vim \
  && rm -rf /var/lib/apt/lists/*

# Add user
RUN groupadd --gid "$USER_GID" "$USERNAME" \
  && useradd --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME" \
  && echo "$USERNAME" ALL=\(root\) NOPASSWD:ALL > "/etc/sudoers.d/$USERNAME" \
  && chmod 0440 "/etc/sudoers.d/$USERNAME" \
  && mkdir -p /workspace /ccache "$XDG_RUNTIME_DIR" \
  && chown -R "$USERNAME":"$USERNAME" /workspace /ccache "$XDG_RUNTIME_DIR" \
  && chmod 700 "$XDG_RUNTIME_DIR"

# Set workspace
COPY --chown=$USERNAME:$USERNAME . /workspace/
WORKDIR /workspace

# Install pyproject dependencies
# hadolint ignore=DL3013
RUN pip install --no-cache-dir hatchling read_version uv wheel_stub \
  && uv pip install --system --no-cache .[dev] --no-build-isolation

# Set user
USER $USERNAME

# Install bash completion for user
RUN autoware-ml --install-completion bash || true

# Metadata
LABEL maintainer="TIER IV, Inc." \
  description="Autoware-ML - Machine learning framework for Autoware." \
  version="0.0.0" \
  org.opencontainers.image.source="https://github.com/tier4/autoware-ml" \
  org.opencontainers.image.vendor="TIER IV, Inc."

CMD ["/bin/bash"]
