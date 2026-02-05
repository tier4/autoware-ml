FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Core environment variables
ENV AUTOWARE_ML_DATA_PATH="/autoware-ml-data" \
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
  git \
  gosu \
  ninja-build \
  sudo \
  vim \
  && rm -rf /var/lib/apt/lists/* \
  && gosu nobody true

# Create workspace and cache directories (ownership will be set by entrypoint)
RUN mkdir -p /workspace /ccache

# Set workspace
COPY . /workspace/
WORKDIR /workspace

# Install pyproject dependencies
# hadolint ignore=DL3013
RUN pip install --no-cache-dir hatchling read_version uv wheel_stub \
  && uv pip install --system --no-cache .[dev] --no-build-isolation

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
