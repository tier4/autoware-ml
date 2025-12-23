# hadolint global ignore=DL3006,DL3008,DL3013,DL3016
ARG BASE_IMAGE=ghcr.io/tier4/autoware-ml:latest
FROM $BASE_IMAGE

# Set shell to bash with pipefail for safer pipe handling
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

# Update ROS key and repository configuration, then install packages via apt-get
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  bash-completion \
  bear \
  curl \
  gdb \
  gdbserver \
  htop \
  jq \
  nvidia-cuda-gdb \
  openssh-server \
  unzip \
  wget \
  && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* "$HOME"/.cache

# Install CSpell
RUN curl -sL https://deb.nodesource.com/setup_22.x -o /tmp/nodesource_setup.sh \
  && bash /tmp/nodesource_setup.sh \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends nodejs -y \
  && npm install -g cspell \
  && npm install -g yarn \
  && rm -rf /tmp/nodesource_setup.sh \
  && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* "$HOME"/.cache

# Install clangd
RUN CLANGD_VERSION=$(curl -fsSL "https://api.github.com/repos/clangd/clangd/releases/latest" | jq -r '.tag_name') \
  && echo "Installing clangd version: $CLANGD_VERSION" \
  && apt-get update \
  && curl -fsSL -o /tmp/clangd-linux.zip "https://github.com/clangd/clangd/releases/download/${CLANGD_VERSION}/clangd-linux-${CLANGD_VERSION}.zip" \
  && unzip /tmp/clangd-linux.zip -d /opt/ \
  && ln -s "/opt/clangd_${CLANGD_VERSION}/bin/clangd" /usr/local/bin/clangd \
  && rm -rf /tmp/clangd-linux.zip \
  && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* "$HOME"/.cache

# Switch back user
USER autoware-ml

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD echo "Container is healthy"
