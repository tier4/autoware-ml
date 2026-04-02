# nvidia_cuda_toolkit

This role installs the NVIDIA CUDA Toolkit for local development on Ubuntu
22.04/24.04 x86_64 hosts using NVIDIA's current network-repository flow and
configures `CUDA_HOME` and `PATH` through `/etc/profile.d/cuda-toolkit.sh`.

## Inputs

| Variable                               | Description                                  | Default             |
| -------------------------------------- | -------------------------------------------- | ------------------- |
| `nvidia_cuda_toolkit_package`          | CUDA toolkit apt package to install          | `cuda-toolkit-12-8` |
| `nvidia_cuda_toolkit_keyring_checksum` | SHA-256 checksum for the `cuda-keyring` deb  | pinned in defaults  |
| `nvidia_cuda_toolkit_keyring_version`  | Version of the NVIDIA `cuda-keyring` package | `1.1`               |

`cuda-toolkit-12-8` is used by default to lock local development to the CUDA
12.8 series used by this repository. NVIDIA's guide also supports the generic
`cuda-toolkit` package if you intentionally want to track the latest toolkit.

## Manual Installation

```bash
UBUNTU_MAJOR_VERSION="$(. /etc/os-release && echo "${VERSION_ID%%.*}")"

if [ "$(uname -m)" != "x86_64" ]; then
  echo "Unsupported architecture: $(uname -m)" >&2
  exit 1
fi

wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_MAJOR_VERSION}04/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8

cat <<'EOF' | sudo tee /etc/profile.d/cuda-toolkit.sh >/dev/null
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
EOF

sudo reboot

# After reboot, open a new shell
nvcc --version
```
