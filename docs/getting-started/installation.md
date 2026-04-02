---
icon: lucide/download
---

# Installation

Autoware-ML runs well in a Docker container with GPU support. We encourage you to use Docker for the smoothest experience during Early Alpha.

---

## Prerequisites

- **Autoware-ML**

  ```bash
  cd
  git clone https://github.com/tier4/autoware-ml.git
  cd autoware-ml
  ```

- **NVIDIA GPU** with CUDA support (Compute Capability 8.0+)

- **NVIDIA Driver** version 570 or higher

    === "With Docker"
        - **Docker Engine** version 20.10 or higher
        - **NVIDIA Container Toolkit** for GPU support in Docker

    === "Without Docker"
        - **NVIDIA CUDA Toolkit** for local development and building CUDA-backed native dependencies and ops

---

## Host Setup

=== "Automated Setup with Ansible"

    We provide separate Ansible playbooks for Docker-based and local
    development:

    ```bash

    # Remove apt-installed Ansible (In Ubuntu 22.04, the Ansible version is old)
    sudo apt purge ansible

    # Install pip
    sudo apt -y update
    sudo apt -y install python3-pip

    # Install Ansible (if not already installed)
    sudo python3 -m pip install ansible==9.13.0

    # Install required Ansible collections
    cd ~/autoware-ml
    ansible-galaxy collection install -f -r ansible-galaxy-requirements.yaml

    # Docker-based development host
    ansible-playbook ansible/playbooks/setup_docker_host.yaml -K

    # Local pixi development host
    ansible-playbook ansible/playbooks/setup_local_host.yaml -K
    ```

    The Docker playbook installs Docker Engine, NVIDIA drivers, NVIDIA Container Toolkit, and
    optionally VS Code with extensions. The local playbook installs NVIDIA drivers, the NVIDIA CUDA
    Toolkit, and optionally VS Code with extensions. System reboot is required for NVIDIA driver
    changes and Docker post-installation steps to take effect.

=== "Manual Setup"

    If you prefer to install components individually, see the tabs below.

    === "NVIDIA Drivers"
        Check if you have a compatible NVIDIA driver installed:
        ```bash
        nvidia-smi
        ```

        If not installed or outdated:

        ```bash
        # Add NVIDIA driver repository
        sudo add-apt-repository ppa:graphics-drivers/ppa
        sudo apt-get update

        # Install prerequisites
        sudo apt-get install -y software-properties-common build-essential dkms

        # Install NVIDIA driver (version 580 recommended)
        sudo apt-get install -y nvidia-driver-580

        # Reboot required
        sudo reboot
        ```

        After rebooting, verify with `nvidia-smi`.

    === "Docker Engine"
        Remove any old Docker installations:

        ```bash
        sudo apt-get remove docker docker-engine docker.io containerd runc
        ```

        Install Docker from the official repository:

        ```bash
        # Install dependencies
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg lsb-release

        # Create keyrings directory
        sudo mkdir -p /etc/apt/keyrings

        # Add Docker's GPG key
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        sudo chmod 644 /etc/apt/keyrings/docker.asc

        # Add the repository
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
            https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
            sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null

        # Install Docker
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

        # Verify installation
        sudo docker run hello-world
        ```

        Post-Installation steps for running Docker without sudo:

        ```bash
        # Create docker group
        sudo groupadd docker

        # Add your user to the group
        sudo usermod -aG docker $USER

        # Log out and back in, then verify
        docker run hello-world
        ```

    === "NVIDIA Container Toolkit"
        This enables Docker to access your GPU:

        ```bash
        # Add NVIDIA GPG key
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        # Add repository
        echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
            https://nvidia.github.io/libnvidia-container/stable/deb/$(dpkg --print-architecture) /" | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

        # Install
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit

        # Configure Docker runtime
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        ```

        Verify GPU access in Docker:

        ```bash
        docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
        ```

        You should see your GPU information printed.

    === "NVIDIA CUDA Toolkit"
        This gives you `nvcc` and CUDA libraries for local development and building CUDA-backed packages from source:

        ```bash
        UBUNTU_MAJOR_VERSION="$(. /etc/os-release && echo "${VERSION_ID%%.*}")"

        if [ "$(uname -m)" != "x86_64" ]; then
          echo "Unsupported architecture: $(uname -m)" >&2
          exit 1
        fi

        sudo apt-get update
        sudo apt-get install -y wget
        wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_MAJOR_VERSION}04/x86_64/cuda-keyring_1.1-1_all.deb"
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update

        # NVIDIA's guide uses `cuda-toolkit`; Autoware-ML pins the 12.8 series
        # to match the repository's current CUDA stack.
        sudo apt-get install -y cuda-toolkit-12-8

        cat <<'EOF' | sudo tee /etc/profile.d/cuda-toolkit.sh >/dev/null
        export CUDA_HOME=/usr/local/cuda
        export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
        EOF

        sudo reboot

        # After reboot, open a new shell
        nvcc --version
        ```

---

## Project Setup

=== "With Docker"

    Pull the latest Docker image from our registry:

    ```bash
    docker pull ghcr.io/tier4/autoware-ml:latest
    ```

    If you need to modify the Docker image or can't pull from our registry:

    ```bash
    cd ~/autoware-ml
    ./docker/build.sh
    ```

    Then run with:

    ```bash
    ./docker/container.sh --run --data-path /path/to/your/datasets
    ```

    The container image resolves and installs the locked contributor
    `pixi` environment (`dev`) while keeping the CUDA base image as the system
    layer.

=== "Without Docker"

    !!! warning "Not Recommended for Alpha"
        Local installation requires careful dependency management. We recommend Docker for the smoothest experience during Early Alpha.

    Local installation uses the same locked `pixi` environments as Docker.
    Before running `pixi`, make sure the machine-level GPU prerequisites are
    already installed:

    - NVIDIA driver compatible with CUDA 12.8
    - CUDA toolkit with `nvcc` available on `PATH`

    The local `dev` environment can still build CUDA-backed native
    dependencies and Autoware-ML ops, so the CUDA toolkit is a required local
    prerequisite even though Docker keeps that system layer inside the image.

    Then install `pixi` and choose the environment that matches your workflow:

    ```bash
    PIXI_VERSION="0.66.0"

    mkdir -p "$HOME/.pixi/bin"
    curl -fsSL -o pixi-x86_64-unknown-linux-musl.tar.gz "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz"
    curl -fsSL -o pixi-x86_64-unknown-linux-musl.tar.gz.sha256 "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz.sha256"
    sha256sum -c pixi-x86_64-unknown-linux-musl.tar.gz.sha256
    tar -xzf pixi-x86_64-unknown-linux-musl.tar.gz -C "$HOME/.pixi/bin"
    rm -f pixi-x86_64-unknown-linux-musl.tar.gz pixi-x86_64-unknown-linux-musl.tar.gz.sha256
    export PATH="$HOME/.pixi/bin:$PATH"

    cd ~/autoware-ml
    ```

    Then choose **one** of the two environments below:

    === "Runtime / use only"

        ```bash
        pixi install --locked --environment default
        pixi run --environment default install-project
        pixi shell --environment default
        ```

    === "Contributor (recommended)"

        Includes the full runtime stack plus tmux, compilers, docs tooling, and
        build utilities.

        ```bash
        pixi install --locked --environment dev
        pixi run --environment dev install-project
        pixi shell --environment dev
        ```

    The separate `docs` environment is reserved for documentation-only
    workflows and CI — you do not need to install it manually.

---

## Dataset Setup

We assume all datasets are stored in the same directory. You can organize paths as you prefer, but you will need to update our configuration files to match your dataset paths. The recommended structure is:

```text
/path/to/your/datasets
                  ├── nuscenes
                  ├── t4dataset
                  ├── ...
```

You can set the internal environment variable `AUTOWARE_ML_DATA_PATH` using the provided script:

```bash
cd ~/autoware-ml
./set_data_path.sh /path/to/your/datasets
source ~/.bashrc
```

The following files will use this variable to locate your datasets:

- `./docker/container.sh --run`
- `.devcontainer/devcontainer.json`
- Model config files

## Next Steps

Navigate to [Quick Start](quickstart.md) to start training your first model.

!!! tip "Dev Containers"
    For the best development experience, see [Dev Containers](devcontainer.md) first.
