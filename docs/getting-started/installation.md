---
icon: lucide/download
---

# Installation

Autoware-ML runs well in a Docker container with GPU support. We encourage you to use Docker for the smoothest experience during Early Alpha.

---

## Prerequisites

1. **Autoware-ML**

  ```bash
  cd
  git clone https://github.com/tier4/autoware-ml.git
  cd autoware-ml
  ```

1. **NVIDIA GPU** with CUDA support (Compute Capability 8.0+)
2. **NVIDIA Driver** version 570 or higher
3. **Docker Engine**
4. **NVIDIA Container Toolkit**

### Checking Your Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker
docker --version

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

If all three commands succeed, skip to [Project Setup](#project-setup).

---

## Host Setup

=== "Automated Setup with Ansible"

    We provide Ansible playbooks to install all prerequisites automatically:

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

    # Run the setup playbook (it will ask for sudo password)
    ansible-playbook ansible/playbooks/setup_host.yaml -K
    ```

    This installs Docker, NVIDIA drivers, NVIDIA Container Toolkit and VS Code with extensions in one command.
    System reboot is required for NVIDIA driver changes and Docker post-installation steps to take effect.

=== "Manual Setup"

    If you prefer to install components individually, see the tabs below.

    === "NVIDIA Drivers"
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
    ./docker/run.sh --data-path /path/to/your/datasets
    ```

=== "Without Docker"

    !!! warning "Not Recommended for Alpha"
        Local installation requires careful dependency management. We recommend Docker for the smoothest experience during Early Alpha.

    If you still want to install locally:

    === "With uv"
        ```bash
        cd ~/autoware-ml

        # Install (creates virtual environment automatically)
        uv sync --extra dev
        ```

    === "With pip"

        ```bash
        cd ~/autoware-ml

        # Create a virtual environment
        python -m venv .venv
        source .venv/bin/activate

        # Install
        pip install --no-cache-dir .[dev] --extra-index-url https://download.pytorch.org/whl/cu128 --no-build-isolation
        ```

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

- `./docker/run.sh`
- `.devcontainer/devcontainer.json`
- Model config files

## Next Steps

Navigate to [Quick Start](quickstart.md) to start training your first model.

!!! tip "Dev Containers"
    For the best development experience, see [Dev Containers](devcontainer.md) first.
