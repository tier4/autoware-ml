# Ansible Collection - autoware_ml.dev_env

This collection contains the playbooks to set up the development environment for Autoware-ML.

## Set up a development environment

### Ansible installation

```bash
# Remove apt installed ansible (In Ubuntu 22.04, ansible version is old)
sudo apt purge ansible

# Install pip
sudo apt -y update
sudo apt -y install python3-pip

# Install ansible
python3 -m pip install ansible==9.13.0
```

### Install ansible collections

This step should be repeated when a new playbook is added.

```bash
cd ~/autoware-ml # The root directory of the cloned repository
ansible-galaxy collection install -f -r ansible-galaxy-requirements.yaml
```

Run the setup playbook for your workflow (may require sudo):

```bash
ansible-playbook ansible/playbooks/setup_docker_host.yaml -K

# or, for local pixi development without Docker
ansible-playbook ansible/playbooks/setup_local_host.yaml -K
```

System reboot is required for NVIDIA driver changes and Docker post-installation
steps to take effect. The local playbook also installs `bash-completion`.
For local development, start a new shell session after installing the NVIDIA
CUDA Toolkit or Autoware-ML so `nvcc`, `CUDA_HOME`, and CLI completion are
available.
