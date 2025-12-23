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

Run the setup playbook (may require sudo):

```bash
ansible-playbook ansible/playbooks/setup_host.yaml -K
```

System reboot is required for NVIDIA driver changes and Docker post-installation steps to take effect.
