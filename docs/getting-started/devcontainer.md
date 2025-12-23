---
icon: lucide/container
---

# Dev Containers

VS Code dev containers give you a fully configured development environment with just a few steps. This is the recommended setup for active development on Autoware-ML.

## Prerequisites

1. **VS Code** ([code.visualstudio.com](https://code.visualstudio.com/))
2. **Dev Containers extension** ([marketplace.visualstudio.com](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers))
3. **Docker** ([docs.docker.com](https://docs.docker.com/engine/install))

!!! note
    You can install the prerequisites using the **Automated Setup with Ansible** in the [Host Setup](installation.md#host-setup) section of the Installation guide.

## Opening the Dev Container

1. Open VS Code
2. `Ctrl+Shift+P` and select `Dev Containers: Open Folder in Container`
3. Select the `autoware-ml` folder and click `Open`

VS Code will build the container (first time only, ~5 minutes) and open the workspace in the container.

## What's Included

The dev container comes with:

- Recommended extensions
- Debugging utilities
- clangd for C/C++
- cSpell for spell checking

In addition, VS Code settings includes:

- Debugging configurations
- Pre-commit hooks

On the first run, you may wait for a while to install the dependencies.

## Debugging

### Python Debugging

1. Set breakpoints in your code
2. Open the Run and Debug panel (`Ctrl+Shift+D`)
3. Select "Python"
4. Press `F5`
5. Fill in the input fields:
    - Command: Pick available commands from the dropdown
    - Config: Type the config name e.g. `calibration_status/resnet18_nuscenes`
    - Arguments: Type the extra arguments you want to pass to the command e.g. `datamodule.train_dataloader_cfg.batch_size=2 datamodule.val_dataloader_cfg.batch_size=2`

For custom commands, you can add a new launch configuration to `.vscode/launch.json`.

## Tools

### Pre-commit Hooks

1. `Ctrl+Shift+P` and select `Tasks: Run Task`
2. Select "Pre-commit: Run"
3. Select the pre-commit config to use from the dropdown

On the first run, you may wait for a while to initialize the environment.

### CSpell

The actual unknown words for currently opened files are visible in the "Problems" panel (`Ctrl+Shift+M`).

To check for spelling errors in the entire workspace:

1. `Ctrl+Shift+P` and select `Tasks: Run Task`
2. Select "CSpell: Check"
