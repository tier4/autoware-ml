---
icon: lucide/bug-off
---

# Troubleshooting

## Docker Training Freezes

If training stops and `nvidia-smi` inside the container returns `Failed to initialize NVML: Unknown Error`, while `nvidia-smi` still works on the host, the container has lost GPU access. This is a known NVIDIA Container Toolkit issue, commonly seen when Docker uses the `systemd` cgroup driver.

### Check

```bash
nvidia-smi
docker info --format '{{.CgroupDriver}} {{.CgroupVersion}}'
docker exec -it <container_name> nvidia-smi
```

You are likely hitting this issue if:

- host `nvidia-smi` works
- container `nvidia-smi` fails
- Docker reports `systemd 2`

If host `nvidia-smi` also fails, this is a host GPU or driver problem instead.

### Recover

Recreate the container:

```bash
./docker/container.sh --stop
./docker/container.sh --run
```

If you are using a managed training session, restart the session after the container comes back.

### Mitigate

NVIDIA recommends switching Docker from `systemd` to `cgroupfs`. Update `/etc/docker/daemon.json`
on the host:

```json
{
  "exec-opts": ["native.cgroupdriver=cgroupfs"],
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  }
}
```

Then restart Docker and recreate the container:

```bash
sudo systemctl restart docker
./docker/container.sh --stop
./docker/container.sh --run
```

### Notes

- This is a Docker / NVIDIA runtime issue, not usually a model issue.
- Validate the Docker config change on your machine before adopting it broadly.
- Reference: [NVIDIA Container Toolkit troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html)
