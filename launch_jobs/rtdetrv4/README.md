# W&B Launch Local Demo

This directory contains a minimal W&B Launch setup for testing Autoware-ML on a
single local GPU workstation.

## What This Tests

- Docker-backed Launch queue.
- One local Launch agent polling one queue.
- A queued RT-DETRv4 train job.
- Live W&B run tracking from the launched job.
- Basic job status, logs, metrics, and artifacts.

This does not test Kubernetes, SageMaker, Vertex AI, queue observability
dashboards, registry automations, or retry policies.

## Pro Plan Feature Notes

Based on current W&B public docs/pricing:

- Launch queues/agents and Docker execution are available for this local test.
- Pro includes CI/CD automations, Slack/email alerts, service accounts, team access controls, Reports, Tables, Artifacts, and Registry.
- Service accounts do not consume user seats, but current docs say service accounts are available on Dedicated Cloud, Self-Managed Enterprise, and Enterprise accounts in Multi-tenant Cloud; confirm your account UI because pricing pages also list service accounts under Pro.
- Queue monitoring dashboards are documented separately and may depend on deployment option.
- Enterprise is still needed for stronger security/compliance features such as dedicated/single-tenant deployment, SSO/SCIM/custom roles, private connectivity, customer-managed keys, and audit/compliance controls.

Docs:

- Launch concepts: https://docs.wandb.ai/platform/launch/launch-terminology
- Docker Launch setup: https://docs.wandb.ai/platform/launch/setup-launch-docker
- Launch agent CLI: https://docs.wandb.ai/models/ref/cli/wandb-launch-agent
- Launch CLI: https://docs.wandb.ai/models/ref/cli/wandb-launch
- Automations: https://docs.wandb.ai/models/automations
- Pricing: https://wandb.ai/site/pricing/

## 1. Login

Run this on the machine that will host the Launch agent:

```bash
cd /home/jacoblambert/rtdetrv4-dev/autoware-ml

export WANDB_API_KEY=<your-api-key>
export WANDB_ENTITY=team-future-solution-ai
export WANDB_PROJECT=mlops

wandb login "$WANDB_API_KEY"
```

## 2. Create A Docker Queue

In the W&B UI:

1. Open Launch.
2. Create queue: `autoware-ml-local-gpu`.
3. Entity: your team.
4. Resource: Docker.
5. Paste a queue config based on `docker-queue-config.template.json`.

Replace placeholders:

```json
"<HOST_AUTOWARE_ML_REPO>": "/home/jacoblambert/rtdetrv4-dev/autoware-ml"
"<HOST_PUBLIC_DATA>": "/home/jacoblambert/public_data"
```

The queue config passes Docker options such as `--gpus all`, bind mounts, and
W&B environment variables to each launched job container.

## 3. Start The Agent

From the host or from a shell that can access Docker:

```bash
cd /home/jacoblambert/rtdetrv4-dev

source /home/jacoblambert/rtdetrv4-dev/autoware-ml/wandb_local_env.sh

wandb launch-agent \
  --entity "$WANDB_ENTITY" \
  --queue autoware-ml-local-gpu \
  --max-jobs 1 \
  --log-file -
```

Alternative with config file:

```bash
wandb launch-agent \
  --entity "$WANDB_ENTITY" \
  --config /home/jacoblambert/rtdetrv4-dev/mlops-tooling-eval/wandb-launch/launch-agent.yaml \
  --log-file -
```

## 4. Submit A Job

Run submission commands from the host, not from inside the Autoware-ML training
container. Source `wandb_local_env.sh` for host-side W&B paths; the launched
training container sources `/workspace/wandb_env.sh` for container-side paths.

```bash
source /home/jacoblambert/rtdetrv4-dev/autoware-ml/wandb_local_env.sh

wandb job create code /home/jacoblambert/rtdetrv4-dev/autoware-ml/launch_jobs/rtdetrv4 \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name rtdetrv4-train-launch \
  --entry-point "bash /workspace/launch_jobs/rtdetrv4/rtdetrv4_train_launch.sh" \
  --base-image ghcr.io/tier4/autoware-ml:latest
```

Then queue the job:

```bash
wandb launch \
  --job "$WANDB_ENTITY/$WANDB_PROJECT/rtdetrv4-train-launch:v0" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --queue autoware-ml-local-gpu \
  --resource local-container \
  --resource-args /tmp/autoware_ml_launch_resource_args.json \
  --name rtdetrv4-launch-smoke \
  --priority medium
```

Use the exact job version printed by `wandb job create` if it is not `v0`.

The script defaults to a small bounded run:

- `RTDETRV4_MAX_EPOCHS=1`
- `RTDETRV4_MAX_TRAIN_SAMPLES=128`
- `RTDETRV4_MAX_VAL_SAMPLES=32`
- `RTDETRV4_TRAIN_BATCH_SIZE=8`
- `RTDETRV4_VAL_BATCH_SIZE=8`

Override with environment variables in the queue config if needed.

## 5. Verify

In W&B:

- The queued job appears in Launch.
- The agent picks it up.
- The run appears in the `mlops` project.
- Metrics update online.
- Artifacts appear after completion.
- Agent logs show the Docker run and job lifecycle.

On the machine:

```bash
docker ps
docker logs <launched-container>
```

## Expected Limitations

- Queue selection is the main routing mechanism. Use one queue per machine or resource class.
- Multiple agents on one queue provide first-available execution, not Slurm-style scheduling.
- `--max-jobs` limits agent concurrency.
- Early stopping is available through W&B Sweeps, not as a general Launch policy.
- Automatic retry/relaunch policy is not yet validated.
- Machine availability is basic unless using the queue observability dashboard or external monitoring.
