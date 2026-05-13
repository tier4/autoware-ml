#!/usr/bin/env bash
set -euo pipefail

if [[ -x /workspace/docker/entrypoint.sh && "${AUTOWARE_ML_LAUNCH_IN_ENTRYPOINT:-0}" != "1" ]]; then
  export AUTOWARE_ML_LAUNCH_IN_ENTRYPOINT=1
  exec /workspace/docker/entrypoint.sh bash "$0" "$@"
fi

cd /workspace

if [[ -f /workspace/wandb_env.sh ]]; then
  # Container-side W&B paths. Host-side W&B commands should source wandb_local_env.sh instead.
  # shellcheck source=/dev/null
  source /workspace/wandb_env.sh
fi

export AUTOWARE_ML_DATA_PATH="${AUTOWARE_ML_DATA_PATH:-/workspace/data}"
export WANDB_PROJECT="${WANDB_PROJECT:-mlops}"
export WANDB_DIR="${WANDB_DIR:-/workspace/wandb_runs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/workspace/.wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/workspace/.wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/workspace/.wandb-cache}"

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

autoware-ml train \
  --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
  logger=wandb \
  model.init_checkpoint_path="${RTDETRV4_INIT_CKPT:-/workspace/weights/rtv4_hgnetv2_s_coco.pth}" \
  trainer.max_epochs="${RTDETRV4_MAX_EPOCHS:-1}" \
  trainer.log_every_n_steps="${RTDETRV4_LOG_EVERY_N_STEPS:-5}" \
  datamodule.max_train_samples="${RTDETRV4_MAX_TRAIN_SAMPLES:-128}" \
  datamodule.max_val_samples="${RTDETRV4_MAX_VAL_SAMPLES:-32}" \
  datamodule.train_dataloader_cfg.batch_size="${RTDETRV4_TRAIN_BATCH_SIZE:-8}" \
  datamodule.val_dataloader_cfg.batch_size="${RTDETRV4_VAL_BATCH_SIZE:-8}" \
  datamodule.train_dataloader_cfg.num_workers="${RTDETRV4_NUM_WORKERS:-2}" \
  datamodule.train_dataloader_cfg.persistent_workers=true \
  datamodule.val_dataloader_cfg.num_workers="${RTDETRV4_NUM_WORKERS:-2}" \
  datamodule.val_dataloader_cfg.persistent_workers=true \
  "${@}"
