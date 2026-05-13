 From host:

  cd /home/jacoblambert/rtdetrv4-dev/autoware-ml
  pip install wandb==0.25.1
  
  ./docker/container.sh --run --headless --data-path /home/jacoblambert/public_data

  Enter the container:

  docker exec -it autoware-ml-jacoblambert /workspace/docker/entrypoint.sh bash

  Start fresh training:

  autoware-ml train \
    --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
    model.init_checkpoint_path=/workspace/weights/rtv4_hgnetv2_s_coco.pth

  Resume from a Lightning checkpoint:

  autoware-ml train \
    --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
    model.init_checkpoint_path=null \
    +checkpoint=/workspace/mlruns/detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer/<date>/<time>/checkpoints/last.ckpt

  Safer resume if DataLoader workers were still causing instability:

  autoware-ml train \
    --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
    model.init_checkpoint_path=null \
    datamodule.train_dataloader_cfg.num_workers=0 \
    datamodule.train_dataloader_cfg.persistent_workers=false \
    datamodule.val_dataloader_cfg.num_workers=0 \
    datamodule.val_dataloader_cfg.persistent_workers=false \
    +checkpoint=/workspace/mlruns/detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer/<date>/<time>/checkpoints/last.ckpt

  If you want tmux session mode inside the container:

  autoware-ml session start --name rtdetrv4-mapillary-transfer --cwd /workspace -- \
    train \
    --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
    model.init_checkpoint_path=/workspace/weights/rtv4_hgnetv2_s_coco.pth

  Attach:

  autoware-ml session attach --name rtdetrv4-mapillary-transfer

// 3 runs with slightly different specs
1. Baseline transfer

  autoware-ml train \
    --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
    model.init_checkpoint_path=/workspace/weights/rtv4_hgnetv2_s_coco.pth \
    trainer.max_epochs=3 \
    datamodule.train_dataloader_cfg.batch_size=32 \
    datamodule.val_dataloader_cfg.batch_size=16 \
    datamodule.train_dataloader_cfg.num_workers=0 \
    datamodule.train_dataloader_cfg.persistent_workers=false \
    datamodule.val_dataloader_cfg.num_workers=0 \
    datamodule.val_dataloader_cfg.persistent_workers=false

  2. Lower LR

  autoware-ml train \
    --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
    model.init_checkpoint_path=/workspace/weights/rtv4_hgnetv2_s_coco.pth \
    trainer.max_epochs=3 \
    datamodule.train_dataloader_cfg.batch_size=32 \
    datamodule.val_dataloader_cfg.batch_size=16 \
    datamodule.train_dataloader_cfg.num_workers=0 \
    datamodule.train_dataloader_cfg.persistent_workers=false \
    datamodule.val_dataloader_cfg.num_workers=0 \
    datamodule.val_dataloader_cfg.persistent_workers=false \
    model.optimizer.lr=0.0002 \
    model.optimizer_group_overrides.backbone.lr=0.0002 \
    model.optimizer_group_overrides.backbone_norm.lr=0.0002

  3. Reduced strong augmentation

  autoware-ml train \
    --config-name detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer \
    model.init_checkpoint_path=/workspace/weights/rtv4_hgnetv2_s_coco.pth \
    trainer.max_epochs=3 \
    datamodule.train_dataloader_cfg.batch_size=32 \
    datamodule.val_dataloader_cfg.batch_size=16 \
    datamodule.train_dataloader_cfg.num_workers=0 \
    datamodule.train_dataloader_cfg.persistent_workers=false \
    datamodule.val_dataloader_cfg.num_workers=0 \
    datamodule.val_dataloader_cfg.persistent_workers=false \
    mosaic_prob=0.0 \
    mixup_prob=0.0 \
    random_iou_crop_prob=0.0 \
    random_zoom_out_prob=0.0 \
    photometric_distort_prob=0.0