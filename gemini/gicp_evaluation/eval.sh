#!/bin/bash
# LIDAR_LIST=(LIDAR_LEFT_UPPER LIDAR_LEFT_LOWER LIDAR_RIGHT_LOWER LIDAR_RIGHT_UPPER LIDAR_FRONT_LOWER LIDAR_FRONT_UPPER LIDAR_REAR_UPPER LIDAR_REAR_LOWER)
LIDAR_LIST=(LIDAR_FRONT_LOWER LIDAR_FRONT_UPPER)

for TARGET_LIDAR in "${LIDAR_LIST[@]}"; do
    echo "Evaluating robustness for $TARGET_LIDAR..."
    OUT_CSV="results/train_${TARGET_LIDAR}.csv"
    
    python3 evaluate_gicp.py \
        --pkl_file /mnt/qnapdata/t4dataset/calibration_info_lidarseg/t4dataset_gen2_lidarseg_infos_train.pkl \
        --target_lidar $TARGET_LIDAR \
        --output_csv $OUT_CSV \
        --dataset_root /mnt/qnapdata/t4dataset \
        --max_frames 50

    python3 plot_results.py --csv_file $OUT_CSV
done
