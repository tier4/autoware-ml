import argparse
import os
import time
import pickle
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import small_gicp

def get_4x4_matrix(t, r_deg):
    """
    Creates a 4x4 transformation matrix from translation [x, y, z] 
    and XYZ Euler rotations in degrees.
    """
    R = Rotation.from_euler('xyz', r_deg, degrees=True).as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

def apply_transformation(points, R, T):
    """
    Applies rotation and translation to Nx3 points.
    """
    return points @ R.T + T

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate small_gicp robustness against LiDAR miscalibration noise.")
    parser.add_argument("--pkl_file", type=str, required=True, help="Path to pickle file containing metadata.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory where binary data is stored.")
    parser.add_argument("--source_lidar", type=str, default=None, help="Name of the source LiDAR (e.g. LIDAR_TOP). If omitted, all other points are used.")
    parser.add_argument("--target_lidar", type=str, required=True, help="Name of the target LiDAR (e.g. LIDAR_FRONT).")
    parser.add_argument("--frame_idx", type=int, default=None, help="Index of the frame to use. If omitted, all frames are used.")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames to process when frame_idx is None.")
    parser.add_argument("--apply_transforms", action="store_true", default=False, help="Apply LiDAR-to-Ego transforms before alignment.")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Path to save the results CSV.")
    return parser.parse_args()

def extract_points(selected_frame_data, dataset_root, src1_name, src2_name):
    """
    Extracts raw points from binary files using the same logic as lidar_visualizer_gui.py
    """
    lidar_path = selected_frame_data['lidar_points']['lidar_path']
    abs_bin_path = os.path.join(dataset_root, lidar_path)
    abs_json_path = abs_bin_path.replace("LIDAR_CONCAT", "LIDAR_CONCAT_INFO").replace(".pcd.bin", ".json")

    if not os.path.exists(abs_bin_path) or not os.path.exists(abs_json_path):
        raise FileNotFoundError(f"Missing BIN or JSON file.\nLooked for:\n{abs_bin_path}\n{abs_json_path}")

    sources_dict = selected_frame_data['lidar_sources']
    
    # Metadata for Ego-frame transforms
    if src1_name is not None:
        ext1 = {'R': np.array(sources_dict[src1_name]['rotation']), 'T': np.array(sources_dict[src1_name]['translation'])}
        token1 = sources_dict[src1_name]['sensor_token']
    else:
        ext1 = {'R': np.eye(3), 'T': np.zeros(3)}
        token1 = None

    ext2 = {'R': np.array(sources_dict[src2_name]['rotation']), 'T': np.array(sources_dict[src2_name]['translation'])}
    token2 = sources_dict[src2_name]['sensor_token']

    with open(abs_json_path, 'r') as f:
        info = json.load(f)

    idx_len_1 = idx_len_2 = None
    for s in info['sources']:
        if token1 is not None and s['sensor_token'] == token1: 
            idx_len_1 = (s['idx_begin'], s['length'])
        if s['sensor_token'] == token2: 
            idx_len_2 = (s['idx_begin'], s['length'])

    if idx_len_2 is None:
        raise ValueError(f"Could not find target sensor token for {src2_name} in JSON info.")
    if src1_name is not None and idx_len_1 is None:
        raise ValueError(f"Could not find source sensor token for {src1_name} in JSON info.")

    raw_data = np.fromfile(abs_bin_path, dtype=np.float32)
    total_pts = sum(s['length'] for s in info['sources'])
    num_features = len(raw_data) // total_pts
    points_nx3 = raw_data.reshape(-1, num_features)[:, :3]

    if src1_name is not None:
        pts1 = points_nx3[idx_len_1[0] : idx_len_1[0] + idx_len_1[1]]
    else:
        # Source is everything EXCEPT target
        mask = np.ones(len(points_nx3), dtype=bool)
        mask[idx_len_2[0] : idx_len_2[0] + idx_len_2[1]] = False
        pts1 = points_nx3[mask]

    pts2 = points_nx3[idx_len_2[0] : idx_len_2[0] + idx_len_2[1]]
    
    return pts1, pts2, ext1, ext2

def main():
    args = parse_args()

    # 1. Data Loading
    if not os.path.exists(args.pkl_file):
        print(f"Error: Pickle file {args.pkl_file} not found.")
        return

    with open(args.pkl_file, 'rb') as f:
        data_all = pickle.load(f)
    
    data_list = data_all.get('data_list', [])
    if not data_list:
        print("Error: No 'data_list' found in pickle.")
        return
    
    if args.frame_idx is not None:
        if args.frame_idx >= len(data_list):
            print(f"Error: frame_idx {args.frame_idx} out of range (max {len(data_list)-1}).")
            return
        frame_indices = [args.frame_idx]
    else:
        frame_indices = list(range(min(len(data_list), args.max_frames)))

    results = []

    # 2. Noise Generation (Axis Sweeping)
    step_m = 0.1
    from_m = -0.3
    to_m = 0.3 + step_m
    step_deg = 0.1
    from_deg = -0.8
    to_deg = 0.8 + step_deg

    sweep_configs = [
        ('X', np.arange(from_m, to_m, step_m), 'trans'),
        ('Y', np.arange(from_m, to_m, step_m), 'trans'),
        ('Z', np.arange(from_m, to_m, step_m), 'trans'),
        ('Roll', np.arange(from_deg, to_deg, step_deg), 'rot'),
        ('Pitch', np.arange(from_deg, to_deg, step_deg), 'rot'),
        ('Yaw', np.arange(from_deg, to_deg, step_deg), 'rot')
    ]

    iters_per_frame = sum(len(c[1]) for c in sweep_configs)
    total_iterations = len(frame_indices) * iters_per_frame
    current_total_iter = 0

    print(f"Starting evaluation of small_gicp robustness on {len(frame_indices)} frame(s) ({total_iterations} total iterations)...")

    for f_idx in frame_indices:
        selected_frame = data_list[f_idx]
        try:
            # Extract raw points
            raw_source_pts, raw_target_pts, ext_source, ext_target = extract_points(
                selected_frame, args.dataset_root, args.source_lidar, args.target_lidar
            )
        except Exception as e:
            print(f"Error during point extraction for frame {f_idx}: {e}")
            continue

        # Prepare Baseline (Common Frame)
        if args.apply_transforms:
            # Transform both to Ego frame for baseline alignment
            source_pts = apply_transformation(raw_source_pts, ext_source['R'], ext_source['T'])
            target_pts = apply_transformation(raw_target_pts, ext_target['R'], ext_target['T'])
        else:
            source_pts = raw_source_pts.copy()
            target_pts = raw_target_pts.copy()

        for axis, values, noise_type in sweep_configs:
            for val in values:
                current_total_iter += 1
                noise_trans = [0.0, 0.0, 0.0]
                noise_rot_deg = [0.0, 0.0, 0.0]
                
                if noise_type == 'trans':
                    noise_trans[['X', 'Y', 'Z'].index(axis)] = val
                else:
                    noise_rot_deg[['Roll', 'Pitch', 'Yaw'].index(axis)] = val

                # 3. Transformation Application (Injected Noise)
                T_noise = get_4x4_matrix(noise_trans, noise_rot_deg)
                # Apply T_noise to source_pts (already in Ego frame if apply_transforms is True)
                noisy_source_pts = (source_pts @ T_noise[:3, :3].T) + T_noise[:3, 3]

                # 4. GICP Execution
                start_time = time.perf_counter()
                result = small_gicp.align(target_pts, noisy_source_pts)
                runtime = time.perf_counter() - start_time

                # 5. Metrics Calculation
                T_est = result.T_target_source
                t_est = T_est[:3, 3]
                r_est_deg = Rotation.from_matrix(T_est[:3, :3]).as_euler('xyz', degrees=True)
                
                # Mahalanobis Distance calculation
                r_est_vec = Rotation.from_matrix(T_est[:3, :3]).as_rotvec()
                drift_vector = np.concatenate([t_est, r_est_vec])
                H = result.H
                H_stable = H + np.eye(6) * 1e-6
                mahalanobis_dist = drift_vector.T @ H @ drift_vector
                
                eigenvalues = np.linalg.eigvalsh(H_stable)
                min_eigenvalue = np.min(eigenvalues)

                # Identification Error
                # We expect T_est to be T_noise^-1
                T_identified = np.linalg.inv(T_est)
                t_id = T_identified[:3, 3]
                r_id_deg = Rotation.from_matrix(T_identified[:3, :3]).as_euler('xyz', degrees=True)

                res_dict = {
                    'frame_idx': f_idx,
                    'noise_axis': axis,
                    'noise_value': val,
                    'runtime_sec': runtime,
                    'gicp_x': t_est[0],
                    'gicp_y': t_est[1],
                    'gicp_z': t_est[2],
                    'gicp_roll_deg': r_est_deg[0],
                    'gicp_pitch_deg': r_est_deg[1],
                    'gicp_yaw_deg': r_est_deg[2],
                    'error_x': t_id[0],
                    'error_y': t_id[1],
                    'error_z': t_id[2],
                    'error_roll': r_id_deg[0],
                    'error_pitch': r_id_deg[1],
                    'error_yaw': r_id_deg[2],
                    'mahalanobis_distance': mahalanobis_dist,
                    'min_eigenvalue': min_eigenvalue
                }
                results.append(res_dict)

                if current_total_iter % 50 == 0 or current_total_iter == total_iterations:
                    print(f"Progress: {current_total_iter}/{total_iterations} iterations complete.")

    # 6. Output to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Evaluation finished. Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
