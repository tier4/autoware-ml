import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from GICP evaluation CSV results.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the input CSV file.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load Data
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        return

    df = pd.read_csv(args.csv_file)
    
    # 2. Aggregate Statistics across all frame_idx
    error_cols = ['error_x', 'error_y', 'error_z', 'error_roll', 'error_pitch', 'error_yaw']

    # Group by axis and noise value
    grouped = df.groupby(['noise_axis', 'noise_value'])
    
    # Calculate stats
    stats_df = grouped[error_cols].agg(['mean', 'min', 'max']).reset_index()
    
    # Flatten multi-index columns
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

    # 3. Create Grid Plot (2x3)
    # top row: X,Y,Z, bottom row: roll, pitch, yaw
    axes_map = [
        ['X', 'Y', 'Z'],
        ['Roll', 'Pitch', 'Yaw']
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"GICP Residual Error Summary\nSource: {os.path.basename(args.csv_file)}", fontsize=16)

    colors = ['red', 'green', 'blue']
    error_dims_trans = ['x', 'y', 'z']
    error_dims_rot = ['roll', 'pitch', 'yaw']

    for row_idx in range(2):
        for col_idx in range(3):
            noise_axis = axes_map[row_idx][col_idx]
            ax = axes[row_idx][col_idx]
            
            axis_df = stats_df[stats_df['noise_axis'] == noise_axis].sort_values(by='noise_value')
            
            if axis_df.empty:
                ax.set_title(f"No data for {noise_axis}")
                continue

            # For top row (trans noise), we show translational errors
            # For bottom row (rot noise), we show rotational errors
            error_dims = error_dims_trans if row_idx == 0 else error_dims_rot
            unit = "meters" if row_idx == 0 else "degrees"
            
            for i, dim in enumerate(error_dims):
                col_base = f'error_{dim}'
                c = colors[i]
                label = dim.upper() if row_idx == 0 else dim.capitalize()
                
                # Area (Min/Max)
                ax.fill_between(axis_df['noise_value'], axis_df[f'{col_base}_min'], axis_df[f'{col_base}_max'], color=c, alpha=0.1)
                # Average (Mean)
                ax.plot(axis_df['noise_value'], axis_df[f'{col_base}_mean'], label=f'{label} Mean', color=c, linewidth=1.5)

            ax.set_title(f"Noise Sweep: {noise_axis}")
            ax.set_ylabel(f"Error ({unit})")
            ax.set_xlabel("Noise Magnitude")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 4. Save Output
    output_path = os.path.splitext(args.csv_file)[0] + ".png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved summary plot to: {output_path}")

if __name__ == "__main__":
    main()
