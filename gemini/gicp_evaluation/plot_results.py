import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from GICP evaluation CSV results.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Directory to save plot images.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load Data
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        return

    df = pd.read_csv(args.csv_file)
    
    # 2. Setup Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # 3. Aggregate Statistics across all frame_idx
    # Columns to process
    error_cols = ['error_x', 'error_y', 'error_z', 'error_roll', 'error_pitch', 'error_yaw']

    # Group by axis and noise value
    grouped = df.groupby(['noise_axis', 'noise_value'])
    
    # Calculate stats
    stats_df = grouped[error_cols].agg(['mean', 'min', 'max']).reset_index()
    
    # Flatten multi-index columns
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

    # 4. Process each Noise Axis
    unique_axes = stats_df['noise_axis'].unique()
    
    for axis in unique_axes:
        print(f"Generating plot for noise axis: {axis}...")
        axis_df = stats_df[stats_df['noise_axis'] == axis].sort_values(by='noise_value')
        
        # Create Figure with 2 vertically stacked subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"GICP Residual Error (Mean, Min/Max\nAxis: {axis}", fontsize=16)

        colors = ['red', 'green', 'blue']
        
        # --- Subplot 1: Translational Errors ---
        for i, dim in enumerate(['x', 'y', 'z']):
            col_base = f'error_{dim}'
            c = colors[i]
            # Area (Min/Max)
            ax1.fill_between(axis_df['noise_value'], axis_df[f'{col_base}_min'], axis_df[f'{col_base}_max'], color=c, alpha=0.1)
            # Average (Mean)
            ax1.plot(axis_df['noise_value'], axis_df[f'{col_base}_mean'], label=f'{dim.upper()} Mean', color=c, linewidth=2)
        
        ax1.set_ylabel("Error (meters)")
        ax1.set_title("Translational Residual Errors")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # --- Subplot 2: Rotational Errors ---
        for i, dim in enumerate(['roll', 'pitch', 'yaw']):
            col_base = f'error_{dim}'
            c = colors[i]
            # Area (Min/Max)
            ax2.fill_between(axis_df['noise_value'], axis_df[f'{col_base}_min'], axis_df[f'{col_base}_max'], color=c, alpha=0.1)
            # Average (Mean)
            ax2.plot(axis_df['noise_value'], axis_df[f'{col_base}_mean'], label=f'{dim.capitalize()} Mean', color=c, linewidth=2)
        
        ax2.set_xlabel("Injected Noise Magnitude")
        ax2.set_ylabel("Error (degrees)")
        ax2.set_title("Rotational Residual Errors")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save Plot
        output_path = os.path.join(args.output_dir, f"{axis}_summary_error.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved plot to: {output_path}")

    print("Visualization complete.")

if __name__ == "__main__":
    main()
