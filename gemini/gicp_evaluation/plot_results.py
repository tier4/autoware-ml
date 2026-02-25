import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

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

    # 3. Process each Noise Axis
    unique_axes = df['noise_axis'].unique()
    
    for axis in unique_axes:
        print(f"Generating plot for noise axis: {axis}...")
        axis_df = df[df['noise_axis'] == axis].sort_values(by='noise_value')

        # Create Figure with 2 vertically stacked subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"GICP Error Response to Injected Noise on Axis: {axis}", fontsize=16)

        # --- Subplot 1: Translational Errors ---
        ax1.plot(axis_df['noise_value'], axis_df['error_x'], label='Error X', marker='o')
        ax1.plot(axis_df['noise_value'], axis_df['error_y'], label='Error Y', marker='s')
        ax1.plot(axis_df['noise_value'], axis_df['error_z'], label='Error Z', marker='^')
        
        ax1.set_ylabel("Error (meters)")
        ax1.set_title("Translational Errors")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # --- Subplot 2: Rotational Errors ---
        ax2.plot(axis_df['noise_value'], axis_df['error_roll'], label='Error Roll', marker='o')
        ax2.plot(axis_df['noise_value'], axis_df['error_pitch'], label='Error Pitch', marker='s')
        ax2.plot(axis_df['noise_value'], axis_df['error_yaw'], label='Error Yaw', marker='^')
        
        ax2.set_xlabel("Injected Noise Magnitude")
        ax2.set_ylabel("Error (degrees)")
        ax2.set_title("Rotational Errors")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        
        # Save Plot
        output_path = os.path.join(args.output_dir, f"{axis}_error_response.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved plot to: {output_path}")

    print("Visualization complete.")

if __name__ == "__main__":
    main()
