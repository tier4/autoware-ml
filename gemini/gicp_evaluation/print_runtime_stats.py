import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Print runtime statistics from a GICP evaluation CSV.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the input CSV file.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        return

    # Load the CSV
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'runtime_sec' not in df.columns:
        print("Error: 'runtime_sec' column not found in the CSV.")
        return

    # Calculate statistics
    avg_runtime = df['runtime_sec'].mean()
    min_runtime = df['runtime_sec'].min()
    max_runtime = df['runtime_sec'].max()
    std_runtime = df['runtime_sec'].std()
    total_samples = len(df)

    # Print results
    print("-" * 40)
    print(f"Runtime Statistics for: {os.path.basename(args.csv_file)}")
    print(f"Total iterations: {total_samples}")
    print("-" * 40)
    print(f"Average Runtime: {avg_runtime:.6f} sec")
    print(f"Minimum Runtime: {min_runtime:.6f} sec")
    print(f"Maximum Runtime: {max_runtime:.6f} sec")
    print(f"Std Deviation:   {std_runtime:.6f} sec")
    print("-" * 40)

if __name__ == "__main__":
    main()
