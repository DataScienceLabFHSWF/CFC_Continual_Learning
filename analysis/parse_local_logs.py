import glob
import os
import re
import numpy as np
import pandas as pd

LOG_DIR = "paper_results/logs"

def parse_log_file(filepath):
    filename = os.path.basename(filepath)
    # Try to parse filename
    # Format: {dataset}_{backbone}_{model_variant}_seed{seed}.log
    # But sometimes backbone is part of model variant or vice versa.
    
    parts = filename.replace('.log', '').split('_')
    
    dataset = parts[0]
    seed_part = parts[-1]
    seed = int(seed_part.replace('seed', ''))
    
    # Heuristic for backbone and model
    # usually parts[1] is backbone, parts[2] is model+buffer
    backbone = parts[1]
    model_variant = "_".join(parts[2:-1])
    
    # Read file for metrics
    mean_acc = None
    with open(filepath, 'r') as f:
        for line in f:
            if "RESULT_class_mean_accs" in line:
                try:
                    # Format: wandb: RESULT_class_mean_accs 82.45077
                    val = float(line.split()[-1])
                    mean_acc = val
                except:
                    pass
    
    return {
        'dataset': dataset,
        'backbone': backbone,
        'model': model_variant,
        'seed': seed,
        'accuracy': mean_acc,
        'file': filename
    }

def main():
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    results = []
    
    print(f"Found {len(log_files)} log files.")
    
    for log_file in log_files:
        res = parse_log_file(log_file)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    # Filter out runs without results
    completed = df[df['accuracy'].notna()]
    failed = df[df['accuracy'].isna()]
    
    print(f"\nCompleted Runs: {len(completed)}")
    print(f"Failed/Incomplete Runs: {len(failed)}")
    
    if len(completed) > 0:
        print("\n=== Results Summary (Mean Accuracy ± Std) ===")
        summary = completed.groupby(['dataset', 'backbone', 'model'])['accuracy'].agg(['mean', 'std', 'count'])
        print(summary.to_string())
        
        # Save to CSV
        summary.to_csv("analysis/results/local_benchmark_summary.csv")
        print("\nSummary saved to analysis/results/local_benchmark_summary.csv")
        
    if len(failed) > 0:
        print("\n=== Failed/Incomplete Runs (First 10) ===")
        print(failed[['dataset', 'backbone', 'model', 'seed']].head(10).to_string())

if __name__ == "__main__":
    main()
