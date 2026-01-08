import os
import re
import glob
import json
import pandas as pd
import numpy as np

def parse_log_file(file_path):
    """
    Parses a Mammoth log file (or output.txt) to extract:
    - Dataset
    - Model
    - Backbone
    - Seed
    - Final Class-IL Accuracy
    - Final Task-IL Accuracy
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # default values
    meta = {
        'dataset': 'unknown',
        'model': 'unknown',
        'backbone': 'unknown',
        'seed': 'unknown'
    }

    # 1. Extract Metadata from Command or Config print
    # Command: ... --dataset seq-cifar10 --model er --backbone resnet18 ...
    cmd_match = re.search(r'Command:.*python.*main\.py\s+(.*)', content)
    if cmd_match:
        args_str = cmd_match.group(1)
        
        d_match = re.search(r'--dataset\s+([\w-]+)', args_str)
        if d_match: meta['dataset'] = d_match.group(1)
        
        m_match = re.search(r'--model\s+([\w-]+)', args_str)
        if m_match: meta['model'] = m_match.group(1)
        
        b_match = re.search(r'--backbone\s+([\w-]+)', args_str)
        if b_match: meta['backbone'] = b_match.group(1)
        
        s_match = re.search(r'--seed\s+(\d+)', args_str)
        if s_match: meta['seed'] = int(s_match.group(1))

    # Backup metadata extraction (from filename) if extraction failed
    # Filename format: {model}_{backbone}_{dataset}_seed{seed}.log
    if meta['dataset'] == 'unknown':
        fname = os.path.basename(file_path)
        parts = fname.replace('.log', '').replace('_output.txt', '').split('_')
        # This is heuristics, might vary
        if 'seed' in parts[-1]:
            # Try to reverse engineer
            pass

    # 2. Extract Results
    # Pattern: "Accuracy for X task(s): [Class-IL]: 82.88 % [Task-IL]: 93.55 %"
    # We want the LAST occurrence of this pattern (Final accuracy)
    
    # metrics
    class_il = None
    task_il = None
    
    matches = re.findall(r'Accuracy for \d+ task\(s\):\s+\[Class-IL\]:\s+([\d.]+)\s+%\s+\[Task-IL\]:\s+([\d.]+)\s+%', content)
    if matches:
        last_match = matches[-1]
        class_il = float(last_match[0])
        task_il = float(last_match[1])
        
    if class_il is None:
        return None  # Run did not finish or failed
        
    return {
        **meta,
        'class_il': class_il,
        'task_il': task_il,
        'path': file_path
    }

def main():
    root_dir = "benchmark_results"
    all_data = []

    print(f"Scanning {root_dir}...")
    
    # Find all potential log files
    files = glob.glob(f"{root_dir}/**/*.log", recursive=True) + glob.glob(f"{root_dir}/**/*_output.txt", recursive=True)
    
    print(f"Found {len(files)} log files.")

    for f in files:
        res = parse_log_file(f)
        if res:
            all_data.append(res)

    if not all_data:
        print("No valid results found.")
        return

    df = pd.DataFrame(all_data)
    
    # Grouping
    grouped = df.groupby(['dataset', 'backbone', 'model']).agg(
        n_seeds=('seed', 'count'),
        class_il_mean=('class_il', 'mean'),
        class_il_std=('class_il', 'std'),
        task_il_mean=('task_il', 'mean'),
        task_il_std=('task_il', 'std')
    ).reset_index()

    # Formatting
    print("\n" + "="*80)
    print("DETAILED BENCHMARK REPORT")
    print("="*80)

    # Clean display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.2f}'.format)

    sort_cols = ['dataset', 'model', 'backbone']
    print(grouped.sort_values(sort_cols))
    
    # Save to Markdown
    md_file = "DETAILED_BENCHMARK_REPORT.md"
    with open(md_file, "w") as f:
        f.write("# Detailed Benchmark Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        # Iterate by dataset for clearer tables
        for dataset in grouped['dataset'].unique():
            f.write(f"## Dataset: {dataset}\n\n")
            sub_df = grouped[grouped['dataset'] == dataset].sort_values(['model', 'backbone'])
            
            # Create nice markdown table
            f.write("| Model | Backbone | Samples | Class-IL (Mean ± Std) | Task-IL (Mean ± Std) |\n")
            f.write("|---|---|---|---|---|\n")
            
            for _, row in sub_df.iterrows():
                cil = f"{row['class_il_mean']:.2f}"
                if not pd.isna(row['class_il_std']):
                     cil += f" ± {row['class_il_std']:.2f}"
                
                til = f"{row['task_il_mean']:.2f}"
                if not pd.isna(row['task_il_std']):
                     til += f" ± {row['task_il_std']:.2f}"
                     
                f.write(f"| {row['model']} | {row['backbone']} | {row['n_seeds']} | {cil} | {til} |\n")
            f.write("\n")
            
    print(f"\nReport saved to {md_file}")

if __name__ == "__main__":
    main()
