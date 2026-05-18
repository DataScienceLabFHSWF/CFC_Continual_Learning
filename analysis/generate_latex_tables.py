import pandas as pd
import numpy as np
import os
import re

INPUT_CSV = "analysis/results/local_benchmark_summary.csv"
OUTPUT_DIR = "LTC_CFC_ContinualLearning/tables"

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_model_buffer(model_str):
    # Extracts method and buffer size from strings like 'er200', 'derpp1000', 'sgd', 'joint'
    match = re.match(r"([a-zA-Z]+)(\d+)?", model_str)
    if match:
        method = match.group(1).upper()
        buffer_size = match.group(2) if match.group(2) else "-"
        
        # Normalize method names
        if method == "DERPP": method = "DER++"
        if method == "ERACE": method = "ER-ACE"
        
        return method, buffer_size
    return model_str, "-"

def format_value(row):
    mean = row['mean']
    std = row['std']
    if pd.isna(mean):
        return "-"
    return f"${mean:.1f} \\pm {std:.1f}$"

def generate_comparison_table(df, dataset, backbone_baseline, backbone_ours, caption, label):
    # Filter for dataset
    subset = df[df['dataset'] == dataset].copy()
    
    if subset.empty:
        print(f"No data for dataset {dataset}")
        return

    # Parse method and buffer then create side-by-side rows for selected backbones.
    subset[['Method', 'Buffer']] = subset['model'].apply(lambda x: pd.Series(parse_model_buffer(x)))

    available_backbones = subset['backbone'].unique()
    if backbone_baseline not in available_backbones or backbone_ours not in available_backbones:
        print(f"Missing backbones for {dataset}. Found: {available_backbones}")
        return

    base = subset[subset['backbone'] == backbone_baseline][['Method', 'Buffer', 'mean', 'std']].rename(
        columns={'mean': 'base_mean', 'std': 'base_std'}
    )
    ours = subset[subset['backbone'] == backbone_ours][['Method', 'Buffer', 'mean', 'std']].rename(
        columns={'mean': 'our_mean', 'std': 'our_std'}
    )
    merged = pd.merge(base, ours, on=['Method', 'Buffer'], how='outer')

    # Create formatted rows
    latex_rows = []

    # Define order of methods
    method_order = ["SGD", "ER", "DER++", "ER-ACE", "JOINT"]

    merged['Method_Rank'] = merged['Method'].apply(lambda x: method_order.index(x) if x in method_order else 99)
    merged['Buffer_Rank'] = merged['Buffer'].apply(lambda x: int(x) if str(x) != "-" else 9999)
    merged = merged.sort_values(['Method_Rank', 'Buffer_Rank', 'Method', 'Buffer'])

    for _, row in merged.iterrows():
        method = row['Method']
        buffer_val = row['Buffer']

        base_mean = row['base_mean']
        base_std = row['base_std']
        our_mean = row['our_mean']
        our_std = row['our_std']

        base_str = f"${base_mean:.1f} \\pm {base_std:.1f}$" if not pd.isna(base_mean) else "-"
        our_str = f"${our_mean:.1f} \\pm {our_std:.1f}$" if not pd.isna(our_mean) else "-"

        # Bold the winner when both values are available.
        if not pd.isna(base_mean) and not pd.isna(our_mean):
            if our_mean > base_mean:
                our_str = f"\\textbf{{{our_str}}}"
            elif base_mean > our_mean:
                base_str = f"\\textbf{{{base_str}}}"

        latex_rows.append(f"{method} & {buffer_val} & {base_str} & {our_str} \\\\")

    # Construct full LaTeX table
    latex_code = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        f"Method & Buffer & {backbone_baseline.upper()} (Baseline) & {backbone_ours.upper()} (Ours) \\\\",
        "\\midrule"
    ]
    latex_code.extend(latex_rows)
    latex_code.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run parse_local_logs.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    ensure_dir(f"{OUTPUT_DIR}/dummy")
    
    # 1. MNIST Table
    mnist_tex = generate_comparison_table(
        df, 'mnist', 'mlp', 'cfc', 
        "Sequential MNIST Results (5 Tasks). Comparison of MLP baseline vs. CfC backbone.",
        "tab:mnist_results"
    )
    if mnist_tex:
        with open(f"{OUTPUT_DIR}/mnist_results.tex", "w") as f:
            f.write(mnist_tex)
        print(f"Generated {OUTPUT_DIR}/mnist_results.tex")

    # 2. CIFAR Table
    cifar_tex = generate_comparison_table(
        df, 'cifar', 'resnet', 'cfc', 
        "Sequential CIFAR-10 Results (5 Tasks). Comparison of ResNet18 baseline vs. CfC backbone.",
        "tab:cifar_results"
    )
    if cifar_tex:
        with open(f"{OUTPUT_DIR}/cifar_results.tex", "w") as f:
            f.write(cifar_tex)
        print(f"Generated {OUTPUT_DIR}/cifar_results.tex")

if __name__ == "__main__":
    main()
