"""
WandB Results Analysis and Visualization for Paper

This script fetches results from WandB and generates the specific tables and figures
required for the paper "Neural Circuit Policies and Liquid Time Constants for Continual Learning".

It requires the `wandb` package and a logged-in user.

Usage:
    python analysis/wandb_analysis.py --entity <your-entity> --project <your-project>
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path
from typing import Dict, List, Optional

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2.5

def fetch_runs(entity: str, project: str) -> pd.DataFrame:
    """
    Fetch all runs from WandB project and return as a DataFrame.
    """
    print(f"Fetching runs from {entity}/{project}...")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    data = []
    for run in runs:
        # Extract config
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        
        # Extract summary metrics
        summary = run.summary._json_dict
        
        # Combine
        entry = {
            'run_id': run.id,
            'name': run.name,
            'state': run.state,
            **config,
            **summary
        }
        data.append(entry)
        
    df = pd.DataFrame(data)
    print(f"Fetched {len(df)} runs.")
    return df

def generate_table_1_main_results(df: pd.DataFrame, output_dir: str):
    """
    Generate Table 1: Main comparative results (Average Accuracy).
    Compares: MLP, CfC, LTC, Random Sparse (with SGD and ER).
    """
    print("Generating Table 1...")
    
    # Filter relevant columns
    cols = ['model', 'dataset', 'buffer_size', 'acc_mean', 'bwt_mean', 'forgetting']
    # Ensure columns exist
    cols = [c for c in cols if c in df.columns]
    
    if 'acc_mean' not in df.columns:
        print("Warning: 'acc_mean' not found in data. Skipping Table 1.")
        return

    # Group by Model, Dataset, Buffer Size
    # We want to show: Split-MNIST, Split-CIFAR-10, TEP
    # Models: mnistmlp, mnistcfc, mnistltc, etc.
    
    # Map model names to paper names
    model_map = {
        'mnistmlp': 'MLP',
        'mnistcfc': 'NCP-CfC',
        'mnistltc': 'NCP-LTC',
        'mnist_random_sparse': 'CfC (Random)',
        'resnet18': 'ResNet-18',
        'cnn_cfc': 'ResNet-CfC',
        'tepcfc': 'NCP-CfC',
        'tepltc': 'NCP-LTC',
        'tep_lstm': 'LSTM'
    }
    
    df['Paper_Model'] = df['model'].map(model_map).fillna(df['model'])
    
    # Calculate mean and std over seeds
    group_cols = ['dataset', 'Paper_Model', 'buffer_size']
    # Handle missing buffer_size (e.g. for SGD)
    df['buffer_size'] = df['buffer_size'].fillna(0)
    
    summary = df.groupby(group_cols)[['acc_mean', 'bwt_mean']].agg(['mean', 'std']).reset_index()
    
    # Format for LaTeX
    summary['Accuracy'] = summary.apply(lambda x: f"{x[('acc_mean', 'mean')]:.2f} ± {x[('acc_mean', 'std')]:.2f}", axis=1)
    summary['BWT'] = summary.apply(lambda x: f"{x[('bwt_mean', 'mean')]:.2f} ± {x[('bwt_mean', 'std')]:.2f}", axis=1)
    
    # Save CSV
    summary.to_csv(f"{output_dir}/table_1_raw.csv")
    
    # Generate LaTeX table
    latex_table = summary[['dataset', 'Paper_Model', 'buffer_size', 'Accuracy', 'BWT']].to_latex(index=False)
    with open(f"{output_dir}/table_1.tex", 'w') as f:
        f.write(latex_table)
    print(f"Saved Table 1 to {output_dir}/table_1.tex")

def plot_accuracy_over_tasks(df: pd.DataFrame, output_dir: str):
    """
    Generate Figure: Accuracy over tasks (Learning Curve).
    """
    print("Generating Accuracy Plots...")
    
    # We need history for this, which is expensive to fetch for all runs.
    # We'll assume 'acc_mean' is the final accuracy.
    # Ideally, we want the accuracy after each task.
    # In Mammoth, this is usually logged as 'accuracy_x_y' or we can use the 'acc_mean' history.
    
    # For now, let's plot the final average accuracy bar chart as a proxy if history is missing
    # Or better, let's try to fetch history for a subset of best runs.
    
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        subset = df[df['dataset'] == dataset]
        if subset.empty: continue
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=subset, x='Paper_Model', y='acc_mean', hue='buffer_size')
        plt.title(f"Average Accuracy on {dataset}")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_{dataset}.pdf")
        plt.close()

def plot_ablation_wiring(df: pd.DataFrame, output_dir: str):
    """
    Generate Figure 1: Ablation 1 - Wiring Structure.
    Compares: AutoNCP vs Random Sparse vs Dense.
    """
    print("Generating Figure 1 (Wiring Ablation)...")
    
    # Filter for MNIST or TEP where we have these variants
    # Models: mnistcfc (AutoNCP), mnist_random_sparse, mnistmlp (Dense - approx)
    
    target_models = ['mnistcfc', 'mnist_random_sparse', 'mnistmlp']
    subset = df[df['model'].isin(target_models) & (df['dataset'] == 'seq-mnist')]
    
    if subset.empty:
        print("No data for Wiring Ablation.")
        return
        
    # Map to readable names
    name_map = {
        'mnistcfc': 'AutoNCP',
        'mnist_random_sparse': 'Random Sparse',
        'mnistmlp': 'Dense (MLP)'
    }
    subset['Wiring'] = subset['model'].map(name_map)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=subset, x='Wiring', y='acc_mean', palette='viridis')
    plt.title("Impact of Wiring Topology on Split-MNIST")
    plt.ylabel("Average Accuracy (%)")
    plt.xlabel("Wiring Strategy")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure_1_wiring_ablation.pdf")
    plt.close()

def plot_ablation_dynamics(df: pd.DataFrame, output_dir: str):
    """
    Generate Figure 2: Ablation 2 - Temporal Dynamics.
    Compares: CfC vs LTC vs LSTM.
    """
    print("Generating Figure 2 (Dynamics Ablation)...")
    
    target_models = ['mnistcfc', 'mnistltc', 'mnistlstm'] # Assuming mnistlstm exists
    subset = df[df['model'].isin(target_models) & (df['dataset'] == 'seq-mnist')]
    
    if subset.empty:
        # Try TEP
        target_models = ['tepcfc', 'tepltc', 'tep_lstm']
        subset = df[df['model'].isin(target_models) & (df['dataset'] == 'tep')]
    
    if subset.empty:
        print("No data for Dynamics Ablation.")
        return
        
    name_map = {
        'mnistcfc': 'CfC', 'tepcfc': 'CfC',
        'mnistltc': 'LTC', 'tepltc': 'LTC',
        'mnistlstm': 'LSTM', 'tep_lstm': 'LSTM'
    }
    subset['Dynamics'] = subset['model'].map(name_map)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=subset, x='Dynamics', y='acc_mean', palette='magma')
    plt.title("Impact of Temporal Dynamics")
    plt.ylabel("Average Accuracy (%)")
    plt.xlabel("Cell Type")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure_2_dynamics_ablation.pdf")
    plt.close()

def plot_tau_distribution(df: pd.DataFrame, output_dir: str):
    """
    Generate Tau Distribution Plot.
    Requires 'tau_mean' and 'tau_std' or raw tau values to be logged.
    """
    print("Generating Tau Distribution Plot...")
    # This is tricky without raw data. We'll check if there's a summary metric for bimodality.
    
    if 'tau_bimodality_coeff' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='Paper_Model', y='tau_bimodality_coeff')
        plt.axhline(y=0.555, color='r', linestyle='--', label='Bimodality Threshold')
        plt.title("Tau Distribution Bimodality Coefficient")
        plt.ylabel("BC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tau_bimodality.pdf")
        plt.close()
    else:
        print("Metric 'tau_bimodality_coeff' not found. Skipping Tau Plot.")

def generate_compute_cost_table(df: pd.DataFrame, output_dir: str):
    """
    Generate Table: Compute Cost (Runtime and Parameters).
    """
    print("Generating Compute Cost Table...")
    
    if '_runtime' not in df.columns:
        print("Warning: '_runtime' not found. Skipping Compute Cost Table.")
        return
        
    # Group by Model
    # We want to see if CfC is slower than MLP/LSTM
    
    # Map model names
    model_map = {
        'mnistmlp': 'MLP',
        'mnistcfc': 'NCP-CfC',
        'mnistltc': 'NCP-LTC',
        'mnist_random_sparse': 'CfC (Random)',
        'tepcfc': 'NCP-CfC',
        'tepltc': 'NCP-LTC',
        'tep_lstm': 'LSTM'
    }
    
    df['Paper_Model'] = df['model'].map(model_map).fillna(df['model'])
    
    # Calculate mean runtime
    summary = df.groupby(['dataset', 'Paper_Model'])['_runtime'].agg(['mean', 'std']).reset_index()
    summary['Runtime (s)'] = summary.apply(lambda x: f"{x['mean']:.0f} ± {x['std']:.0f}", axis=1)
    
    # Save
    latex_table = summary[['dataset', 'Paper_Model', 'Runtime (s)']].to_latex(index=False)
    with open(f"{output_dir}/table_compute_cost.tex", 'w') as f:
        f.write(latex_table)
    print(f"Saved Compute Cost Table to {output_dir}/table_compute_cost.tex")

def main():
    parser = argparse.ArgumentParser(description='Generate Paper Figures and Tables')
    parser.add_argument('--entity', type=str, required=True, help='WandB Entity')
    parser.add_argument('--project', type=str, required=True, help='WandB Project')
    parser.add_argument('--output_dir', type=str, default='analysis/results', help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Fetch Data
    df = fetch_runs(args.entity, args.project)
    
    if df.empty:
        print("No runs found. Exiting.")
        return
        
    # 2. Generate Table 1
    generate_table_1_main_results(df, args.output_dir)
    
    # 3. Generate Compute Cost Table
    generate_compute_cost_table(df, args.output_dir)
    
    # 4. Generate Figures
    plot_accuracy_over_tasks(df, args.output_dir)
    plot_ablation_wiring(df, args.output_dir)
    plot_ablation_dynamics(df, args.output_dir)
    plot_tau_distribution(df, args.output_dir)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
