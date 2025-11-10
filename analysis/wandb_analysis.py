"""
WandB Results Analysis and Visualization

This script analyzes experimental results from WandB logs and creates visualizations
to understand the performance of CfC/NCP networks for continual learning.

Usage:
    python analysis/wandb_analysis.py --entity fneubuerger --project mammoth
"""

import argparse
import os
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
except ImportError:
    HAS_PLOTTING = False
    print("Note: matplotlib/seaborn not available. Plots will be skipped.")


def parse_local_wandb_runs(wandb_dir: str) -> List[Dict]:
    """
    Parse WandB runs from local directory structure.
    
    Args:
        wandb_dir: Path to wandb directory
        
    Returns:
        List of run configurations
    """
    runs = []
    wandb_path = Path(wandb_dir)
    
    for run_dir in wandb_path.glob("run-*"):
        config_file = run_dir / "files" / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Extract relevant info
            run_info = {
                'run_id': run_dir.name.split('-')[-1],
                'timestamp': run_dir.name.split('-')[1] + '_' + run_dir.name.split('-')[2],
                'config': {}
            }
            
            # Parse config
            for key, value in config.items():
                if isinstance(value, dict) and 'value' in value:
                    run_info['config'][key] = value['value']
            
            runs.append(run_info)
    
    return runs


def summarize_runs(runs: List[Dict]) -> None:
    """Print summary of all runs."""
    print("\n" + "="*80)
    print("WANDB RUNS SUMMARY")
    print("="*80)
    
    print(f"\nTotal runs found: {len(runs)}")
    
    # Group by dataset
    by_dataset = defaultdict(list)
    by_model = defaultdict(list)
    
    for run in runs:
        dataset = run['config'].get('dataset', 'unknown')
        model = run['config'].get('model', 'unknown')
        by_dataset[dataset].append(run)
        by_model[model].append(run)
    
    print("\n--- By Dataset ---")
    for dataset, dataset_runs in sorted(by_dataset.items()):
        print(f"  {dataset}: {len(dataset_runs)} runs")
    
    print("\n--- By Model ---")
    for model, model_runs in sorted(by_model.items()):
        print(f"  {model}: {len(model_runs)} runs")
    
    print("\n--- Run Details ---")
    for i, run in enumerate(runs, 1):
        config = run['config']
        print(f"\nRun {i} ({run['timestamp']}):")
        print(f"  ID: {run['run_id']}")
        print(f"  Dataset: {config.get('dataset', 'N/A')}")
        print(f"  Model: {config.get('model', 'N/A')}")
        print(f"  Epochs: {config.get('n_epochs', 'N/A')}")
        print(f"  Learning Rate: {config.get('lr', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        if 'buffer_size' in config:
            print(f"  Buffer Size: {config['buffer_size']}")


def create_experiment_matrix_plot(runs: List[Dict], save_path: str = None):
    """
    Create a matrix visualization of experiments run.
    """
    datasets = sorted(set(r['config'].get('dataset', 'unknown') for r in runs))
    models = sorted(set(r['config'].get('model', 'unknown') for r in runs))
    
    # Create matrix: rows=datasets, cols=models
    matrix = np.zeros((len(datasets), len(models)))
    
    for run in runs:
        dataset = run['config'].get('dataset', 'unknown')
        model = run['config'].get('model', 'unknown')
        if dataset in datasets and model in models:
            i = datasets.index(dataset)
            j = models.index(model)
            matrix[i, j] += 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(models)*1.5), max(6, len(datasets)*0.8)))
    
    sns.heatmap(matrix, annot=True, fmt='g', cmap='YlOrRd', 
                xticklabels=models, yticklabels=datasets,
                cbar_kws={'label': 'Number of Runs'}, ax=ax)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('Experiment Coverage Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved experiment matrix to {save_path}")
    
    return fig


def create_configuration_distribution_plots(runs: List[Dict], save_path: str = None):
    """
    Create plots showing distribution of hyperparameters.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Learning rates
    lrs = [r['config'].get('lr', None) for r in runs if r['config'].get('lr') is not None]
    if lrs:
        axes[0].hist(lrs, bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Learning Rate Distribution')
        axes[0].set_xscale('log')
    
    # Batch sizes
    batch_sizes = [r['config'].get('batch_size', None) for r in runs if r['config'].get('batch_size') is not None]
    if batch_sizes:
        unique_bs = sorted(set(batch_sizes))
        counts = [batch_sizes.count(bs) for bs in unique_bs]
        axes[1].bar(range(len(unique_bs)), counts, edgecolor='black', alpha=0.7)
        axes[1].set_xticks(range(len(unique_bs)))
        axes[1].set_xticklabels(unique_bs)
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Batch Size Distribution')
    
    # Epochs
    epochs = [r['config'].get('n_epochs', None) for r in runs if r['config'].get('n_epochs') is not None]
    if epochs:
        unique_ep = sorted(set(epochs))
        counts = [epochs.count(ep) for ep in unique_ep]
        axes[2].bar(range(len(unique_ep)), counts, edgecolor='black', alpha=0.7, color='coral')
        axes[2].set_xticks(range(len(unique_ep)))
        axes[2].set_xticklabels(unique_ep)
        axes[2].set_xlabel('Number of Epochs')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Training Epochs Distribution')
    
    # Buffer sizes (for replay methods)
    buffer_sizes = [r['config'].get('buffer_size', None) for r in runs if r['config'].get('buffer_size') is not None]
    if buffer_sizes:
        axes[3].hist(buffer_sizes, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[3].set_xlabel('Buffer Size')
        axes[3].set_ylabel('Count')
        axes[3].set_title('Buffer Size Distribution (Replay Methods)')
    
    # Models
    models = [r['config'].get('model', 'unknown') for r in runs]
    unique_models = sorted(set(models))
    counts = [models.count(m) for m in unique_models]
    axes[4].barh(range(len(unique_models)), counts, edgecolor='black', alpha=0.7, color='purple')
    axes[4].set_yticks(range(len(unique_models)))
    axes[4].set_yticklabels(unique_models)
    axes[4].set_xlabel('Count')
    axes[4].set_title('Model Distribution')
    
    # Datasets
    datasets = [r['config'].get('dataset', 'unknown') for r in runs]
    unique_datasets = sorted(set(datasets))
    counts = [datasets.count(d) for d in unique_datasets]
    axes[5].barh(range(len(unique_datasets)), counts, edgecolor='black', alpha=0.7, color='orange')
    axes[5].set_yticks(range(len(unique_datasets)))
    axes[5].set_yticklabels(unique_datasets)
    axes[5].set_xlabel('Count')
    axes[5].set_title('Dataset Distribution')
    
    plt.suptitle('Hyperparameter and Configuration Distributions', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved configuration distributions to {save_path}")
    
    return fig


def create_timeline_plot(runs: List[Dict], save_path: str = None):
    """
    Create timeline of experiments.
    """
    # Parse timestamps
    run_times = []
    for run in runs:
        timestamp_str = run['timestamp']
        # Format: YYYYMMDD_HHMMSS
        run_times.append(timestamp_str)
    
    run_times_sorted = sorted(run_times)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot each run as a point
    x = list(range(len(run_times_sorted)))
    y = [1] * len(run_times_sorted)
    
    ax.scatter(x, y, s=100, alpha=0.6)
    
    # Annotate with dates
    step = max(1, len(run_times_sorted) // 10)  # Show ~10 labels
    for i in range(0, len(run_times_sorted), step):
        ax.annotate(run_times_sorted[i][:8], (x[i], y[i]), 
                   rotation=45, ha='right', fontsize=8)
    
    ax.set_xlabel('Run Number', fontsize=12)
    ax.set_ylabel('')
    ax.set_title('Experiment Timeline', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timeline to {save_path}")
    
    return fig


def generate_report(runs: List[Dict], output_dir: str = "analysis/results"):
    """
    Generate comprehensive analysis report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING ANALYSIS REPORT")
    print("="*80)
    
    # Summary
    summarize_runs(runs)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    create_experiment_matrix_plot(runs, 
                                  save_path=f"{output_dir}/experiment_matrix.png")
    
    create_configuration_distribution_plots(runs, 
                                           save_path=f"{output_dir}/configuration_distributions.png")
    
    create_timeline_plot(runs, 
                        save_path=f"{output_dir}/experiment_timeline.png")
    
    # Generate text report
    report_path = f"{output_dir}/analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CONTINUAL LEARNING EXPERIMENTS ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Runs: {len(runs)}\n\n")
        
        # Group by dataset
        by_dataset = defaultdict(list)
        by_model = defaultdict(list)
        
        for run in runs:
            dataset = run['config'].get('dataset', 'unknown')
            model = run['config'].get('model', 'unknown')
            by_dataset[dataset].append(run)
            by_model[model].append(run)
        
        f.write("DATASETS TESTED:\n")
        f.write("-" * 40 + "\n")
        for dataset, dataset_runs in sorted(by_dataset.items()):
            f.write(f"  {dataset}: {len(dataset_runs)} runs\n")
        
        f.write("\nMODELS TESTED:\n")
        f.write("-" * 40 + "\n")
        for model, model_runs in sorted(by_model.items()):
            f.write(f"  {model}: {len(model_runs)} runs\n")
        
        f.write("\nKEY OBSERVATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Most experiments focused on:\n")
        f.write(f"   - Primary dataset: {max(by_dataset.items(), key=lambda x: len(x[1]))[0]}\n")
        f.write(f"   - Primary model: {max(by_model.items(), key=lambda x: len(x[1]))[0]}\n")
        
        f.write("\n2. Experiment coverage:\n")
        f.write(f"   - Unique datasets: {len(by_dataset)}\n")
        f.write(f"   - Unique models: {len(by_model)}\n")
        
        f.write("\n3. GAPS AND RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        
        # Check for CfC experiments
        has_cfc = any('cfc' in run['config'].get('model', '').lower() or 
                     'ncp' in run['config'].get('model', '').lower() 
                     for run in runs)
        
        if not has_cfc:
            f.write("   ⚠️  NO CfC/NCP EXPERIMENTS FOUND!\n")
            f.write("      This is the main focus of the project.\n")
            f.write("      Action: Run experiments with corrected CfC backbones.\n\n")
        
        # Check for statistical rigor
        config_keys = set()
        for run in runs:
            config_keys.update(run['config'].keys())
        
        if 'seed' not in config_keys or all(run['config'].get('seed') is None for run in runs):
            f.write("   ⚠️  NO RANDOM SEEDS DETECTED!\n")
            f.write("      Multiple seeds needed for statistical validity.\n")
            f.write("      Action: Run each experiment with seeds 0-9.\n\n")
        
        # Check for baselines
        baseline_models = {'sgd', 'er', 'ewc', 'lwf'}
        found_baselines = set(by_model.keys()) & baseline_models
        missing_baselines = baseline_models - found_baselines
        
        if missing_baselines:
            f.write(f"   ⚠️  MISSING BASELINE MODELS: {', '.join(missing_baselines)}\n")
            f.write("      Action: Run these baselines for fair comparison.\n\n")
        
        f.write("\n4. NEXT STEPS:\n")
        f.write("-" * 40 + "\n")
        f.write("   1. Fix and test CfC/NCP implementations\n")
        f.write("   2. Run systematic experiments with multiple seeds\n")
        f.write("   3. Include all relevant baselines\n")
        f.write("   4. Perform ablation studies (sparse vs. dense, CfC vs. RNN)\n")
        f.write("   5. Collect performance metrics (accuracy, BWT, FWT, forgetting)\n")
    
    print(f"\nText report saved to {report_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Analyze WandB continual learning experiments')
    parser.add_argument('--wandb_dir', type=str, 
                       default='mammoth/wandb',
                       help='Path to WandB directory')
    parser.add_argument('--output_dir', type=str,
                       default='analysis/results',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Parse runs
    print(f"Parsing WandB runs from: {args.wandb_dir}")
    runs = parse_local_wandb_runs(args.wandb_dir)
    
    if not runs:
        print("No runs found!")
        print("Note: Actual metrics may require WandB API access.")
        print("This script currently only parses local config files.")
        return
    
    # Generate report
    generate_report(runs, args.output_dir)


if __name__ == '__main__':
    main()
