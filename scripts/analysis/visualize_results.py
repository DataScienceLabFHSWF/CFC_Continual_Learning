#!/usr/bin/env python3
"""
Visualize benchmark results from JSON files.
Creates comparison plots and summary tables.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(json_file):
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_accuracy_comparison(results, output_dir):
    """Create bar plot comparing average accuracy across methods."""
    # Group by method
    method_results = defaultdict(list)
    for r in results:
        if r['accuracy'] is not None:
            method_results[r['method']].append(r['accuracy'])
    
    # Calculate statistics
    methods = []
    means = []
    stds = []
    
    for method in sorted(method_results.keys()):
        methods.append(method)
        accuracies = method_results[method]
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Continual Learning Methods - Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Color bars by performance
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if mean > 80:
            bar.set_color('green')
        elif mean > 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")
    plt.close()


def create_forgetting_comparison(results, output_dir):
    """Create bar plot comparing forgetting across methods."""
    method_results = defaultdict(list)
    for r in results:
        if r.get('forgetting') is not None:
            method_results[r['method']].append(r['forgetting'])
    
    if not method_results:
        print("No forgetting metrics found in results")
        return
    
    methods = []
    means = []
    stds = []
    
    for method in sorted(method_results.keys()):
        methods.append(method)
        forgettings = method_results[method]
        means.append(np.mean(forgettings))
        stds.append(np.std(forgettings))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Forgetting (%)', fontsize=12)
    ax.set_title('Continual Learning Methods - Forgetting Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Lower forgetting is better
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if mean < 10:
            bar.set_color('green')
        elif mean < 30:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'forgetting_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'forgetting_comparison.png'}")
    plt.close()


def create_heatmap(results, output_dir):
    """Create heatmap of methods × datasets."""
    # Create matrix
    method_dataset_acc = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r['accuracy'] is not None:
            method_dataset_acc[r['method']][r['dataset']].append(r['accuracy'])
    
    # Get unique methods and datasets
    methods = sorted(method_dataset_acc.keys())
    datasets = set()
    for method_data in method_dataset_acc.values():
        datasets.update(method_data.keys())
    datasets = sorted(datasets)
    
    # Build matrix
    matrix = np.zeros((len(methods), len(datasets)))
    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            if dataset in method_dataset_acc[method]:
                matrix[i, j] = np.mean(method_dataset_acc[method][dataset])
            else:
                matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=datasets, yticklabels=methods,
                cbar_kws={'label': 'Accuracy (%)'}, ax=ax,
                vmin=0, vmax=100)
    
    ax.set_title('Methods × Datasets Performance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'heatmap.png'}")
    plt.close()


def create_summary_table(results, output_dir):
    """Create summary table with all metrics."""
    # Group by method and dataset
    method_dataset_results = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r['method'], r.get('dataset', 'unknown'))
        method_dataset_results[key]['accuracy'].append(r.get('accuracy'))
        method_dataset_results[key]['forgetting'].append(r.get('forgetting'))
        method_dataset_results[key]['elapsed'].append(r.get('elapsed', 0))
        method_dataset_results[key]['success'].append(r.get('success', False))
    
    # Build DataFrame
    rows = []
    for (method, dataset), metrics in method_dataset_results.items():
        accuracies = [a for a in metrics['accuracy'] if a is not None]
        forgettings = [f for f in metrics['forgetting'] if f is not None]
        times = metrics['elapsed']
        successes = sum(metrics['success'])
        total = len(metrics['success'])
        
        row = {
            'Method': method,
            'Dataset': dataset,
            'Accuracy (%)': f"{np.mean(accuracies):.2f}±{np.std(accuracies):.2f}" if accuracies else 'N/A',
            'Forgetting (%)': f"{np.mean(forgettings):.2f}±{np.std(forgettings):.2f}" if forgettings else 'N/A',
            'Avg Time (s)': f"{np.mean(times):.1f}" if times else 'N/A',
            'Success Rate': f"{successes}/{total}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['Dataset', 'Method'])
    
    # Save as CSV
    csv_file = output_dir / 'summary_table.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")
    
    # Print to console
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    return df


def create_runtime_analysis(results, output_dir):
    """Create runtime comparison plot."""
    method_times = defaultdict(list)
    for r in results:
        if r.get('elapsed') and r['success']:
            method_times[r['method']].append(r['elapsed'])
    
    if not method_times:
        print("No runtime data found")
        return
    
    methods = []
    means = []
    stds = []
    
    for method in sorted(method_times.keys()):
        methods.append(method)
        times = method_times[method]
        means.append(np.mean(times))
        stds.append(np.std(times))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Method Runtime Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'runtime_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'runtime_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize CL benchmark results')
    parser.add_argument('input', type=str, help='Input JSON file with results')
    parser.add_argument('--output', type=str, default='results/plots',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Load results
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    results = load_results(input_file)
    print(f"Loaded {len(results)} experiment results from {input_file}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_accuracy_comparison(results, output_dir)
    create_forgetting_comparison(results, output_dir)
    create_heatmap(results, output_dir)
    create_runtime_analysis(results, output_dir)
    create_summary_table(results, output_dir)
    
    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
