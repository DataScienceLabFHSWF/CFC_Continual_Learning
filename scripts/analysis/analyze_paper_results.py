#!/usr/bin/env python3
"""
CfC Continual Learning - Paper Results Analyzer

This script analyzes the benchmark results and generates summary tables
and plots for the paper.

Usage:
    python analyze_paper_results.py [--results-dir DIR] [--output-dir DIR]
"""

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


def parse_log_file(log_path: str) -> Dict:
    """Extract metrics from a log file."""
    metrics = {
        'completed': False,
        'class_il_acc': [],
        'task_il_acc': [],
        'final_class_il': None,
        'final_task_il': None,
        'avg_class_il': None,
        'avg_task_il': None,
        'forgetting': None,
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # Check if completed
        if 'Experiment completed' in content or 'Task 5 accuracy' in content:
            metrics['completed'] = True
        
        # Extract accuracy values
        # Pattern: "Task X accuracy: Y.YY%"
        task_matches = re.findall(r'Task \d+ accuracy: ([\d.]+)%', content)
        if task_matches:
            metrics['task_il_acc'] = [float(x) for x in task_matches]
        
        # Extract Class-IL and Task-IL from final evaluation
        class_il_match = re.search(r'Class-IL accuracy: ([\d.]+)%', content)
        task_il_match = re.search(r'Task-IL accuracy: ([\d.]+)%', content)
        
        if class_il_match:
            metrics['final_class_il'] = float(class_il_match.group(1))
        if task_il_match:
            metrics['final_task_il'] = float(task_il_match.group(1))
        
        # Calculate averages
        if metrics['task_il_acc']:
            metrics['avg_task_il'] = np.mean(metrics['task_il_acc'])
        
        # Calculate forgetting (drop from max to final)
        if len(metrics['task_il_acc']) > 1:
            max_acc = max(metrics['task_il_acc'])
            final_acc = metrics['task_il_acc'][-1]
            metrics['forgetting'] = max_acc - final_acc
    
    return metrics


def parse_csv_results(csv_path: str) -> Dict:
    """Parse CSV results file if it exists."""
    if not os.path.exists(csv_path):
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        # Extract final metrics
        return {
            'class_il_acc': df['class_il_acc'].values if 'class_il_acc' in df else [],
            'task_il_acc': df['task_il_acc'].values if 'task_il_acc' in df else [],
            'final_class_il': df['class_il_acc'].iloc[-1] if 'class_il_acc' in df and len(df) > 0 else None,
            'final_task_il': df['task_il_acc'].iloc[-1] if 'task_il_acc' in df and len(df) > 0 else None,
        }
    except Exception as e:
        print(f"Warning: Could not parse {csv_path}: {e}")
        return {}


def aggregate_results(results_dir: Path) -> pd.DataFrame:
    """Aggregate all results into a DataFrame."""
    
    rows = []
    
    # Find all log files
    log_files = glob.glob(str(results_dir / 'logs' / '*.log'))
    
    for log_path in log_files:
        # Parse filename: {dataset}_{backbone}_{model}[_{buffer}]_seed{seed}.log
        filename = Path(log_path).stem
        
        # Extract components
        parts = filename.split('_')
        
        # Try to parse the filename
        try:
            dataset = parts[0]
            backbone = None
            model = None
            buffer_size = None
            seed = None
            
            # Find seed
            for i, part in enumerate(parts):
                if part.startswith('seed'):
                    seed = int(part.replace('seed', ''))
                    remaining = parts[:i]
                    break
            else:
                seed = 0
                remaining = parts
            
            # Parse dataset_backbone_model[buffer]
            if len(remaining) >= 3:
                dataset = remaining[0]
                backbone = remaining[1]
                
                # Check if last part is a buffer size
                if remaining[-1].isdigit():
                    buffer_size = int(remaining[-1])
                    model = '_'.join(remaining[2:-1])
                else:
                    model = '_'.join(remaining[2:])
            
            # Parse metrics
            metrics = parse_log_file(log_path)
            
            # Try to parse CSV if exists
            csv_path = str(results_dir / f"{filename}.csv")
            csv_metrics = parse_csv_results(csv_path)
            
            # Merge metrics (CSV takes precedence)
            if csv_metrics:
                metrics.update(csv_metrics)
            
            row = {
                'dataset': dataset,
                'backbone': backbone,
                'model': model,
                'buffer_size': buffer_size,
                'seed': seed,
                'completed': metrics['completed'],
                'final_class_il': metrics.get('final_class_il'),
                'final_task_il': metrics.get('final_task_il'),
                'avg_task_il': metrics.get('avg_task_il'),
                'forgetting': metrics.get('forgetting'),
                'log_file': log_path,
            }
            
            rows.append(row)
            
        except Exception as e:
            print(f"Warning: Could not parse {log_path}: {e}")
            continue
    
    df = pd.DataFrame(rows)
    return df


def generate_summary_tables(df: pd.DataFrame, output_dir: Path):
    """Generate summary tables for the paper."""
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Group by dataset, backbone, model, buffer_size
    grouped = df.groupby(['dataset', 'backbone', 'model', 'buffer_size'])
    
    # Calculate mean and std across seeds
    summary = grouped.agg({
        'final_class_il': ['mean', 'std', 'count'],
        'final_task_il': ['mean', 'std', 'count'],
        'forgetting': ['mean', 'std'],
    }).round(2)
    
    # Save full summary
    summary.to_csv(output_dir / 'summary_all.csv')
    print(f"Saved: {output_dir / 'summary_all.csv'}")
    
    # Generate dataset-specific summaries
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        dataset_summary = dataset_df.groupby(['backbone', 'model', 'buffer_size']).agg({
            'final_class_il': ['mean', 'std', 'count'],
            'final_task_il': ['mean', 'std', 'count'],
            'forgetting': ['mean', 'std'],
        }).round(2)
        
        dataset_summary.to_csv(output_dir / f'summary_{dataset}.csv')
        print(f"Saved: {output_dir / f'summary_{dataset}.csv'}")
    
    # Generate LaTeX tables
    generate_latex_tables(df, output_dir)
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal experiments: {len(df)}")
    print(f"Completed: {df['completed'].sum()} / {len(df)}")
    print(f"\nDatasets: {df['dataset'].unique()}")
    print(f"Backbones: {df['backbone'].unique()}")
    print(f"Models: {df['model'].unique()}")
    
    # Best results per dataset
    print("\n" + "-"*80)
    print("BEST RESULTS (by Final Class-IL Accuracy)")
    print("-"*80)
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        if len(dataset_df) == 0:
            continue
        
        # Average across seeds
        avg_df = dataset_df.groupby(['backbone', 'model', 'buffer_size']).agg({
            'final_class_il': 'mean',
            'final_task_il': 'mean',
        }).reset_index()
        
        # Sort by Class-IL accuracy
        avg_df = avg_df.sort_values('final_class_il', ascending=False)
        
        print(f"\n{dataset.upper()}:")
        print(avg_df.head(10).to_string(index=False))


def generate_latex_tables(df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX tables for the paper."""
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # Average across seeds
        summary = dataset_df.groupby(['backbone', 'model', 'buffer_size']).agg({
            'final_class_il': ['mean', 'std'],
            'final_task_il': ['mean', 'std'],
        }).round(2)
        
        # Format as mean ± std
        latex_rows = []
        for idx, row in summary.iterrows():
            backbone, model, buffer = idx
            
            class_il_mean = row[('final_class_il', 'mean')]
            class_il_std = row[('final_class_il', 'std')]
            task_il_mean = row[('final_task_il', 'mean')]
            task_il_std = row[('final_task_il', 'std')]
            
            buffer_str = str(buffer) if buffer is not None else '-'
            
            latex_row = f"{backbone} & {model} & {buffer_str} & "
            latex_row += f"{class_il_mean:.2f} $\\pm$ {class_il_std:.2f} & "
            latex_row += f"{task_il_mean:.2f} $\\pm$ {task_il_std:.2f} \\\\"
            
            latex_rows.append(latex_row)
        
        # Create LaTeX table
        latex_table = "\\begin{table}[h]\n"
        latex_table += "\\centering\n"
        latex_table += "\\begin{tabular}{llccc}\n"
        latex_table += "\\toprule\n"
        latex_table += "Backbone & Model & Buffer & Class-IL Acc. & Task-IL Acc. \\\\\n"
        latex_table += "\\midrule\n"
        latex_table += "\n".join(latex_rows)
        latex_table += "\n\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += f"\\caption{{Results on {dataset.upper()} dataset}}\n"
        latex_table += f"\\label{{tab:{dataset}}}\n"
        latex_table += "\\end{table}\n"
        
        # Save LaTeX table
        with open(output_dir / f'table_{dataset}.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"Saved: {output_dir / f'table_{dataset}.tex'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze CfC paper benchmark results')
    parser.add_argument('--results-dir', type=str, 
                        default='paper_results',
                        help='Directory containing results')
    parser.add_argument('--output-dir', type=str,
                        default='paper_results/analysis',
                        help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print("Aggregating results...")
    df = aggregate_results(results_dir)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"Found {len(df)} experiment results")
    
    # Save raw data
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_dir / 'raw_results.csv', index=False)
    print(f"Saved: {output_dir / 'raw_results.csv'}")
    
    # Generate summaries
    generate_summary_tables(df, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
