#!/usr/bin/env python3
"""
CfC Continual Learning - WandB Results Analyzer

This script analyzes completed runs from WandB and generates summary tables.

Usage:
    python analyze_wandb_results.py [--output-dir DIR]
"""

import os
import json
import argparse
import pandas as pd
import wandb
from pathlib import Path
from typing import List, Dict


def load_wandb_api():
    """Load WandB API with credentials."""
    secrets_path = Path(__file__).parent.parent.parent / '.secrets.json'
    with open(secrets_path) as f:
        api_key = json.load(f)['wandb_api_key']
    
    wandb.login(key=api_key, relogin=True)
    return wandb.Api()


def fetch_finished_runs(api, entity="fneubuerger", project="mammoth", min_year=2024) -> List:
    """Fetch all finished runs from WandB.
    
    Args:
        api: WandB API object
        entity: WandB entity name
        project: WandB project name
        min_year: Minimum year to include (default 2024, filters out old 2023 runs)
    
    Returns:
        List of recent finished runs
    """
    from datetime import datetime
    
    all_runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
    
    # Filter by year
    recent_runs = []
    archived_count = 0
    
    for run in all_runs:
        created = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
        if created.year >= min_year:
            recent_runs.append(run)
        else:
            archived_count += 1
    
    if archived_count > 0:
        print(f"Note: {archived_count} runs from before {min_year} archived (not analyzed)")
    
    return recent_runs


def parse_run_name(name: str) -> Dict:
    """Parse experiment name into components.
    
    Expected format: {dataset}_{backbone}_{model}_seed{seed}
    Examples:
        mnist_mlp_sgd_seed0
        cifar_cnn_er200_seed1
        tep_lstm_derpp500_seed2
    """
    parts = name.split('_')
    
    result = {
        'dataset': None,
        'backbone': None,
        'model': None,
        'buffer_size': None,
        'seed': 0,
    }
    
    # Find seed
    for i, part in enumerate(parts):
        if part.startswith('seed'):
            result['seed'] = int(part.replace('seed', ''))
            remaining = parts[:i]
            break
    else:
        remaining = parts
    
    if len(remaining) >= 3:
        result['dataset'] = remaining[0]
        result['backbone'] = remaining[1]
        
        # Check if model name contains buffer size (e.g., er200, derpp500)
        model_part = '_'.join(remaining[2:])
        
        # Extract buffer size if present
        import re
        buffer_match = re.search(r'(\d+)$', model_part)
        if buffer_match:
            result['buffer_size'] = int(buffer_match.group(1))
            result['model'] = model_part[:buffer_match.start()]
        else:
            result['model'] = model_part
    
    return result


def aggregate_wandb_results(runs: List) -> pd.DataFrame:
    """Aggregate WandB runs into a DataFrame."""
    
    rows = []
    
    for run in runs:
        # Parse run name
        parsed = parse_run_name(run.name)
        
        # Extract metrics from summary
        summary = run.summary
        
        row = {
            'run_id': run.id,
            'run_name': run.name,
            'dataset': parsed['dataset'],
            'backbone': parsed['backbone'],
            'model': parsed['model'],
            'buffer_size': parsed['buffer_size'],
            'seed': parsed['seed'],
            'state': run.state,
            'created_at': run.created_at,
            'runtime_seconds': (run.summary.get('_runtime', 0)),
        }
        
        # Add result metrics
        for key, value in summary.items():
            if key.startswith('RESULT_'):
                # Clean key name
                metric_name = key.replace('RESULT_', '').lower()
                row[metric_name] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_summary_tables(df: pd.DataFrame, output_dir: Path):
    """Generate summary tables by dataset and model."""
    
    if len(df) == 0:
        print("No data to summarize")
        return
    
    # Group by dataset
    for dataset in df['dataset'].unique():
        if pd.isna(dataset):
            continue
            
        dataset_df = df[df['dataset'] == dataset]
        
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*80}")
        
        # Group by model and aggregate seeds
        if 'class_mean_accs' in dataset_df.columns:
            # Group by model (and buffer_size if it varies)
            group_cols = ['model']
            if dataset_df['buffer_size'].notna().any():
                group_cols.append('buffer_size')
            
            summary = dataset_df.groupby(group_cols).agg({
                'class_mean_accs': ['mean', 'std', 'count'],
                'task_mean_accs': ['mean', 'std', 'count'] if 'task_mean_accs' in dataset_df.columns else lambda x: None,
            }).round(2)
            
            print("\nResults Summary (Class-IL / Task-IL):")
            print(summary)
            
            # Save to CSV
            summary.to_csv(output_dir / f'summary_{dataset}.csv')
            print(f"Saved: {output_dir / f'summary_{dataset}.csv'}")
            
            # Create LaTeX table
            latex_table = summary.to_latex()
            with open(output_dir / f'table_{dataset}.tex', 'w') as f:
                f.write(latex_table)
            print(f"Saved: {output_dir / f'table_{dataset}.tex'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze WandB results')
    parser.add_argument('--output-dir', type=str,
                        default='paper_results/wandb_analysis',
                        help='Directory to save analysis outputs')
    parser.add_argument('--entity', type=str,
                        default='fneubuerger',
                        help='WandB entity')
    parser.add_argument('--project', type=str,
                        default='mammoth',
                        help='WandB project')
    parser.add_argument('--min-year', type=int,
                        default=2024,
                        help='Minimum year to include (filters out older runs)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Connecting to WandB...")
    api = load_wandb_api()
    
    print(f"Fetching finished runs from {args.entity}/{args.project}...")
    runs = fetch_finished_runs(api, args.entity, args.project, args.min_year)
    
    print(f"Found {len(runs)} finished runs (from {args.min_year} onwards)")
    
    if len(runs) == 0:
        print("No finished runs found!")
        return
    
    print("\nAggregating results...")
    df = aggregate_wandb_results(runs)
    
    # Save raw data
    df.to_csv(output_dir / 'raw_wandb_results.csv', index=False)
    print(f"Saved: {output_dir / 'raw_wandb_results.csv'}")
    
    # Display summary
    print(f"\n{'='*80}")
    print("Dataset Distribution:")
    print(df['dataset'].value_counts())
    
    print(f"\n{'='*80}")
    print("Model Distribution:")
    print(df['model'].value_counts())
    
    # Generate summaries
    generate_summary_tables(df, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
