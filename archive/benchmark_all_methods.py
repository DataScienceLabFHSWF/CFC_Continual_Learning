#!/usr/bin/env python3
"""
Comprehensive benchmark of all CL methods across multiple datasets.
Runs experiments in parallel across available GPUs.
"""

import subprocess
import multiprocessing as mp
from pathlib import Path
import time
import json
import numpy as np
from collections import defaultdict
import argparse

# All methods with their configurations
METHOD_CONFIGS = {
    # Baseline
    'sgd': {},
    'joint': {},
    
    # Replay-based
    'er': {'buffer_size': 500},
    'der': {'buffer_size': 500, 'alpha': 0.1},
    'derpp': {'buffer_size': 500, 'alpha': 0.1, 'beta': 0.5},
    'gdumb': {'buffer_size': 500},
    'gss': {'buffer_size': 500},
    'hal': {'buffer_size': 500, 'hal_lambda': 0.1},
    'icarl': {'buffer_size': 2000},
    'mer': {'buffer_size': 500, 'beta': 0.1, 'gamma': 1.0},
    'er_ace': {'buffer_size': 500},
    'fdr': {'buffer_size': 500, 'alpha': 0.7},
    
    # Regularization
    'ewc_on': {'e_lambda': 1000, 'gamma': 1.0},
    'si': {'c': 0.5, 'xi': 1.0},
    'lwf': {'alpha': 0.5, 'softmax_temp': 2.0},
    
    # Architecture
    'pnn': {},
    'rpc': {'rho': 0.3},
    
    # Distillation
    'bic': {'buffer_size': 500},
    'lucir': {'buffer_size': 500},
}

# Dataset configurations
DATASET_CONFIGS = {
    'seq-mnist': {'lr': 0.03, 'n_epochs': 3, 'batch_size': 32},
    'perm-mnist': {'lr': 0.03, 'n_epochs': 3, 'batch_size': 32},
    'rot-mnist': {'lr': 0.03, 'n_epochs': 3, 'batch_size': 32},
}

NUM_GPUS = 2
EXPERIMENTS_PER_GPU = 3


def run_experiment(method, method_params, dataset, dataset_params, gpu_id, seed=0):
    """Run a single experiment."""
    cmd = [
        'python', 'utils/main.py',
        '--dataset', dataset,
        '--model', method,
        '--seed', str(seed),
        '--nowand', '1',
    ]
    
    # Add dataset params
    for param, value in dataset_params.items():
        cmd.extend([f'--{param}', str(value)])
    
    # Add method params
    for param, value in method_params.items():
        cmd.extend([f'--{param}', str(value)])
    
    env = {'CUDA_VISIBLE_DEVICES': str(gpu_id)}
    
    exp_name = f"{method}_{dataset}_s{seed}"
    print(f"[GPU {gpu_id}] Starting {exp_name}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/home/fneubuerger/CFC_Continual_Learning/mammoth',
            env={**subprocess.os.environ.copy(), **env},
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        # Parse output
        accuracy = None
        forgetting = None
        for line in result.stdout.split('\n'):
            if 'Class-IL' in line or 'Task-IL' in line:
                try:
                    accuracy = float(line.split()[-1])
                except:
                    pass
            if 'Forgetting' in line:
                try:
                    forgetting = float(line.split()[-1])
                except:
                    pass
        
        status = "✓" if result.returncode == 0 else "✗"
        print(f"[GPU {gpu_id}] {status} {exp_name} - {elapsed:.1f}s - Acc: {accuracy}")
        
        return {
            'method': method,
            'dataset': dataset,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': accuracy,
            'forgetting': forgetting,
            'elapsed': elapsed,
            'success': result.returncode == 0,
            'method_params': method_params,
            'dataset_params': dataset_params,
        }
        
    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] ✗ {exp_name} TIMEOUT")
        return {
            'method': method,
            'dataset': dataset,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': None,
            'forgetting': None,
            'elapsed': 7200,
            'success': False,
            'method_params': method_params,
            'dataset_params': dataset_params,
        }
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ {exp_name} ERROR: {e}")
        return {
            'method': method,
            'dataset': dataset,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': None,
            'forgetting': None,
            'elapsed': 0,
            'success': False,
            'method_params': method_params,
            'dataset_params': dataset_params,
        }


def worker(task_queue, result_queue):
    """Worker process."""
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        method, method_params, dataset, dataset_params, gpu_id, seed = task
        result = run_experiment(method, method_params, dataset, dataset_params, gpu_id, seed)
        result_queue.put(result)


def print_results_table(results, datasets):
    """Print formatted results table."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 120)
    
    # Group by method and dataset
    method_dataset_results = defaultdict(lambda: defaultdict(list))
    for r in results:
        method_dataset_results[r['method']][r['dataset']].append(r)
    
    # Header
    header = f"{'Method':<15}"
    for dataset in datasets:
        header += f" {dataset:<25}"
    print("\n" + header)
    print("-" * 120)
    
    # Print each method
    for method in sorted(method_dataset_results.keys()):
        row = f"{method:<15}"
        for dataset in datasets:
            runs = method_dataset_results[method][dataset]
            if runs:
                accuracies = [r['accuracy'] for r in runs if r['accuracy'] is not None]
                if accuracies:
                    avg = np.mean(accuracies)
                    std = np.std(accuracies)
                    row += f" {avg:5.2f}±{std:4.2f}%           "
                else:
                    row += f" {'FAILED':<25}"
            else:
                row += f" {'-':<25}"
        print(row)
    
    print("-" * 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', help='Specific methods to test (default: all)')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_CONFIGS.keys()),
                       help='Datasets to test')
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds')
    parser.add_argument('--gpus', type=int, default=NUM_GPUS, help='Number of GPUs to use')
    args = parser.parse_args()
    
    methods = args.methods if args.methods else list(METHOD_CONFIGS.keys())
    datasets = args.datasets
    num_seeds = args.seeds
    num_gpus = args.gpus
    
    print(f"Comprehensive CL Benchmark")
    print(f"Methods: {len(methods)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Seeds per config: {num_seeds}")
    print(f"GPUs: {num_gpus} x {EXPERIMENTS_PER_GPU} parallel experiments")
    print(f"Total experiments: {len(methods) * len(datasets) * num_seeds}")
    print("-" * 80)
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Create all tasks
    total_tasks = 0
    for method in methods:
        method_params = METHOD_CONFIGS[method]
        for dataset in datasets:
            dataset_params = DATASET_CONFIGS[dataset]
            for seed in range(num_seeds):
                gpu_id = total_tasks % num_gpus
                task_queue.put((method, method_params, dataset, dataset_params, gpu_id, seed))
                total_tasks += 1
    
    # Add poison pills
    num_workers = num_gpus * EXPERIMENTS_PER_GPU
    for _ in range(num_workers):
        task_queue.put(None)
    
    # Start workers
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(task_queue, result_queue))
        p.start()
        workers.append(p)
    
    # Collect results
    results = []
    for i in range(total_tasks):
        result = result_queue.get()
        results.append(result)
        if (i + 1) % 10 == 0:
            print(f"\nProgress: {i+1}/{total_tasks} experiments completed")
    
    for p in workers:
        p.join()
    
    # Print results
    print_results_table(results, datasets)
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = Path(f'/home/fneubuerger/CFC_Continual_Learning/results/benchmark_{timestamp}.json')
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Summary statistics
    successes = sum(1 for r in results if r['success'])
    failures = len(results) - successes
    total_time = sum(r['elapsed'] for r in results)
    
    print(f"\nSummary:")
    print(f"  Total experiments: {len(results)}")
    print(f"  Successful: {successes}")
    print(f"  Failed: {failures}")
    print(f"  Total compute time: {total_time/3600:.2f} GPU-hours")
    print(f"  Wall clock time: {total_time/(num_gpus * EXPERIMENTS_PER_GPU * 3600):.2f} hours")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
