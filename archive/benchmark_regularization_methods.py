#!/usr/bin/env python3
"""
Parallel benchmarking of regularization-based continual learning methods.
"""

import subprocess
import multiprocessing as mp
from pathlib import Path
import time
import json
import numpy as np
from collections import defaultdict

# Regularization methods with their key hyperparameters
REGULARIZATION_METHODS = {
    'ewc_on': {'e_lambda': 1000, 'gamma': 1.0},     # Online EWC
    'si': {'c': 0.5, 'xi': 1.0},                     # Synaptic Intelligence
    'lwf': {'alpha': 0.5, 'softmax_temp': 2.0},     # Learning without Forgetting
    'lwf_mc': {'alpha': 0.5, 'softmax_temp': 2.0},  # LwF Multi-Class
}

COMMON_ARGS = {
    'dataset': 'seq-mnist',
    'lr': 0.03,
    'n_epochs': 3,
    'batch_size': 32,
}

NUM_GPUS = 2
EXPERIMENTS_PER_GPU = 3


def run_experiment(method, method_params, gpu_id, seed=0):
    """Run a single experiment on a specific GPU."""
    cmd = [
        'python', 'utils/main.py',
        '--dataset', COMMON_ARGS['dataset'],
        '--model', method,
        '--lr', str(COMMON_ARGS['lr']),
        '--n_epochs', str(COMMON_ARGS['n_epochs']),
        '--batch_size', str(COMMON_ARGS['batch_size']),
        '--seed', str(seed),
        '--nowand', '1',
    ]
    
    # Add method-specific parameters
    for param, value in method_params.items():
        cmd.extend([f'--{param}', str(value)])
    
    env = {'CUDA_VISIBLE_DEVICES': str(gpu_id)}
    
    print(f"[GPU {gpu_id}] Starting {method} (seed={seed})")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/home/fneubuerger/CFC_Continual_Learning/mammoth',
            env={**subprocess.os.environ.copy(), **env},
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        elapsed = time.time() - start_time
        
        # Extract metrics
        accuracy = None
        forgetting = None
        for line in result.stdout.split('\n'):
            if 'Class-IL' in line:
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
        print(f"[GPU {gpu_id}] {status} {method} completed in {elapsed:.1f}s - Acc: {accuracy}, Forget: {forgetting}")
        
        return {
            'method': method,
            'params': method_params,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': accuracy,
            'forgetting': forgetting,
            'elapsed': elapsed,
            'success': result.returncode == 0,
            'stdout': result.stdout[-1000:],
            'stderr': result.stderr[-1000:],
        }
        
    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] ✗ {method} TIMEOUT")
        return {
            'method': method,
            'params': method_params,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': None,
            'forgetting': None,
            'elapsed': 3600,
            'success': False,
            'stdout': '',
            'stderr': 'TIMEOUT',
        }
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ {method} ERROR: {e}")
        return {
            'method': method,
            'params': method_params,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': None,
            'forgetting': None,
            'elapsed': 0,
            'success': False,
            'stdout': '',
            'stderr': str(e),
        }


def worker(task_queue, result_queue):
    """Worker process."""
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        method, params, gpu_id, seed = task
        result = run_experiment(method, params, gpu_id, seed)
        result_queue.put(result)


def main():
    """Run all regularization method experiments."""
    print(f"Starting regularization methods benchmark")
    print(f"Using {NUM_GPUS} GPUs with {EXPERIMENTS_PER_GPU} parallel experiments per GPU")
    print(f"Total methods to test: {len(REGULARIZATION_METHODS)}")
    print("-" * 80)
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add tasks (3 seeds per method)
    num_seeds = 3
    total_tasks = 0
    for method, params in REGULARIZATION_METHODS.items():
        for seed in range(num_seeds):
            gpu_id = total_tasks % NUM_GPUS
            task_queue.put((method, params, gpu_id, seed))
            total_tasks += 1
    
    # Add poison pills
    num_workers = NUM_GPUS * EXPERIMENTS_PER_GPU
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
    for _ in range(total_tasks):
        result = result_queue.get()
        results.append(result)
    
    for p in workers:
        p.join()
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Group by method
    method_results = defaultdict(list)
    for r in results:
        method_results[r['method']].append(r)
    
    # Print summary
    print(f"\n{'Method':<15} {'Avg Accuracy':<15} {'Std':<10} {'Avg Forgetting':<15} {'Success':<10}")
    print("-" * 80)
    
    for method in sorted(method_results.keys()):
        runs = method_results[method]
        accuracies = [r['accuracy'] for r in runs if r['accuracy'] is not None]
        forgettings = [r['forgetting'] for r in runs if r['forgetting'] is not None]
        successes = sum(1 for r in runs if r['success'])
        
        if accuracies:
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_forget = np.mean(forgettings) if forgettings else 0.0
            success_rate = f"{successes}/{len(runs)}"
            print(f"{method:<15} {avg_acc:>6.2f}%        {std_acc:>5.2f}%    {avg_forget:>6.2f}%        {success_rate:<10}")
        else:
            print(f"{method:<15} {'FAILED':<15} {'N/A':<10} {'N/A':<15} {successes}/{len(runs):<10}")
    
    # Save results
    output_file = Path('/home/fneubuerger/CFC_Continual_Learning/results/regularization_methods_benchmark.json')
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
