#!/usr/bin/env python3
"""
Parallel benchmarking of replay-based continual learning methods.
Runs multiple experiments across available GPUs.
"""

import subprocess
import multiprocessing as mp
from pathlib import Path
import time
import sys

# Replay-based methods with their buffer sizes
REPLAY_METHODS = {
    'er': 500,       # Experience Replay
    'der': 500,      # Dark Experience Replay
    'derpp': 500,    # DER++
    'gdumb': 500,    # GDumb
    'gss': 500,      # Gradient-based Sample Selection
    'hal': 500,      # Hindsight Anchor Learning
    'icarl': 2000,   # iCaRL (needs larger buffer)
    'mer': 500,      # Meta-Experience Replay
    'er_ace': 500,   # ER-ACE
    'xder': 500,     # X-DER
    'xder_ce': 500,  # X-DER with CE
    'xder_rpc': 500, # X-DER with RPC
    'fdr': 500,      # Feature Distillation Replay
}

# Common hyperparameters
COMMON_ARGS = {
    'dataset': 'seq-mnist',
    'lr': 0.03,
    'n_epochs': 3,
    'batch_size': 32,
}

# GPU configuration
NUM_GPUS = 2
EXPERIMENTS_PER_GPU = 3  # Run 3 experiments per GPU in parallel


def run_experiment(method, buffer_size, gpu_id, seed=0):
    """Run a single experiment on a specific GPU."""
    cmd = [
        'python', 'utils/main.py',
        '--dataset', COMMON_ARGS['dataset'],
        '--model', method,
        '--lr', str(COMMON_ARGS['lr']),
        '--n_epochs', str(COMMON_ARGS['n_epochs']),
        '--batch_size', str(COMMON_ARGS['batch_size']),
        '--buffer_size', str(buffer_size),
        '--seed', str(seed),
        '--nowand', '1',  # Disable wandb for now
    ]
    
    env = {'CUDA_VISIBLE_DEVICES': str(gpu_id)}
    
    print(f"[GPU {gpu_id}] Starting {method} with buffer_size={buffer_size}, seed={seed}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/home/fneubuerger/CFC_Continual_Learning/mammoth',
            env={**subprocess.os.environ.copy(), **env},
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        
        elapsed = time.time() - start_time
        
        # Extract final accuracy from output
        accuracy = None
        for line in result.stdout.split('\n'):
            if 'Class-IL' in line or 'Task-IL' in line:
                try:
                    accuracy = float(line.split()[-1])
                except:
                    pass
        
        status = "✓" if result.returncode == 0 else "✗"
        print(f"[GPU {gpu_id}] {status} {method} completed in {elapsed:.1f}s - Accuracy: {accuracy}")
        
        return {
            'method': method,
            'buffer_size': buffer_size,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': accuracy,
            'elapsed': elapsed,
            'success': result.returncode == 0,
            'stdout': result.stdout[-1000:] if result.stdout else '',  # Last 1000 chars
            'stderr': result.stderr[-1000:] if result.stderr else '',
        }
        
    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] ✗ {method} TIMEOUT after 1 hour")
        return {
            'method': method,
            'buffer_size': buffer_size,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': None,
            'elapsed': 3600,
            'success': False,
            'stdout': '',
            'stderr': 'TIMEOUT',
        }
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ {method} ERROR: {e}")
        return {
            'method': method,
            'buffer_size': buffer_size,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': None,
            'elapsed': 0,
            'success': False,
            'stdout': '',
            'stderr': str(e),
        }


def worker(task_queue, result_queue):
    """Worker process that consumes tasks from queue."""
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break
        
        method, buffer_size, gpu_id, seed = task
        result = run_experiment(method, buffer_size, gpu_id, seed)
        result_queue.put(result)


def main():
    """Run all experiments in parallel across GPUs."""
    print(f"Starting replay-based methods benchmark")
    print(f"Using {NUM_GPUS} GPUs with {EXPERIMENTS_PER_GPU} parallel experiments per GPU")
    print(f"Total parallel experiments: {NUM_GPUS * EXPERIMENTS_PER_GPU}")
    print(f"Total methods to test: {len(REPLAY_METHODS)}")
    print("-" * 80)
    
    # Create task queue
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add all tasks to queue (3 seeds per method)
    num_seeds = 3
    total_tasks = 0
    for method, buffer_size in REPLAY_METHODS.items():
        for seed in range(num_seeds):
            gpu_id = total_tasks % NUM_GPUS  # Round-robin GPU assignment
            task_queue.put((method, buffer_size, gpu_id, seed))
            total_tasks += 1
    
    # Add poison pills
    num_workers = NUM_GPUS * EXPERIMENTS_PER_GPU
    for _ in range(num_workers):
        task_queue.put(None)
    
    # Start worker processes
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
    
    # Wait for all workers to finish
    for p in workers:
        p.join()
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Group results by method
    from collections import defaultdict
    method_results = defaultdict(list)
    for r in results:
        method_results[r['method']].append(r)
    
    # Print summary
    print(f"\n{'Method':<15} {'Avg Accuracy':<15} {'Std':<10} {'Success Rate':<15} {'Avg Time':<10}")
    print("-" * 80)
    
    for method in sorted(method_results.keys()):
        method_runs = method_results[method]
        accuracies = [r['accuracy'] for r in method_runs if r['accuracy'] is not None]
        times = [r['elapsed'] for r in method_runs]
        successes = sum(1 for r in method_runs if r['success'])
        
        if accuracies:
            import numpy as np
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_time = np.mean(times)
            success_rate = f"{successes}/{len(method_runs)}"
            print(f"{method:<15} {avg_acc:>6.2f}%        {std_acc:>5.2f}%    {success_rate:<15} {avg_time:>6.1f}s")
        else:
            print(f"{method:<15} {'FAILED':<15} {'N/A':<10} {successes}/{len(method_runs):<15}")
    
    # Save detailed results
    import json
    output_file = Path('/home/fneubuerger/CFC_Continual_Learning/results/replay_methods_benchmark.json')
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print any failures
    failures = [r for r in results if not r['success']]
    if failures:
        print(f"\n⚠️  {len(failures)} failed experiments:")
        for r in failures:
            print(f"  - {r['method']} (seed={r['seed']}): {r['stderr'][:100]}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
