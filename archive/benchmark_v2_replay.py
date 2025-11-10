#!/usr/bin/env python3
"""
Parallel benchmarking of replay-based continual learning methods for Mammoth v2.0.
Runs multiple experiments across available GPUs.
"""

import subprocess
import multiprocessing as mp
from pathlib import Path
import time
import sys
import os

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


def run_experiment(method, buffer_size, gpu_id, seed=0, backbone='mnistmlp'):
    """Run a single experiment on a specific GPU."""
    cmd = [
        'python', 'utils/main.py',
        '--dataset', COMMON_ARGS['dataset'],
        '--model', method,
        '--backbone', backbone,
        '--lr', str(COMMON_ARGS['lr']),
        '--n_epochs', str(COMMON_ARGS['n_epochs']),
        '--batch_size', str(COMMON_ARGS['batch_size']),
        '--buffer_size', str(buffer_size),
        '--seed', str(seed),
    ]
    
    # Set GPU via environment variable
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Starting {method} (backbone={backbone}) with buffer_size={buffer_size}, seed={seed}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd='mammoth',
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        
        elapsed = time.time() - start_time
        
        # Extract final accuracy from output
        accuracy = None
        forgetting = None
        for line in result.stdout.split('\n'):
            if 'Class-IL' in line or 'Task-IL' in line:
                try:
                    parts = line.split()
                    accuracy = float(parts[-1])
                except:
                    pass
            if 'Forgetting' in line:
                try:
                    forgetting = float(line.split()[-1])
                except:
                    pass
        
        return {
            'method': method,
            'backbone': backbone,
            'buffer_size': buffer_size,
            'seed': seed,
            'accuracy': accuracy,
            'forgetting': forgetting,
            'time': elapsed,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] {method} TIMEOUT after 1 hour")
        return {
            'method': method,
            'backbone': backbone,
            'buffer_size': buffer_size,
            'seed': seed,
            'success': False,
            'error': 'timeout',
        }
    except Exception as e:
        print(f"[GPU {gpu_id}] {method} ERROR: {e}")
        return {
            'method': method,
            'backbone': backbone,
            'buffer_size': buffer_size,
            'seed': seed,
            'success': False,
            'error': str(e),
        }


def worker(task_queue, result_queue):
    """Worker process to run experiments from queue."""
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break
        
        method, buffer_size, gpu_id, seed, backbone = task
        result = run_experiment(method, buffer_size, gpu_id, seed, backbone)
        result_queue.put(result)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark replay methods on Mammoth v2.0')
    parser.add_argument('--backbone', type=str, default='mnistmlp', 
                       choices=['mnistmlp', 'mnistcfc'],
                       help='Backbone to use')
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds')
    parser.add_argument('--output', type=str, default='results/v2_replay_benchmark.json',
                       help='Output JSON file')
    args = parser.parse_args()
    
    # Create task queue
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Populate task queue
    tasks = []
    gpu_id = 0
    for method, buffer_size in REPLAY_METHODS.items():
        for seed in range(args.seeds):
            task = (method, buffer_size, gpu_id % NUM_GPUS, seed, args.backbone)
            tasks.append(task)
            task_queue.put(task)
            gpu_id += 1
    
    # Add poison pills
    for _ in range(NUM_GPUS * EXPERIMENTS_PER_GPU):
        task_queue.put(None)
    
    # Start worker processes
    workers = []
    for _ in range(NUM_GPUS * EXPERIMENTS_PER_GPU):
        p = mp.Process(target=worker, args=(task_queue, result_queue))
        p.start()
        workers.append(p)
    
    # Collect results
    results = []
    for _ in range(len(tasks)):
        result = result_queue.get()
        results.append(result)
        if result['success']:
            print(f"✓ {result['method']} (seed={result['seed']}): "
                  f"acc={result.get('accuracy', 'N/A'):.2f}%, "
                  f"time={result['time']:.1f}s")
        else:
            print(f"✗ {result['method']} (seed={result['seed']}): FAILED")
    
    # Wait for workers
    for p in workers:
        p.join()
    
    # Save results
    import json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    from collections import defaultdict
    import numpy as np
    
    method_stats = defaultdict(list)
    for r in results:
        if r['success'] and r.get('accuracy') is not None:
            method_stats[r['method']].append(r['accuracy'])
    
    for method in sorted(method_stats.keys()):
        accs = method_stats[method]
        mean = np.mean(accs)
        std = np.std(accs)
        print(f"{method:15s}: {mean:.2f}% ± {std:.2f}%")


if __name__ == '__main__':
    main()
