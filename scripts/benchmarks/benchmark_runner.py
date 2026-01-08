#!/usr/bin/env python3
"""
Benchmark Runner for CfC Continual Learning
Runs multiple experiments based on YAML configuration files.

Usage:
    python benchmark_runner.py --config configs/benchmark_config.yaml
    python benchmark_runner.py --config configs/quick_test.yaml --dry-run
"""

import argparse
import yaml
import subprocess
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import itertools
from multiprocessing import Pool, Manager
import queue


class BenchmarkRunner:
    """Runs multiple benchmark experiments based on configuration."""
    
    def __init__(self, config_path: str, dry_run: bool = False, num_parallel: int = 1):
        self.config_path = config_path
        self.dry_run = dry_run
        self.num_parallel = num_parallel
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create run-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{timestamp}"
        if not dry_run:
            self.run_dir.mkdir(exist_ok=True)
            # Save config for reproducibility
            with open(self.run_dir / "config.yaml", 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        
        # GPU assignment for parallel execution
        self.gpus = self.config.get('gpus', [0])
        if num_parallel > len(self.gpus):
            print(f"WARNING: num_parallel ({num_parallel}) > available GPUs ({len(self.gpus)})")
            print(f"         Will use {len(self.gpus)} parallel processes")
    
    def generate_experiments(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations from the config file."""
        experiments = []
        
        # Get global settings
        global_args = self.config.get('global_args', {})
        base_path = self.config.get('base_path', './mammoth')
        python_cmd = self.config.get('python_cmd', 'python')
        venv_path = self.config.get('venv_path', None)
        
        # Check for explicit experiments list
        explicit_experiments = self.config.get('explicit_experiments', None)
        if explicit_experiments:
            for exp_conf in explicit_experiments:
                # Expand seeds if list
                exp_seeds = exp_conf.get('seeds', [0])
                if isinstance(exp_seeds, int):
                    exp_seeds = [exp_seeds]
                    
                for seed in exp_seeds:
                    exp = {
                        'dataset': exp_conf['dataset'],
                        'model': exp_conf['model'],
                        'backbone': exp_conf.get('backbone', None),
                        'seed': seed,
                        'base_path': base_path,
                        'python_cmd': python_cmd,
                        'venv_path': venv_path,
                        'traditional_ml': exp_conf.get('traditional_ml', False),
                        'args': {}
                    }
                    
                    # Merge arguments
                    exp['args'].update(global_args)
                    exp['args'].update(exp_conf.get('args', {}))
                    exp['args']['seed'] = seed
                    
                    experiments.append(exp)
            
            return experiments

        # Get experiment grid
        datasets = self.config.get('datasets', ['seq-mnist'])
        models = self.config.get('models', ['sgd'])
        backbones = self.config.get('backbones', ['mnistcfc'])
        seeds = self.config.get('seeds', [0])
        
        # Get model-specific and backbone-specific args
        model_args = self.config.get('model_args', {})
        backbone_args = self.config.get('backbone_args', {})
        dataset_args = self.config.get('dataset_args', {})
        
        # Check if we have traditional ML methods (tree-based)
        traditional_ml = self.config.get('traditional_ml', None)
        
        if traditional_ml:
            # Generate experiments for traditional ML methods
            ml_models = traditional_ml.get('models', [])
            ml_datasets = traditional_ml.get('datasets', datasets)
            ml_seeds = traditional_ml.get('seeds', seeds)
            ml_args = traditional_ml.get('args', {})
            
            for dataset, model, seed in itertools.product(ml_datasets, ml_models, ml_seeds):
                exp = {
                    'dataset': dataset,
                    'model': model,
                    'backbone': None,  # No backbone for traditional ML
                    'seed': seed,
                    'base_path': base_path,
                    'python_cmd': python_cmd,
                    'venv_path': venv_path,
                    'traditional_ml': True,
                    'args': {}
                }
                
                # Add global and ML-specific args
                exp['args'].update(global_args)
                exp['args'].update(ml_args)
                
                # Add model-specific args for ML
                if model in traditional_ml.get('model_args', {}):
                    exp['args'].update(traditional_ml['model_args'][model])
                
                exp['args']['seed'] = seed
                experiments.append(exp)
        
        # Generate all combinations for neural network methods
        for dataset, model, backbone, seed in itertools.product(datasets, models, backbones, seeds):
            exp = {
                'dataset': dataset,
                'model': model,
                'backbone': backbone,
                'seed': seed,
                'base_path': base_path,
                'python_cmd': python_cmd,
                'venv_path': venv_path,
                'traditional_ml': False,
                'args': {}
            }
            
            # Merge arguments in order of priority: global < dataset-specific < model-specific < backbone-specific
            exp['args'].update(global_args)
            
            # Add dataset-specific args
            if dataset in dataset_args:
                exp['args'].update(dataset_args[dataset])
            
            # Add model-specific args
            if model in model_args:
                exp['args'].update(model_args[model])
            
            # Add backbone-specific args
            if backbone in backbone_args:
                exp['args'].update(backbone_args[backbone])
            
            # Add seed
            exp['args']['seed'] = seed
            
            experiments.append(exp)
        
        return experiments
    
    def build_command(self, exp: Dict[str, Any], gpu_id: int = None) -> str:
        """Build the command line for an experiment."""
        base_path = exp['base_path']
        python_cmd = exp['python_cmd']
        venv_path = exp['venv_path']
        
        # Check if this is a traditional ML experiment
        if exp.get('traditional_ml', False):
            # Use the gradient boosting script
            script_path = '../scripts/analysis/tep_gradient_boosting.py'
            args_list = [
                f"--dataset {exp['dataset']}",
                f"--model {exp['model']}",
                f"--seed {exp['seed']}"
            ]
            
            # Add other arguments
            for key, value in exp['args'].items():
                if key != 'seed':
                    if isinstance(value, bool):
                        if value:
                            args_list.append(f"--{key}")
                    elif isinstance(value, list):
                        args_list.append(f"--{key} {' '.join(map(str, value))}")
                    else:
                        args_list.append(f"--{key} {value}")
            
            # Build command for traditional ML
            if venv_path:
                cmd = f"cd {base_path} && source {venv_path} && {python_cmd} {script_path} {' '.join(args_list)}"
            else:
                cmd = f"cd {base_path} && {python_cmd} {script_path} {' '.join(args_list)}"
        else:
            # Neural network experiment
            args_list = [
                f"--dataset {exp['dataset']}",
                f"--model {exp['model']}",
                f"--backbone {exp['backbone']}"
            ]
            
            # Add all other arguments
            for key, value in exp['args'].items():
                if isinstance(value, bool):
                    if value:
                        args_list.append(f"--{key}")
                elif isinstance(value, list):
                    args_list.append(f"--{key} {' '.join(map(str, value))}")
                else:
                    args_list.append(f"--{key} {value}")
            
            # Build full command
            if venv_path:
                cmd = f"cd {base_path} && source {venv_path} && {python_cmd} utils/main.py {' '.join(args_list)}"
            else:
                cmd = f"cd {base_path} && {python_cmd} utils/main.py {' '.join(args_list)}"
        
        # Add CUDA_VISIBLE_DEVICES if GPU specified
        if gpu_id is not None:
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
        
        return cmd
    
    def run_experiment(self, exp: Dict[str, Any], exp_idx: int, total_exps: int, gpu_id: int = None) -> Dict[str, Any]:
        """Run a single experiment."""
        # Generate experiment name
        if exp.get('traditional_ml', False):
            exp_name = f"{exp['model']}_{exp['dataset']}_seed{exp['seed']}"
        else:
            exp_name = f"{exp['model']}_{exp['backbone']}_{exp['dataset']}_seed{exp['seed']}"
        
        gpu_str = f" [GPU {gpu_id}]" if gpu_id is not None else ""
        print(f"\n{'='*80}")
        print(f"Experiment {exp_idx + 1}/{total_exps}: {exp_name}{gpu_str}")
        print(f"{'='*80}")
        
        cmd = self.build_command(exp, gpu_id)
        print(f"Command: {cmd}\n")
        
        if self.dry_run:
            print("[DRY RUN] Would execute the above command")
            return {
                'name': exp_name,
                'status': 'dry_run',
                'command': cmd,
                'gpu_id': gpu_id
            }
        
        # Run the experiment
        start_time = time.time()
        
        output_file = self.run_dir / f"{exp_name}.log"
        
        try:
            with open(output_file, 'w') as f:
                f.write(f"Command: {cmd}\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write("="*80 + "\n")
                f.flush()
                
                # Use bash to execute the command, streaming output to file
                result = subprocess.run(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            duration = time.time() - start_time
            
            # Append summary to the log
            with open(output_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Exit Code: {result.returncode}\n")
            
            success = result.returncode == 0
            
            # Print error details immediately if failed
            if not success:
                print(f"\n{'!'*80}")
                print(f"FAILED: {exp_name} (exit code: {result.returncode})")
                print(f"See log: {output_file}")
                print(f"{'!'*80}\n")
            
            return {
                'name': exp_name,
                'status': 'success' if success else 'failed',
                'duration': duration,
                'exit_code': result.returncode,
                'command': cmd,
                'gpu_id': gpu_id,
                'output_file': str(output_file)
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n{'!'*80}")
            print(f"EXCEPTION: {exp_name}")
            print(f"{'!'*80}")
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'!'*80}\n")
            
            return {
                'name': exp_name,
                'status': 'error',
                'duration': duration,
                'error': str(e),
                'command': cmd,
                'gpu_id': gpu_id
            }
    
    def _run_experiment_worker(self, args):
        """Worker function for parallel execution."""
        exp, exp_idx, total_exps, gpu_id = args
        return self.run_experiment(exp, exp_idx, total_exps, gpu_id)
    
    def run_all(self):
        """Run all experiments."""
        experiments = self.generate_experiments()
        total_exps = len(experiments)
        
        print(f"\n{'='*80}")
        print(f"Benchmark Runner")
        print(f"Config: {self.config_path}")
        print(f"Total experiments: {total_exps}")
        print(f"Parallel processes: {self.num_parallel}")
        print(f"Available GPUs: {self.gpus}")
        print(f"Dry run: {self.dry_run}")
        if not self.dry_run:
            print(f"Results directory: {self.run_dir}")
        print(f"{'='*80}\n")
        
        # Print experiment summary
        print("Experiments to run:")
        for i, exp in enumerate(experiments):
            if exp.get('traditional_ml', False):
                print(f"  {i+1}. {exp['model']} on {exp['dataset']} (seed={exp['seed']}) [Traditional ML]")
            else:
                print(f"  {i+1}. {exp['model']} + {exp['backbone']} on {exp['dataset']} (seed={exp['seed']})")
        print()
        
        if self.dry_run:
            print("DRY RUN MODE - No experiments will be executed\n")
        
        # Run experiments
        if self.num_parallel <= 1:
            # Sequential execution
            results = []
            for i, exp in enumerate(experiments):
                result = self.run_experiment(exp, i, total_exps)
                results.append(result)
        else:
            # Parallel execution
            print(f"Running {self.num_parallel} experiments in parallel...\n")
            
            # Assign GPUs to experiments in round-robin fashion
            experiment_args = []
            for i, exp in enumerate(experiments):
                gpu_id = self.gpus[i % len(self.gpus)]
                experiment_args.append((exp, i, total_exps, gpu_id))
            
            # Use multiprocessing Pool
            with Pool(processes=min(self.num_parallel, len(self.gpus))) as pool:
                results = pool.map(self._run_experiment_worker, experiment_args)
        
        # Save summary
        if not self.dry_run:
            summary_file = self.run_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    'config_path': self.config_path,
                    'total_experiments': total_exps,
                    'num_parallel': self.num_parallel,
                    'gpus': self.gpus,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        
        if self.dry_run:
            print(f"Dry run completed for {total_exps} experiments")
        else:
            success_count = sum(1 for r in results if r['status'] == 'success')
            failed_count = sum(1 for r in results if r['status'] == 'failed')
            error_count = sum(1 for r in results if r['status'] == 'error')
            
            print(f"Total experiments: {total_exps}")
            print(f"Successful: {success_count}")
            print(f"Failed: {failed_count}")
            print(f"Errors: {error_count}")
            print(f"\nResults saved to: {self.run_dir}")
            
            if failed_count > 0 or error_count > 0:
                print("\nFailed/Error experiments:")
                for r in results:
                    if r['status'] in ['failed', 'error']:
                        error_msg = r.get('error', f'exit code {r.get("exit_code", "unknown")}')
                        print(f"  - {r['name']}: {error_msg}")
                        if 'output_file' in r:
                            print(f"    Full output: {r['output_file']}")
        
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run CfC continual learning benchmarks from config files')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing them')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of experiments to run in parallel (default: 1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        return 1
    
    runner = BenchmarkRunner(args.config, dry_run=args.dry_run, num_parallel=args.parallel)
    runner.run_all()
    
    return 0


if __name__ == '__main__':
    exit(main())
