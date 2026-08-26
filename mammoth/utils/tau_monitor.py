# Copyright 2024-present
# Tau (Time Constant) Monitoring for LTC Networks

import torch
import numpy as np
from typing import Dict, Optional
import wandb


class TauMonitor:
    """
    Monitor and analyze tau (time constant) values in LTC networks.
    
    Purpose: Test H2 (Temporal Stability Hypothesis)
    - Track distribution of tau values
    - Identify fast vs. slow neurons
    - Measure stability across tasks
    """
    
    def __init__(self, enabled: bool = True, log_every_n_steps: int = 100):
        self.enabled = enabled
        self.log_every_n_steps = log_every_n_steps
        self.step_counter = 0
        
        # Historical tracking
        self.tau_history = []
        self.task_boundaries = []
        
    def extract_tau_values(self, model) -> Optional[torch.Tensor]:
        """
        Extract tau values from LTC model.
        
        Args:
            model: The backbone model (should have ltc attribute)
            
        Returns:
            Tensor of tau values or None if not available
        """
        if not self.enabled:
            return None
        
        # Search all named parameters for an ncps tau-like parameter. Only
        # CfC/LTC cells run in mode="pure" expose a literal per-neuron
        # `w_tau`; the default gated mode has no single time-constant
        # parameter (it uses input-dependent time_a/time_b gates instead).
        for name, param in model.named_parameters():
            if name.endswith('w_tau') or name.endswith('.tau'):
                return param.detach().cpu()
        
        # Legacy fallback paths (kept for backward compatibility).
        ltc_layer = None
        if hasattr(model, 'ltc'):
            ltc_layer = model.ltc
        elif hasattr(model, 'rnn') and hasattr(model.rnn, 'tau'):
            ltc_layer = model.rnn
        
        if ltc_layer is None:
            return None
        
        if hasattr(ltc_layer, 'tau'):
            return ltc_layer.tau.detach().cpu()
        elif hasattr(ltc_layer, '_wiring') and hasattr(ltc_layer._wiring, 'tau'):
            return ltc_layer._wiring.tau.detach().cpu()
        
        return None
    
    def compute_statistics(self, tau_values: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistical metrics for tau distribution.
        
        Args:
            tau_values: Tensor of tau values
            
        Returns:
            Dictionary of statistics
        """
        tau_np = tau_values.numpy()
        
        stats = {
            'tau_mean': float(np.mean(tau_np)),
            'tau_std': float(np.std(tau_np)),
            'tau_min': float(np.min(tau_np)),
            'tau_max': float(np.max(tau_np)),
            'tau_median': float(np.median(tau_np)),
            'tau_q25': float(np.percentile(tau_np, 25)),
            'tau_q75': float(np.percentile(tau_np, 75)),
        }
        
        # Bimodality coefficient (test for fast/slow split)
        # BC = (skew^2 + 1) / kurtosis
        # BC > 0.555 suggests bimodal distribution
        from scipy import stats as scipy_stats
        skew = scipy_stats.skew(tau_np)
        kurt = scipy_stats.kurtosis(tau_np)
        if kurt != 0:
            stats['tau_bimodality'] = float((skew**2 + 1) / kurt)
        else:
            stats['tau_bimodality'] = 0.0
        
        # Count fast vs slow neurons (threshold at median)
        stats['tau_fast_count'] = int(np.sum(tau_np < stats['tau_median']))
        stats['tau_slow_count'] = int(np.sum(tau_np >= stats['tau_median']))
        
        return stats
    
    def log_tau_distribution(self, tau_values: torch.Tensor, task_id: int, 
                            epoch: int, use_wandb: bool = True):
        """
        Log tau distribution to WandB or console.
        
        Args:
            tau_values: Tensor of tau values
            task_id: Current task ID
            epoch: Current epoch
            use_wandb: Whether to log to WandB
        """
        stats = self.compute_statistics(tau_values)
        
        # Add context
        stats['task_id'] = task_id
        stats['epoch'] = epoch
        
        if use_wandb and wandb.run is not None:
            # Log statistics
            wandb.log(stats)
            
            # Log histogram
            wandb.log({
                'tau_distribution': wandb.Histogram(tau_values.numpy()),
                'task_id': task_id,
                'epoch': epoch
            })
        else:
            # Console logging
            print(f"\n[Tau Monitor] Task {task_id}, Epoch {epoch}")
            print(f"  Mean: {stats['tau_mean']:.4f} ± {stats['tau_std']:.4f}")
            print(f"  Range: [{stats['tau_min']:.4f}, {stats['tau_max']:.4f}]")
            print(f"  Bimodality: {stats['tau_bimodality']:.4f} "
                  f"({'bimodal' if stats['tau_bimodality'] > 0.555 else 'unimodal'})")
            print(f"  Fast/Slow: {stats['tau_fast_count']}/{stats['tau_slow_count']}")
    
    def update(self, model, task_id: int, epoch: int, use_wandb: bool = True):
        """
        Update tau monitoring (call this during training).
        
        Args:
            model: The model being trained
            task_id: Current task ID
            epoch: Current epoch
            use_wandb: Whether to log to WandB
        """
        if not self.enabled:
            return
        
        self.step_counter += 1
        
        if self.step_counter % self.log_every_n_steps != 0:
            return
        
        tau_values = self.extract_tau_values(model)
        if tau_values is not None:
            self.log_tau_distribution(tau_values, task_id, epoch, use_wandb)
            self.tau_history.append({
                'task_id': task_id,
                'epoch': epoch,
                'tau_values': tau_values.clone()
            })
    
    def on_task_end(self, task_id: int):
        """Mark task boundary for analysis."""
        self.task_boundaries.append(task_id)
    
    def analyze_stability(self) -> Dict[str, float]:
        """
        Analyze how tau values change across tasks.
        
        Returns:
            Dictionary of stability metrics
        """
        if len(self.tau_history) < 2:
            return {}
        
        # Compare tau values at task boundaries
        task_tau = {}
        for entry in self.tau_history:
            task_id = entry['task_id']
            if task_id not in task_tau:
                task_tau[task_id] = []
            task_tau[task_id].append(entry['tau_values'])
        
        # Compute tau stability (correlation between tasks)
        stability = {}
        tasks = sorted(task_tau.keys())
        for i in range(len(tasks) - 1):
            task_a = tasks[i]
            task_b = tasks[i + 1]
            
            # Use last tau from task A and first tau from task B
            tau_a = task_tau[task_a][-1].numpy()
            tau_b = task_tau[task_b][0].numpy()
            
            # Compute correlation
            correlation = np.corrcoef(tau_a, tau_b)[0, 1]
            stability[f'tau_stability_{task_a}_to_{task_b}'] = float(correlation)
        
        return stability


# Global instance for easy access
_tau_monitor = None

def get_tau_monitor(enabled: bool = True, log_every_n_steps: int = 100) -> TauMonitor:
    """Get or create global tau monitor instance."""
    global _tau_monitor
    if _tau_monitor is None:
        _tau_monitor = TauMonitor(enabled, log_every_n_steps)
    return _tau_monitor
