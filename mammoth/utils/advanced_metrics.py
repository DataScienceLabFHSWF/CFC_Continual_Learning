# Copyright 2024-present
# Advanced Metrics for Continual Learning Analysis

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import wandb
from copy import deepcopy


class RepresentationalStabilityMetric:
    """
    Measure how much hidden representations change after learning new tasks.
    
    Purpose: Test if stable neurons preserve representations.
    Metric: Cosine similarity between representations on Task A before/after Task B.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.reference_representations = {}  # task_id -> representations
        
    def extract_representations(self, model, dataloader, device='cuda') -> torch.Tensor:
        """
        Extract hidden representations from model.
        
        Args:
            model: The model
            dataloader: DataLoader for task data
            device: Device to run on
            
        Returns:
            Tensor of shape (num_samples, feature_dim)
        """
        model.eval()
        representations = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                # Get features (not logits)
                features = model(inputs, returnt='features')
                representations.append(features.cpu())
        
        return torch.cat(representations, dim=0)
    
    def compute_stability(self, repr_before: torch.Tensor, 
                         repr_after: torch.Tensor) -> Dict[str, float]:
        """
        Compute stability metrics between two representation sets.
        
        Args:
            repr_before: Representations before learning new task
            repr_after: Representations after learning new task
            
        Returns:
            Dictionary of stability metrics
        """
        # Cosine similarity
        cos_sim = nn.functional.cosine_similarity(repr_before, repr_after, dim=1)
        
        # L2 distance
        l2_dist = torch.norm(repr_after - repr_before, p=2, dim=1)
        
        # Relative change
        rel_change = l2_dist / (torch.norm(repr_before, p=2, dim=1) + 1e-8)
        
        return {
            'repr_cosine_sim_mean': float(cos_sim.mean()),
            'repr_cosine_sim_std': float(cos_sim.std()),
            'repr_l2_distance_mean': float(l2_dist.mean()),
            'repr_l2_distance_std': float(l2_dist.std()),
            'repr_relative_change_mean': float(rel_change.mean()),
            'repr_relative_change_std': float(rel_change.std()),
        }
    
    def update(self, model, task_id: int, dataloader, phase: str = 'before', 
              device='cuda', use_wandb: bool = True):
        """
        Update stability tracking.
        
        Args:
            model: The model
            task_id: Current task ID
            dataloader: DataLoader for the task
            phase: 'before' or 'after' learning new task
            device: Device
            use_wandb: Log to WandB
        """
        if not self.enabled:
            return
        
        key = f'task_{task_id}_{phase}'
        repr_current = self.extract_representations(model, dataloader, device)
        
        if phase == 'before':
            # Store reference
            self.reference_representations[key] = repr_current
        elif phase == 'after':
            # Compare with reference
            key_before = f'task_{task_id}_before'
            if key_before in self.reference_representations:
                repr_before = self.reference_representations[key_before]
                metrics = self.compute_stability(repr_before, repr_current)
                
                if use_wandb and wandb.run is not None:
                    wandb.log({f'{k}_task{task_id}': v for k, v in metrics.items()})
                else:
                    print(f"\n[Repr Stability] Task {task_id}:")
                    for k, v in metrics.items():
                        print(f"  {k}: {v:.4f}")


class WeightChangeAnalyzer:
    """
    Track weight changes per layer to identify which parts of the network change most.
    
    Purpose: Test if sparse connectivity protects certain weights.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.weight_snapshots = {}  # task_id -> {layer_name -> weights}
        
    def save_weights(self, model, task_id: int):
        """Save current weights as snapshot."""
        if not self.enabled:
            return
        
        snapshot = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                snapshot[name] = param.data.clone().cpu()
        
        self.weight_snapshots[task_id] = snapshot
    
    def compute_weight_change(self, weights_before: Dict[str, torch.Tensor],
                             weights_after: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute weight change metrics.
        
        Args:
            weights_before: Weights before task
            weights_after: Weights after task
            
        Returns:
            Dictionary of per-layer changes
        """
        changes = {}
        
        for name in weights_before.keys():
            if name in weights_after:
                w_before = weights_before[name]
                w_after = weights_after[name]
                
                # Frobenius norm of change
                frobenius = torch.norm(w_after - w_before, p='fro').item()
                
                # Relative change
                relative = frobenius / (torch.norm(w_before, p='fro').item() + 1e-8)
                
                # Layer name cleanup
                clean_name = name.replace('.', '_')
                changes[f'weight_change_frobenius_{clean_name}'] = frobenius
                changes[f'weight_change_relative_{clean_name}'] = relative
        
        return changes
    
    def analyze_task_transition(self, task_from: int, task_to: int, 
                                use_wandb: bool = True) -> Dict[str, float]:
        """
        Analyze weight changes between two tasks.
        
        Args:
            task_from: Starting task ID
            task_to: Ending task ID
            use_wandb: Log to WandB
            
        Returns:
            Dictionary of weight change metrics
        """
        if not self.enabled:
            return {}
        
        if task_from not in self.weight_snapshots or task_to not in self.weight_snapshots:
            return {}
        
        changes = self.compute_weight_change(
            self.weight_snapshots[task_from],
            self.weight_snapshots[task_to]
        )
        
        if use_wandb and wandb.run is not None:
            wandb.log({f'{k}_t{task_from}_to_t{task_to}': v for k, v in changes.items()})
        
        return changes


class GradientInterferenceAnalyzer:
    """
    Measure gradient conflict between tasks.
    
    Purpose: Test H3 (Gradient Isolation Hypothesis).
    Metric: Cosine similarity of gradients from different tasks.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.gradient_cache = {}  # task_id -> gradients
        
    def extract_gradients(self, model) -> Dict[str, torch.Tensor]:
        """Extract gradients from model."""
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().cpu()
        return gradients
    
    def flatten_gradients(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten all gradients into a single vector."""
        grad_list = [g.flatten() for g in gradients.values()]
        return torch.cat(grad_list)
    
    def compute_gradient_similarity(self, grads_a: Dict[str, torch.Tensor],
                                    grads_b: Dict[str, torch.Tensor]) -> float:
        """
        Compute cosine similarity between two gradient sets.
        
        Args:
            grads_a: Gradients from task A
            grads_b: Gradients from task B
            
        Returns:
            Cosine similarity (-1 to 1)
            Values near 0 = orthogonal (no interference)
            Values near -1 = conflicting (high interference)
        """
        flat_a = self.flatten_gradients(grads_a)
        flat_b = self.flatten_gradients(grads_b)
        
        cos_sim = nn.functional.cosine_similarity(flat_a.unsqueeze(0), 
                                                 flat_b.unsqueeze(0))
        return float(cos_sim)
    
    def cache_gradients(self, model, task_id: int):
        """Cache gradients for later comparison."""
        if not self.enabled:
            return
        
        gradients = self.extract_gradients(model)
        self.gradient_cache[task_id] = gradients
    
    def analyze_interference(self, task_pairs: List[Tuple[int, int]], 
                           use_wandb: bool = True) -> Dict[str, float]:
        """
        Analyze gradient interference between task pairs.
        
        Args:
            task_pairs: List of (task_a, task_b) pairs to compare
            use_wandb: Log to WandB
            
        Returns:
            Dictionary of interference metrics
        """
        if not self.enabled:
            return {}
        
        results = {}
        
        for task_a, task_b in task_pairs:
            if task_a in self.gradient_cache and task_b in self.gradient_cache:
                similarity = self.compute_gradient_similarity(
                    self.gradient_cache[task_a],
                    self.gradient_cache[task_b]
                )
                
                key = f'gradient_similarity_t{task_a}_t{task_b}'
                results[key] = similarity
        
        if use_wandb and wandb.run is not None:
            wandb.log(results)
        
        return results


class AdvancedMetricsManager:
    """
    Manager for all advanced metrics.
    Provides unified interface for tracking.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        
        self.repr_stability = RepresentationalStabilityMetric(
            enabled=config.get('representational_stability', {}).get('enabled', True)
        )
        
        self.weight_change = WeightChangeAnalyzer(
            enabled=config.get('weight_change', {}).get('enabled', True)
        )
        
        self.gradient_interference = GradientInterferenceAnalyzer(
            enabled=config.get('gradient_interference', {}).get('enabled', True)
        )
    
    def on_task_start(self, model, task_id: int, dataloader, device='cuda'):
        """Call at start of task."""
        self.repr_stability.update(model, task_id, dataloader, 'before', device)
        self.weight_change.save_weights(model, task_id)
    
    def on_task_end(self, model, task_id: int, dataloader, device='cuda'):
        """Call at end of task."""
        self.repr_stability.update(model, task_id, dataloader, 'after', device)
        self.weight_change.save_weights(model, task_id + 1)  # Save for next comparison
        
        if task_id > 0:
            self.weight_change.analyze_task_transition(task_id - 1, task_id)
    
    def on_backward(self, model, task_id: int):
        """Call after backward pass to cache gradients."""
        self.gradient_interference.cache_gradients(model, task_id)
    
    def analyze_all(self, task_pairs: Optional[List[Tuple[int, int]]] = None):
        """Run all analyses at the end."""
        if task_pairs:
            self.gradient_interference.analyze_interference(task_pairs)
