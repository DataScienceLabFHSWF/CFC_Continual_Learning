#!/usr/bin/env python3
"""
CfC Interpretability Analysis for Continual Learning

This script analyzes:
1. Wiring structure visualization
2. Neuron activation patterns across tasks
3. Task-critical pathway identification
4. Feature importance over time

For paper: "Interpretable Continual Learning with Closed-form Continuous-time Networks"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
sys.path.insert(0, 'mammoth')

from backbone.MNISTcfc import BaseMNISTcfc
from backbone.TEPcfc import BaseTEPCfC
from datasets import get_dataset
from models import get_model
import json


class CfCInterpreter:
    """Interpretability analysis for CfC networks."""
    
    def __init__(self, model, dataset, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture activations."""
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach().cpu()
            return hook
        
        # Hook into CfC layers
        if hasattr(self.model.net, 'rnn'):
            h = self.model.net.rnn.register_forward_hook(get_activation('cfc_output'))
            self.hooks.append(h)
        
        if hasattr(self.model.net, 'cfc'):
            h = self.model.net.cfc.register_forward_hook(get_activation('cfc_output'))
            self.hooks.append(h)
            
        # Hook into classifier
        if hasattr(self.model.net, 'classifier'):
            h = self.model.net.classifier.register_forward_hook(get_activation('classifier'))
            self.hooks.append(h)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def extract_activations_per_task(self, num_samples=100):
        """Extract neuron activations for each task."""
        self.model.eval()
        task_activations = {}
        
        with torch.no_grad():
            for task_id in range(self.dataset.N_TASKS):
                _, test_loader = self.dataset.get_data_loaders()
                
                batch_activations = []
                samples_collected = 0
                
                for x, y, t in test_loader:
                    if samples_collected >= num_samples:
                        break
                    
                    # Filter by task
                    task_mask = (t == task_id)
                    if not task_mask.any():
                        continue
                    
                    x_task = x[task_mask].to(self.device)
                    
                    # Forward pass
                    _ = self.model(x_task)
                    
                    # Collect activations
                    if 'cfc_output' in self.activations:
                        batch_activations.append(self.activations['cfc_output'])
                    
                    samples_collected += x_task.size(0)
                
                if batch_activations:
                    task_activations[task_id] = torch.cat(batch_activations, dim=0)
        
        return task_activations
    
    def visualize_wiring(self, output_path='figures/cfc_wiring.pdf'):
        """Visualize NCP wiring structure."""
        # Extract wiring from model
        if not hasattr(self.model.net, 'rnn'):
            print("Model doesn't have RNN layer - skipping wiring visualization")
            return
        
        rnn = self.model.net.rnn
        if not hasattr(rnn, '_wiring'):
            print("No wiring structure found")
            return
        
        wiring = rnn._wiring
        
        # Create adjacency matrix
        adjacency = wiring.adjacency_matrix.cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot adjacency matrix
        im = axes[0].imshow(adjacency, cmap='Blues', aspect='auto')
        axes[0].set_title('CfC Wiring Structure (Adjacency Matrix)')
        axes[0].set_xlabel('Source Neuron')
        axes[0].set_ylabel('Target Neuron')
        plt.colorbar(im, ax=axes[0])
        
        # Plot sparsity pattern
        sparsity = (adjacency == 0).mean()
        density = 1 - sparsity
        
        # Neuron type distribution
        if hasattr(wiring, 'neuron_types'):
            types = wiring.neuron_types
            type_names = ['Sensory', 'Inter', 'Command', 'Motor']
            type_counts = [np.sum(types == i) for i in range(4)]
            
            axes[1].bar(type_names, type_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[1].set_title(f'Neuron Distribution (Sparsity: {sparsity:.1%})')
            axes[1].set_ylabel('Number of Neurons')
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Wiring visualization saved to {output_path}")
        plt.close()
    
    def visualize_activation_patterns(self, task_activations, output_path='figures/activation_patterns.pdf'):
        """Visualize neuron activation patterns across tasks."""
        num_tasks = len(task_activations)
        
        fig, axes = plt.subplots(1, num_tasks, figsize=(4*num_tasks, 4))
        if num_tasks == 1:
            axes = [axes]
        
        for task_id, ax in enumerate(axes):
            if task_id not in task_activations:
                continue
            
            activations = task_activations[task_id]
            
            # Average activations across samples
            if activations.dim() == 3:  # (samples, seq_len, hidden)
                activations = activations[:, -1, :]  # Take last timestep
            
            mean_act = activations.mean(dim=0).numpy()
            
            # Plot as heatmap
            im = ax.imshow(mean_act.reshape(-1, 1), cmap='viridis', aspect='auto')
            ax.set_title(f'Task {task_id}')
            ax.set_xlabel('Neuron Index')
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('CfC Neuron Activations Across Tasks')
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Activation patterns saved to {output_path}")
        plt.close()
    
    def identify_task_critical_neurons(self, task_activations, threshold=0.8):
        """Identify neurons that are critical for each task."""
        num_tasks = len(task_activations)
        
        # Compute activation magnitudes
        task_neuron_importance = {}
        
        for task_id, activations in task_activations.items():
            if activations.dim() == 3:
                activations = activations[:, -1, :]
            
            # Mean absolute activation per neuron
            importance = activations.abs().mean(dim=0).numpy()
            task_neuron_importance[task_id] = importance
        
        # Find task-specific neurons
        critical_neurons = {}
        all_importance = np.stack([task_neuron_importance[i] for i in range(num_tasks)])
        
        for task_id in range(num_tasks):
            # Neurons with high activation for this task
            task_imp = all_importance[task_id]
            other_imp = np.delete(all_importance, task_id, axis=0).max(axis=0)
            
            # Selectivity: high for this task, low for others
            selectivity = task_imp / (other_imp + 1e-8)
            
            # Top selective neurons
            top_k = int(len(selectivity) * 0.1)  # Top 10%
            critical_idx = np.argsort(selectivity)[-top_k:]
            critical_neurons[task_id] = critical_idx.tolist()
        
        return critical_neurons, task_neuron_importance
    
    def visualize_critical_pathways(self, critical_neurons, output_path='figures/critical_pathways.pdf'):
        """Visualize task-critical neural pathways."""
        num_tasks = len(critical_neurons)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create task-neuron matrix
        all_neurons = set()
        for neurons in critical_neurons.values():
            all_neurons.update(neurons)
        
        neuron_list = sorted(all_neurons)
        task_neuron_matrix = np.zeros((num_tasks, len(neuron_list)))
        
        for task_id, neurons in critical_neurons.items():
            for neuron_id in neurons:
                if neuron_id in neuron_list:
                    idx = neuron_list.index(neuron_id)
                    task_neuron_matrix[task_id, idx] = 1
        
        # Plot
        im = ax.imshow(task_neuron_matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Task ID')
        ax.set_title('Task-Critical Neurons')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Critical (1) / Non-critical (0)')
        
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Critical pathways saved to {output_path}")
        plt.close()
    
    def generate_report(self, output_path='results/interpretability_report.json'):
        """Generate comprehensive interpretability report."""
        report = {
            'model_type': type(self.model.net).__name__,
            'dataset': type(self.dataset).__name__,
            'num_tasks': self.dataset.N_TASKS,
        }
        
        # Extract activations
        print("Extracting activations...")
        self.register_hooks()
        task_activations = self.extract_activations_per_task(num_samples=100)
        self.remove_hooks()
        
        # Identify critical neurons
        print("Identifying critical neurons...")
        critical_neurons, neuron_importance = self.identify_task_critical_neurons(task_activations)
        
        report['critical_neurons'] = {int(k): [int(n) for n in v] for k, v in critical_neurons.items()}
        report['neuron_importance'] = {int(k): v.tolist() for k, v in neuron_importance.items()}
        
        # Compute overlap statistics
        overlaps = []
        for i in range(len(critical_neurons)):
            for j in range(i+1, len(critical_neurons)):
                overlap = len(set(critical_neurons[i]) & set(critical_neurons[j]))
                overlaps.append(overlap)
        
        report['avg_neuron_overlap'] = float(np.mean(overlaps)) if overlaps else 0
        report['neuron_specialization'] = 1 - (np.mean(overlaps) / len(critical_neurons[0])) if overlaps else 1
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to {output_path}")
        
        # Generate visualizations
        print("Generating visualizations...")
        self.visualize_wiring()
        self.visualize_activation_patterns(task_activations)
        self.visualize_critical_pathways(critical_neurons)
        
        return report


def main():
    parser = argparse.ArgumentParser(description='CfC Interpretability Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='seq-mnist', help='Dataset name')
    parser.add_argument('--model', type=str, default='sgd', help='Model name')
    parser.add_argument('--backbone', type=str, default='mnistcfc', help='Backbone name')
    parser.add_argument('--output_dir', type=str, default='interpretability_results', help='Output directory')
    args = parser.parse_args()
    
    # Add missing args needed by Mammoth
    args.input_size = 784
    args.output_size = 10
    args.joint = 0
    args.batch_size = 32
    args.num_workers = 0
    args.base_path = './data/'
    args.custom_task_order = None
    args.custom_class_order = None
    args.validation = 0
    args.label_perc_by_task = 1.0
    args.label_perc_by_class = 1.0
    args.noise_type = None
    args.noise_rate = 0.0
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and dataset
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    
    # Initialize dataset
    dataset = get_dataset(args)
    
    # Initialize model
    model = get_model(args, None, dataset)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run interpretability analysis
    interpreter = CfCInterpreter(model, dataset)
    report = interpreter.generate_report(output_path=f'{args.output_dir}/report.json')
    
    print("\n" + "="*60)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("="*60)
    print(f"Neuron Specialization: {report['neuron_specialization']:.3f}")
    print(f"Average Neuron Overlap: {report['avg_neuron_overlap']:.1f}")
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
