#!/usr/bin/env python3
"""
Tennessee Eastman Process: Incremental Fault Learning Experiment

This script demonstrates continual learning on industrial fault detection:
1. Incremental learning: Learn faults one-by-one (Normal → F1 → F2 → ... → F21)
2. Joint learning: Train on all faults at once (upper bound)

Models compared:
- CfC with NCP wiring (our approach)
- CfC fully-connected (ablation)
- LSTM (standard baseline)

Hypothesis: CfC's bounded dynamics and sparse wiring will show:
- Less catastrophic forgetting when learning new faults
- Faster convergence toward joint training performance
- More robust fault detection over time
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add mammoth directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mammoth'))

from backbone.TEPcfc import TEPCfC, TEPLSTM
from datasets.tennessee_eastman import TennesseeEastmanDataset


class IncrementalFaultLearner:
    """
    Incremental fault detection learner.
    Learns faults one at a time and tracks forgetting.
    """
    
    def __init__(self, model, device='cuda', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        
        # Track metrics per fault
        self.fault_accuracies = {i: [] for i in range(22)}  # Accuracy after each task
        self.forgetting_scores = {i: [] for i in range(22)}  # Forgetting measure
        
    def train_on_fault(self, fault_id, data_path, epochs=10, batch_size=32):
        """Train on a single fault."""
        print(f"\n{'='*60}")
        print(f"Training on Fault {fault_id}")
        print(f"{'='*60}")
        
        # Create dataset for this fault only
        train_dataset = TennesseeEastmanDataset(
            data_path=data_path,
            fault_ids=[fault_id],
            train=True,
            window_size=50,
            stride=10
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Optimizer for this task
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x, batch_y, _ in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Metrics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            epoch_acc = 100. * correct / total
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={epoch_acc:.2f}%")
        
        # Reset hidden state after training task
        if hasattr(self.model, 'reset_hidden'):
            self.model.reset_hidden()
    
    def evaluate_all_faults(self, data_path, faults_seen, batch_size=32):
        """
        Evaluate on all faults seen so far.
        Returns per-fault accuracies.
        """
        self.model.eval()
        fault_correct = {i: 0 for i in faults_seen}
        fault_total = {i: 0 for i in faults_seen}
        
        # Evaluate on each fault separately
        for fault_id in faults_seen:
            test_dataset = TennesseeEastmanDataset(
                data_path=data_path,
                fault_ids=[fault_id],
                train=False,
                window_size=50,
                stride=10
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
            
            with torch.no_grad():
                for batch_x, batch_y, _ in test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    logits = self.model(batch_x)
                    _, predicted = logits.max(1)
                    
                    fault_total[fault_id] += batch_y.size(0)
                    fault_correct[fault_id] += predicted.eq(batch_y).sum().item()
        
        # Compute accuracies
        fault_accs = {}
        for fault_id in faults_seen:
            if fault_total[fault_id] > 0:
                fault_accs[fault_id] = 100. * fault_correct[fault_id] / fault_total[fault_id]
            else:
                fault_accs[fault_id] = 0.0
        
        return fault_accs
    
    def incremental_training(self, data_path, num_faults=22, epochs_per_fault=10):
        """
        Incrementally train on faults 0 → 1 → 2 → ... → num_faults-1.
        Track accuracy and forgetting after each fault.
        """
        print(f"\n{'='*60}")
        print(f"INCREMENTAL FAULT LEARNING")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total faults: {num_faults}")
        print(f"{'='*60}\n")
        
        for fault_id in range(num_faults):
            # Train on this fault
            self.train_on_fault(fault_id, data_path, epochs=epochs_per_fault)
            
            # Evaluate on all faults seen so far
            faults_seen = list(range(fault_id + 1))
            fault_accs = self.evaluate_all_faults(data_path, faults_seen)
            
            # Store accuracies
            for fid in faults_seen:
                self.fault_accuracies[fid].append(fault_accs[fid])
            
            # Compute forgetting
            for fid in range(fault_id):  # All faults except current
                # Forgetting = max accuracy - current accuracy
                max_acc = max(self.fault_accuracies[fid])
                current_acc = self.fault_accuracies[fid][-1]
                forgetting = max_acc - current_acc
                self.forgetting_scores[fid].append(forgetting)
            
            # Print current status
            print(f"\nAfter training Fault {fault_id}:")
            print(f"Accuracies: {fault_accs}")
            avg_acc = np.mean([fault_accs[fid] for fid in faults_seen])
            print(f"Average accuracy: {avg_acc:.2f}%")
            
            if fault_id > 0:
                avg_forgetting = np.mean([self.forgetting_scores[fid][-1] for fid in range(fault_id)])
                print(f"Average forgetting: {avg_forgetting:.2f}%")
        
        return self.fault_accuracies, self.forgetting_scores


class JointFaultLearner:
    """
    Joint fault detection learner (upper bound).
    Trains on all faults simultaneously.
    """
    
    def __init__(self, model, device='cuda', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        
    def train_joint(self, data_path, epochs=50, batch_size=32):
        """Train on all faults jointly."""
        print(f"\n{'='*60}")
        print(f"JOINT TRAINING ON ALL FAULTS")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"{'='*60}\n")
        
        # Create dataset with all faults
        train_dataset = TennesseeEastmanDataset(
            data_path=data_path,
            fault_ids=None,  # All faults
            train=True,
            window_size=50,
            stride=10
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_acc = 0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x, batch_y, _ in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            epoch_acc = 100. * correct / total
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={epoch_acc:.2f}%")
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
        
        print(f"\nBest training accuracy: {best_acc:.2f}%")
        return best_acc
    
    def evaluate_all(self, data_path, batch_size=32):
        """Evaluate on all faults."""
        self.model.eval()
        
        test_dataset = TennesseeEastmanDataset(
            data_path=data_path,
            fault_ids=None,
            train=False,
            window_size=50,
            stride=10
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        fault_correct = {i: 0 for i in range(22)}
        fault_total = {i: 0 for i in range(22)}
        
        with torch.no_grad():
            for batch_x, batch_y, _ in tqdm(test_loader, desc="Evaluating"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = self.model(batch_x)
                _, predicted = logits.max(1)
                
                for i in range(batch_y.size(0)):
                    fault_id = batch_y[i].item()
                    fault_total[fault_id] += 1
                    if predicted[i] == batch_y[i]:
                        fault_correct[fault_id] += 1
        
        # Compute per-fault accuracies
        fault_accs = {}
        for fault_id in range(22):
            if fault_total[fault_id] > 0:
                fault_accs[fault_id] = 100. * fault_correct[fault_id] / fault_total[fault_id]
            else:
                fault_accs[fault_id] = 0.0
        
        avg_acc = np.mean(list(fault_accs.values()))
        print(f"\nPer-fault accuracies: {fault_accs}")
        print(f"Average accuracy: {avg_acc:.2f}%")
        
        return fault_accs, avg_acc


def plot_results(incremental_accs, joint_accs, model_name, save_path):
    """
    Plot incremental learning progress vs joint training upper bound.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy matrix (heatmap)
    ax = axes[0, 0]
    num_faults = len(incremental_accs)
    acc_matrix = np.zeros((num_faults, num_faults))
    for fault_id in range(num_faults):
        accs = incremental_accs[fault_id]
        acc_matrix[fault_id, :len(accs)] = accs
    
    sns.heatmap(acc_matrix, annot=False, cmap='RdYlGn', vmin=0, vmax=100,
                ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_xlabel('Task (Fault Learned)')
    ax.set_ylabel('Fault ID')
    ax.set_title(f'{model_name}: Accuracy Matrix\n(Row i = Fault i accuracy over time)')
    
    # Plot 2: Average accuracy over tasks
    ax = axes[0, 1]
    avg_accs = []
    for task in range(num_faults):
        faults_seen = list(range(task + 1))
        task_accs = [incremental_accs[fid][task] for fid in faults_seen]
        avg_accs.append(np.mean(task_accs))
    
    ax.plot(range(num_faults), avg_accs, marker='o', label='Incremental', linewidth=2)
    ax.axhline(y=np.mean(list(joint_accs.values())), color='r', linestyle='--',
               label='Joint Training (Upper Bound)', linewidth=2)
    ax.set_xlabel('Number of Faults Learned')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title(f'{model_name}: Convergence to Joint Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Forgetting over time
    ax = axes[1, 0]
    forgetting_matrix = np.zeros((num_faults, num_faults))
    for fault_id in range(num_faults):
        for task_idx, accs in enumerate(incremental_accs[fault_id]):
            if task_idx > 0:
                max_acc = max(incremental_accs[fault_id][:task_idx+1])
                forgetting = max_acc - accs
                forgetting_matrix[fault_id, task_idx] = forgetting
    
    sns.heatmap(forgetting_matrix, annot=False, cmap='RdYlGn_r', vmin=0, vmax=50,
                ax=ax, cbar_kws={'label': 'Forgetting (%)'})
    ax.set_xlabel('Task (Fault Learned)')
    ax.set_ylabel('Fault ID')
    ax.set_title(f'{model_name}: Forgetting Matrix')
    
    # Plot 4: Per-fault final accuracy comparison
    ax = axes[1, 1]
    fault_ids = list(range(num_faults))
    incremental_final = [incremental_accs[fid][-1] for fid in fault_ids]
    joint_final = [joint_accs[fid] for fid in fault_ids]
    
    x = np.arange(len(fault_ids))
    width = 0.35
    ax.bar(x - width/2, incremental_final, width, label='Incremental', alpha=0.8)
    ax.bar(x + width/2, joint_final, width, label='Joint', alpha=0.8)
    ax.set_xlabel('Fault ID')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model_name}: Final Accuracy per Fault')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='TEP Incremental Fault Learning')
    parser.add_argument('--data_path', type=str, default='./data/TEP',
                        help='Path to TEP dataset')
    parser.add_argument('--model', type=str, default='cfc-ncp',
                        choices=['cfc-ncp', 'cfc-full', 'lstm'],
                        help='Model to use')
    parser.add_argument('--epochs_per_fault', type=int, default=10,
                        help='Training epochs per fault (incremental)')
    parser.add_argument('--joint_epochs', type=int, default=50,
                        help='Training epochs for joint model')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size for models')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_faults', type=int, default=22,
                        help='Number of faults to learn (max 22)')
    parser.add_argument('--output_dir', type=str, default='./results/tep',
                        help='Output directory for results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    print(f"\nCreating models...")
    if args.model == 'cfc-ncp':
        incremental_model = TEPCfC(hidden_size=args.hidden_size, use_ncp_wiring=True)
        joint_model = TEPCfC(hidden_size=args.hidden_size, use_ncp_wiring=True)
        model_name = "CfC-NCP"
    elif args.model == 'cfc-full':
        incremental_model = TEPCfC(hidden_size=args.hidden_size, use_ncp_wiring=False)
        joint_model = TEPCfC(hidden_size=args.hidden_size, use_ncp_wiring=False)
        model_name = "CfC-Full"
    else:  # lstm
        incremental_model = TEPLSTM(hidden_size=args.hidden_size)
        joint_model = TEPLSTM(hidden_size=args.hidden_size)
        model_name = "LSTM"
    
    print(f"Model: {model_name}")
    print(f"Parameters (incremental): {incremental_model.get_params():,}")
    print(f"Parameters (joint): {joint_model.get_params():,}")
    
    # Incremental learning
    print(f"\n{'#'*60}")
    print(f"# INCREMENTAL LEARNING")
    print(f"{'#'*60}")
    incremental_learner = IncrementalFaultLearner(incremental_model, device=device, lr=args.lr)
    inc_accs, inc_forgetting = incremental_learner.incremental_training(
        args.data_path,
        num_faults=args.num_faults,
        epochs_per_fault=args.epochs_per_fault
    )
    
    # Joint learning
    print(f"\n{'#'*60}")
    print(f"# JOINT LEARNING (UPPER BOUND)")
    print(f"{'#'*60}")
    joint_learner = JointFaultLearner(joint_model, device=device, lr=args.lr)
    joint_learner.train_joint(args.data_path, epochs=args.joint_epochs)
    joint_accs, joint_avg = joint_learner.evaluate_all(args.data_path)
    
    # Plot results
    plot_path = os.path.join(args.output_dir, f'tep_{args.model}_results.png')
    plot_results(inc_accs, joint_accs, model_name, plot_path)
    
    # Save numerical results
    results = {
        'model': model_name,
        'incremental_accuracies': inc_accs,
        'incremental_forgetting': inc_forgetting,
        'joint_accuracies': joint_accs,
        'joint_average': joint_avg
    }
    
    import pickle
    results_path = os.path.join(args.output_dir, f'tep_{args.model}_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    final_avg_inc = np.mean([inc_accs[i][-1] for i in range(args.num_faults)])
    avg_forgetting = np.mean([max(inc_forgetting[i]) if inc_forgetting[i] else 0 
                              for i in range(args.num_faults-1)])
    print(f"Incremental final average: {final_avg_inc:.2f}%")
    print(f"Joint average: {joint_avg:.2f}%")
    print(f"Gap: {joint_avg - final_avg_inc:.2f}%")
    print(f"Average forgetting: {avg_forgetting:.2f}%")


if __name__ == '__main__':
    main()
