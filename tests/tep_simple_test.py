#!/usr/bin/env python3
"""
Standalone TEP experiment (no mammoth framework dependencies).
Tests incremental vs joint training for fault detection.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimpleTEPDataset(Dataset):
    """Simple TEP dataset without mammoth dependencies."""
    
    def __init__(self, data_path, fault_ids, window_size=50, stride=10, train=True):
        self.window_size = window_size
        self.stride = stride
        
        # Load data for specified faults
        all_data = []
        all_labels = []
        
        for fault_id in fault_ids:
            # Load normal (d00) and fault data
            suffix = ".dat" if train else "_te.dat"
            
            # Normal operation
            normal_file = os.path.join(data_path, f"d00{suffix}")
            normal_data = np.loadtxt(normal_file)
            # Training: d00.dat is (52, 500) -> transpose to (500, 52)
            # Test: d00_te.dat is already (960, 52)
            if train:
                normal_data = normal_data.T
            
            # Fault data
            fault_file = os.path.join(data_path, f"d{fault_id:02d}{suffix}")
            fault_data = np.loadtxt(fault_file)
            # Training fault files:
            #   - d01.dat onwards are ALREADY (samples, 52) - NO transpose!
            # Test: already (samples, 52)
            # Only transpose if first dimension is 52
            if fault_data.shape[1] != 52:
                fault_data = fault_data.T
            
            # Create sliding windows
            normal_windows = self._create_windows(normal_data)
            fault_windows = self._create_windows(fault_data)
            
            # Labels: 0 for normal, fault_id for fault
            all_data.extend(normal_windows)
            all_labels.extend([0] * len(normal_windows))
            all_data.extend(fault_windows)
            all_labels.extend([fault_id] * len(fault_windows))
        
        self.data = np.array(all_data, dtype=np.float32)
        self.labels = np.array(all_labels, dtype=np.int64)
        
        # Z-score normalization
        self.mean = self.data.mean(axis=(0, 1))
        self.std = self.data.std(axis=(0, 1)) + 1e-8
        self.data = (self.data - self.mean) / self.std
    
    def _create_windows(self, data):
        """Create sliding windows from time series data."""
        windows = []
        max_start = len(data) - self.window_size
        if max_start < 0:
            # Not enough data for even one window, skip
            return windows
        for i in range(0, max_start + 1, self.stride):
            window = data[i:i + self.window_size]
            windows.append(window)
        return windows
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleLSTM(nn.Module):
    """Simple LSTM for TEP fault detection."""
    
    def __init__(self, input_size=52, num_classes=22, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


def train_model(model, train_loader, device, epochs=10, lr=0.001):
    """Train model on data."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%")


def evaluate_model(model, test_loader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/TEP')
    parser.add_argument('--num_faults', type=int, default=5)
    parser.add_argument('--epochs_per_fault', type=int, default=5)
    parser.add_argument('--joint_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"\n{'='*60}")
        print(f"ERROR: TEP dataset not found at {args.data_path}")
        print(f"Please run: ./download_tep.sh")
        print(f"{'='*60}\n")
        return
    
    # Incremental learning
    print(f"\n{'='*60}")
    print(f"INCREMENTAL LEARNING")
    print(f"{'='*60}\n")
    
    incremental_model = SimpleLSTM().to(device)
    incremental_accs = []
    
    for fault_id in range(args.num_faults):
        print(f"\n--- Training on Fault {fault_id} ---")
        
        train_dataset = SimpleTEPDataset(args.data_path, [fault_id], train=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        train_model(incremental_model, train_loader, device, epochs=args.epochs_per_fault)
        
        # Evaluate on all faults seen so far
        test_dataset = SimpleTEPDataset(args.data_path, list(range(fault_id + 1)), train=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        acc = evaluate_model(incremental_model, test_loader, device)
        incremental_accs.append(acc)
        print(f"Accuracy on faults 0-{fault_id}: {acc:.2f}%")
    
    final_incremental_acc = incremental_accs[-1]
    
    # Joint learning
    print(f"\n{'='*60}")
    print(f"JOINT LEARNING (UPPER BOUND)")
    print(f"{'='*60}\n")
    
    joint_model = SimpleLSTM().to(device)
    
    train_dataset = SimpleTEPDataset(args.data_path, list(range(args.num_faults)), train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    train_model(joint_model, train_loader, device, epochs=args.joint_epochs)
    
    test_dataset = SimpleTEPDataset(args.data_path, list(range(args.num_faults)), train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    joint_acc = evaluate_model(joint_model, test_loader, device)
    
    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Incremental final accuracy: {final_incremental_acc:.2f}%")
    print(f"Joint training accuracy:    {joint_acc:.2f}%")
    print(f"Gap (cost of incremental):  {joint_acc - final_incremental_acc:.2f}%")
    print(f"Convergence ratio:          {final_incremental_acc / joint_acc:.3f}")
    print(f"{'='*60}\n")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.num_faults), incremental_accs, 'o-', label='Incremental', linewidth=2)
    plt.axhline(y=joint_acc, color='r', linestyle='--', label='Joint (upper bound)', linewidth=2)
    plt.xlabel('Number of Faults Learned')
    plt.ylabel('Accuracy (%)')
    plt.title('Incremental vs Joint Learning on TEP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/tep_simple_test.png', dpi=150)
    print(f"Saved plot to ./results/tep_simple_test.png")


if __name__ == '__main__':
    main()
