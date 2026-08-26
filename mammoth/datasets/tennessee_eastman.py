# Copyright 2024-present
# Tennessee Eastman Process dataset for continual fault detection learning

import os
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from backbone.TEPcfc import BaseTEPCfC
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils import set_default_from_args
from utils.conf import base_path


class TennesseeEastmanDataset(Dataset):
    """
    Tennessee Eastman Process dataset for fault detection.
    
    The TEP dataset contains:
    - Normal operation (Fault 0)
    - 21 different fault types (Faults 1-21)
    - 52 process variables (41 measured + 11 manipulated)
    - Time-series data with 500 samples per run
    
    For continual learning:
    - Each fault type is a separate task
    - Model learns incrementally: Normal → Fault1 → Fault2 → ... → Fault21
    - Goal: Learn new faults without forgetting detection of old faults
    """
    
    def __init__(self, data_path, fault_ids=None, train=True, window_size=50, stride=10,
                 norm_mean=None, norm_std=None):
        """
        Args:
            data_path: Path to TEP dataset files
            fault_ids: List of fault IDs to include (0-21). If None, include all.
            train: Whether this is training or test data
            window_size: Number of timesteps in each sequence window
            stride: Stride for sliding window
            norm_mean, norm_std: Fixed per-channel normalization statistics
                (shape (52,)), computed once from normal-operation (fault 0)
                training data and shared across all fault files/splits. If
                None, they are computed here from fault 0's training file.
        """
        self.data_path = data_path
        self.fault_ids = fault_ids if fault_ids is not None else list(range(22))
        self.train = train
        self.window_size = window_size
        self.stride = stride
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
        # Load data
        self.windows = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess TEP data."""
        mode = 'train' if self.train else 'test'

        # Fix: compute normalization stats ONCE from normal-operation (fault 0)
        # training data, and reuse them for every fault file and split. The
        # previous implementation normalized each fault's file independently
        # (per-file z-score), which erases the very mean-shift signal that
        # distinguishes a fault from normal operation -- every file ends up
        # with mean 0 / std 1 per channel regardless of the fault, so no
        # backbone (including the Joint upper bound) can discriminate classes.
        if self.norm_mean is None or self.norm_std is None:
            ref_path = os.path.join(self.data_path, 'd00.dat')
            ref_data = np.loadtxt(ref_path)
            if ref_data.shape[0] == 52:
                ref_data = ref_data.T
            self.norm_mean = ref_data.mean(axis=0)
            self.norm_std = ref_data.std(axis=0) + 1e-8

        for fault_id in self.fault_ids:
            # TEP dataset files are typically named: d{fault_id}.dat or d{fault_id}_te.dat
            if self.train:
                filename = f'd{fault_id:02d}.dat'
            else:
                filename = f'd{fault_id:02d}_te.dat'
            
            filepath = os.path.join(self.data_path, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found, skipping fault {fault_id}")
                continue
            
            # Load data (typically space-separated values)
            try:
                data = np.loadtxt(filepath)
                
                # TEP data format: (52, timesteps) - transpose to (timesteps, 52)
                if data.shape[0] == 52:
                    data = data.T
                
                # Normalize with the SHARED reference stats (not per-file stats),
                # so that fault-induced mean/variance shifts remain visible.
                data = (data - self.norm_mean) / self.norm_std
                
                # Create sliding windows
                num_windows = (len(data) - self.window_size) // self.stride + 1
                for i in range(num_windows):
                    start_idx = i * self.stride
                    end_idx = start_idx + self.window_size
                    window = data[start_idx:end_idx]
                    
                    self.windows.append(window)
                    self.labels.append(fault_id)
                    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Mammoth v2 requires 'data' and 'targets' attributes
        self.data = torch.from_numpy(self.windows)
        self.targets = torch.from_numpy(self.labels)
        
        print(f"Loaded {len(self.windows)} windows for fault IDs {self.fault_ids} ({mode})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            window: (window_size, num_features) tensor
            label: fault class (0-21)
            window: same as first return (for compatibility with Mammoth)
        """
        window = self.data[idx]
        label = self.targets[idx]
        
        return window, label, window


class TennesseeEastmanContinual(ContinualDataset):
    """
    Tennessee Eastman Process for continual fault detection learning.

    Setting: Class-incremental learning, 11 tasks of 2 classes each.

    The original 22-class space (Normal + 21 faults) is grouped pairwise so
    that each task contains two classes; this makes Class-IL non-degenerate
    (a single-class-per-task split would lower-bound accuracy at chance,
    1/22 \approx 4.5\,\%).

    Task layout (default ``pairwise`` grouping):
        Task 0: classes {0, 1}     (Normal,  Fault 1)
        Task 1: classes {2, 3}     (Fault 2, Fault 3)
        ...
        Task 10: classes {20, 21}  (Fault 20, Fault 21)

    Each task involves learning to discriminate two new fault classes while
    retaining the ability to discriminate all previously learned ones.
    """

    NAME = 'tennessee-eastman'
    SETTING = 'class-il'  # Class-incremental learning
    N_CLASSES_PER_TASK = 2  # Two faults per task
    N_TASKS = 11  # 22 classes / 2 per task
    SIZE = (52,)  # 52 process variables (time-series input)
    
    def __init__(self, args):
        super().__init__(args)
        self.window_size = getattr(args, 'tep_window_size', 50)
        self.stride = getattr(args, 'tep_stride', 10)
        self._task_idx = 0
        
    def get_data_loaders(self):
        """Get train and test loaders for current task."""
        # Load ALL faults and let store_masked_loaders do the task-based filtering
        # This is the correct way for Mammoth v2 - it will mask based on targets
        train_dataset = TennesseeEastmanDataset(
            data_path=os.path.join(base_path(), 'TEP'),
            fault_ids=None,  # Load ALL faults
            train=True,
            window_size=self.window_size,
            stride=self.stride
        )
        
        test_dataset = TennesseeEastmanDataset(
            data_path=os.path.join(base_path(), 'TEP'),
            fault_ids=None,  # Load ALL faults
            train=False,
            window_size=self.window_size,
            stride=self.stride
        )
        
        # Use store_masked_loaders to properly wrap datasets for Mammoth v2
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        
        # Don't append test_loaders here - parent class handles it
        self.train_loader = train
        
        return train, test
    
    @staticmethod
    @set_default_from_args("backbone")
    def get_backbone():
        """
        Return CfC backbone for TEP.
        
        Input: (window_size, 52) - sequence of 52 process variables
        Output: 22 classes (normal + 21 faults)
        """
        return "tepcfc"
    
    @staticmethod
    def get_transform():
        return None
    
    @staticmethod
    def get_normalization_transform():
        return None
    
    @staticmethod
    def get_denormalization_transform():
        return None
    
    @staticmethod
    def get_loss():
        return F.cross_entropy
    
    @staticmethod
    def get_scheduler(model, args):
        return None
    
    @staticmethod
    def get_batch_size() -> int:
        return 32
    
    @staticmethod
    def get_minibatch_size() -> int:
        return TennesseeEastmanContinual.get_batch_size()


class TennesseeEastmanJoint(ContinualDataset):
    """
    Tennessee Eastman with joint training (all faults at once).
    Used as upper bound for comparison with incremental learning.
    """
    
    NAME = 'tennessee-eastman-joint'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 22  # All faults in single task
    N_TASKS = 1
    SIZE = (52,)  # 52 process variables
    N_TASKS = 1
    
    def __init__(self, args):
        super().__init__(args)
        self.window_size = getattr(args, 'tep_window_size', 50)
        self.stride = getattr(args, 'tep_stride', 10)
        
    def get_data_loaders(self):
        """Get train and test loaders with all faults."""
        train_dataset = TennesseeEastmanDataset(
            data_path=os.path.join(base_path(), 'TEP'),
            fault_ids=None,  # All faults
            train=True,
            window_size=self.window_size,
            stride=self.stride
        )
        
        test_dataset = TennesseeEastmanDataset(
            data_path=os.path.join(base_path(), 'TEP'),
            fault_ids=None,  # All faults
            train=False,
            window_size=self.window_size,
            stride=self.stride
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        
        return train_loader, test_loader
    
    @staticmethod
    @set_default_from_args("backbone")
    def get_backbone():
        return "tepcfc"
    
    @staticmethod
    def get_transform():
        return None
    
    @staticmethod
    def get_normalization_transform():
        return None
    
    @staticmethod
    def get_denormalization_transform():
        return None
    
    @staticmethod
    def get_loss():
        return F.cross_entropy
    
    @staticmethod
    def get_scheduler(model, args):
        return None
    
    @staticmethod
    def get_batch_size() -> int:
        return 32
    
    @staticmethod
    def get_minibatch_size() -> int:
        return TennesseeEastmanJoint.get_batch_size()
