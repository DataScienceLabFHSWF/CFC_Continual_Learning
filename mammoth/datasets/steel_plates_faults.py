# Copyright 2024-present
# Steel Plates Faults dataset for continual fault-classification learning.
#
# Source: Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults
#         [Dataset]. UCI Machine Learning Repository.
#         https://doi.org/10.24432/C5J88N  (CC BY 4.0)
#
# The raw file (Faults.NNA) has 1941 rows x 34 columns: 27 numeric features
# describing a defect region on a steel plate, followed by a 7-column
# one-hot fault-type indicator (Pastry, Z_Scratch, K_Scatch, Stains,
# Dirtiness, Bumps, Other_Faults). We merge the two smallest classes
# ("Stains", 72 samples, and "Dirtiness", 55 samples -- both surface-blemish
# defects) into a single "Surface_Blemish" class, giving 6 evenly-groupable
# classes for a 3-task, 2-classes-per-task Class-IL split.

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils import set_default_from_args
from utils.conf import base_path

CLASS_NAMES = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Surface_Blemish', 'Bumps', 'Other_Faults']


class SteelPlatesFaultsDataset(Dataset):
    """Steel Plates Faults dataset wrapped for Mammoth v2 (static feature vectors)."""

    def __init__(self, data_path, train=True, test_size=0.2, seed=0,
                 feat_mean=None, feat_std=None):
        self.data_path = data_path
        self.train = train
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self._load_data(test_size=test_size, seed=seed)

    def _load_data(self, test_size, seed):
        raw = np.loadtxt(os.path.join(self.data_path, 'Faults.NNA'))
        features = raw[:, :27].astype(np.float32)
        onehot = raw[:, 27:34]
        labels = onehot.argmax(axis=1).astype(np.int64)
        # Merge Stains (3) and Dirtiness (4) -> class index 3 ("Surface_Blemish");
        # Bumps (5) -> 4, Other_Faults (6) -> 5.
        remap = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5}
        labels = np.array([remap[l] for l in labels], dtype=np.int64)

        # Stratified, seeded train/test split (fixed regardless of `train`
        # flag or CL protocol, so the split doesn't change with args.seed).
        rng = np.random.default_rng(1234 + seed)
        train_idx, test_idx = [], []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            n_test = max(1, int(round(len(idx) * test_size)))
            test_idx.extend(idx[:n_test])
            train_idx.extend(idx[n_test:])
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)

        if self.feat_mean is None or self.feat_std is None:
            # Stats from the training split only (avoid test-set leakage).
            self.feat_mean = features[train_idx].mean(axis=0)
            self.feat_std = features[train_idx].std(axis=0) + 1e-8

        features = (features - self.feat_mean) / self.feat_std
        sel = train_idx if self.train else test_idx

        self.data = torch.from_numpy(features[sel])
        self.targets = torch.from_numpy(labels[sel])
        print(f"Loaded {len(sel)} Steel Plates Faults samples "
              f"({'train' if self.train else 'test'})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y, x


class SteelPlatesFaultsContinual(ContinualDataset):
    """
    Steel Plates Faults for continual fault-classification learning.

    Setting: Class-incremental learning, 3 tasks of 2 classes each.
        Task 0: {Pastry, Z_Scratch}
        Task 1: {K_Scatch, Surface_Blemish}
        Task 2: {Bumps, Other_Faults}

    Unlike TEP, feature vectors here are static (one row per defect region,
    no temporal structure), so this benchmark primarily tests whether CfC's
    architectural bias still helps when there is no genuine sequential
    signal for its continuous-time dynamics to exploit.
    """

    NAME = 'steel-plates-faults'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 3
    SIZE = (27,)

    def __init__(self, args):
        super().__init__(args)

    def get_data_loaders(self):
        train_dataset = SteelPlatesFaultsDataset(
            data_path=os.path.join(base_path(), 'SteelPlatesFaults'),
            train=True, seed=self.args.seed)
        test_dataset = SteelPlatesFaultsDataset(
            data_path=os.path.join(base_path(), 'SteelPlatesFaults'),
            train=False, seed=self.args.seed,
            feat_mean=train_dataset.feat_mean, feat_std=train_dataset.feat_std)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        self.train_loader = train
        return train, test

    @staticmethod
    @set_default_from_args("backbone")
    def get_backbone():
        return "steelplatescfc"

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
        return SteelPlatesFaultsContinual.get_batch_size()
