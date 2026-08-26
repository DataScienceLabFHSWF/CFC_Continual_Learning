# Copyright 2024-present
# SECOM dataset for continual process-monitoring (pass/fail) learning.
#
# Source: McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine
#         Learning Repository. https://doi.org/10.24432/C54305  (CC BY 4.0)
#
# SECOM only has 2 native classes (pass/fail), which does not support a
# multi-class Class-IL split on its own. Each example carries a timestamp, so
# we sort chronologically and cut the record into 3 contiguous "eras" of
# roughly equal size (representing a shifting manufacturing-line state over
# time), and treat each era's pass/fail pair as a *separate* pair of classes.
# This is the same "task-specific class copies" construction used to turn a
# domain-shift benchmark into a Class-IL one: 3 tasks x 2 classes = 6 total
# classes, one pass/fail pair per era.

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils import set_default_from_args
from utils.conf import base_path

N_ERAS = 3
NAN_COL_THRESHOLD = 0.5  # drop feature columns with >50% missing values


class SecomDataset(Dataset):
    """SECOM dataset wrapped for Mammoth v2, split into 3 chronological eras."""

    def __init__(self, data_path, train=True, test_size=0.2, seed=0,
                 valid_cols=None, era_bounds=None,
                 era_means=None, era_stds=None):
        self.data_path = data_path
        self.train = train
        self.valid_cols = valid_cols
        self.era_bounds = era_bounds
        self.era_means = era_means
        self.era_stds = era_stds
        self._load_data(test_size=test_size, seed=seed)

    def _load_data(self, test_size, seed):
        raw = np.genfromtxt(os.path.join(self.data_path, 'secom.data'),
                             missing_values='NaN', filling_values=np.nan)
        with open(os.path.join(self.data_path, 'secom_labels.data')) as f:
            lines = [ln.split() for ln in f.read().splitlines()]
        pass_fail = np.array([int(ln[0]) for ln in lines], dtype=np.int64)
        # timestamp looks like: -1 "19/07/2008 11:55:00" -> ln[1], ln[2]
        timestamps = np.array([f"{ln[1]} {ln[2]}".strip('"') for ln in lines])
        order = np.argsort(timestamps)  # chronological order

        if self.valid_cols is None:
            nan_frac = np.isnan(raw).mean(axis=0)
            self.valid_cols = np.where(nan_frac <= NAN_COL_THRESHOLD)[0]
        raw = raw[:, self.valid_cols]

        if self.era_bounds is None:
            n = len(order)
            edges = np.linspace(0, n, N_ERAS + 1).astype(int)
            self.era_bounds = list(zip(edges[:-1], edges[1:]))

        self.era_means = self.era_means or [None] * N_ERAS
        self.era_stds = self.era_stds or [None] * N_ERAS

        all_feats, all_labels = [], []
        rng = np.random.default_rng(1234 + seed)
        for era_idx, (lo, hi) in enumerate(self.era_bounds):
            era_order = order[lo:hi]
            era_feats = raw[era_order]
            era_pf = pass_fail[era_order]
            # 0 = pass, 1 = fail (locally), remapped to global era-specific ids
            era_local = (era_pf == 1).astype(np.int64)

            idx = np.arange(len(era_order))
            rng.shuffle(idx)
            n_test = max(1, int(round(len(idx) * test_size)))
            test_idx, train_idx = idx[:n_test], idx[n_test:]

            if self.era_means[era_idx] is None:
                train_block = era_feats[train_idx]
                col_mean = np.nanmean(train_block, axis=0)
                col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
                self.era_means[era_idx] = col_mean
                filled = np.where(np.isnan(train_block), col_mean, train_block)
                col_std = filled.std(axis=0) + 1e-8
                self.era_stds[era_idx] = col_std

            mean, std = self.era_means[era_idx], self.era_stds[era_idx]
            era_feats = np.where(np.isnan(era_feats), mean, era_feats)
            era_feats = (era_feats - mean) / std

            sel = train_idx if self.train else test_idx
            all_feats.append(era_feats[sel].astype(np.float32))
            all_labels.append(era_local[sel] + era_idx * 2)

        features = np.concatenate(all_feats, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        self.data = torch.from_numpy(features)
        self.targets = torch.from_numpy(labels)
        print(f"Loaded {len(labels)} SECOM samples across {N_ERAS} eras "
              f"({'train' if self.train else 'test'}), {raw.shape[1]} features")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y, x


class SecomContinual(ContinualDataset):
    """
    SECOM for continual process-monitoring learning.

    Setting: Class-incremental learning, 3 tasks of 2 classes each, where
    each task is one chronological "era" of the manufacturing line and its
    own pass/fail pair (Task i: {2i (pass), 2i+1 (fail)}). This construction
    is necessary because SECOM natively only has 2 classes; see
    docs/industrial_benchmarks.md for the full rationale.
    """

    NAME = 'secom'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = N_ERAS
    SIZE = (562,)  # columns surviving the >50%-missing filter (see _load_data)

    def __init__(self, args):
        super().__init__(args)

    def get_data_loaders(self):
        train_dataset = SecomDataset(
            data_path=os.path.join(base_path(), 'SECOM'),
            train=True, seed=self.args.seed)
        test_dataset = SecomDataset(
            data_path=os.path.join(base_path(), 'SECOM'),
            train=False, seed=self.args.seed,
            valid_cols=train_dataset.valid_cols, era_bounds=train_dataset.era_bounds,
            era_means=train_dataset.era_means, era_stds=train_dataset.era_stds)

        self.SIZE = (len(train_dataset.valid_cols),)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        self.train_loader = train
        return train, test

    @staticmethod
    @set_default_from_args("backbone")
    def get_backbone():
        return "secomcfc"

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
        return SecomContinual.get_batch_size()
