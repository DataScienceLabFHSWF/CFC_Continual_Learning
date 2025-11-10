"""Test TEP data loading to verify format."""

import numpy as np
import os

data_dir = "data/TEP"

# Load one training file
print("Loading d00.dat (normal operation)...")
data_train = np.loadtxt(os.path.join(data_dir, "d00.dat"))
print(f"Raw shape: {data_train.shape}")

# Transpose to get (samples, features)
data_train = data_train.T
print(f"After transpose: {data_train.shape}")
print(f"Expected: (500 samples, 52 features)")
print()

# Load corresponding test file
print("Loading d00_te.dat (normal operation test)...")
data_test = np.loadtxt(os.path.join(data_dir, "d00_te.dat"))
print(f"Raw shape: {data_test.shape}")
data_test = data_test.T
print(f"After transpose: {data_test.shape}")
print(f"Expected: (960 samples, 52 features)")
print()

# Check a fault file
print("Loading d01.dat (fault IDV(1))...")
data_fault = np.loadtxt(os.path.join(data_dir, "d01.dat"))
data_fault = data_fault.T
print(f"Shape: {data_fault.shape}")
print()

# Verify numerical range
print("Sample statistics (d00 train):")
print(f"  Min: {data_train.min():.6f}")
print(f"  Max: {data_train.max():.6f}")
print(f"  Mean: {data_train.mean():.6f}")
print(f"  Std: {data_train.std():.6f}")
print()

# Show first few values
print("First 5 samples, first 3 features:")
print(data_train[:5, :3])
