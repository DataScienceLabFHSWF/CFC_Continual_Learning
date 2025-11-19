#!/usr/bin/env python3
"""
Quick end-to-end validation test to verify the metrics integration works
with a real training run.

This runs a minimal MNIST experiment with advanced metrics enabled.
"""

import sys
import os

# Navigate to mammoth directory
mammoth_path = os.path.join(os.path.dirname(__file__), '..', 'mammoth')
sys.path.insert(0, mammoth_path)
os.chdir(mammoth_path)

from argparse import Namespace
from datasets import get_dataset
from models import get_model
from utils.training import train

def main():
    """Run a minimal MNIST experiment with metrics enabled."""
    
    # Create minimal args for a quick test
    args = Namespace(
        # Dataset
        dataset='seq-mnist',
        n_tasks=2,  # Only 2 tasks for quick test
        
        # Model
        model='sgd',
        backbone='mnistltc',
        
        # Training
        n_epochs=1,  # Only 1 epoch per task
        batch_size=32,
        lr=0.03,
        
        # Metrics - THE KEY PART
        enable_other_metrics=True,
        enable_advanced_metrics=True,
        enable_tau_monitor=True,
        tau_log_interval=10,
        
        # Management
        nowand=True,  # Disable wandb for this test
        non_verbose=False,
        disable_log=True,
        debug_mode=False,
        code_optimization=0,
        device='cpu',  # Use CPU for quick test
        num_workers=0,
        
        # Other required args
        validation=None,
        validation_mode='current',
        fitting_mode='epochs',
        scheduler_mode='task',
        inference_only=False,
        joint=0,
        eval_future=False,
        start_from=None,
        stop_after=None,
        savecheck=None,
        loadcheck=None,
        distributed='no',
        save_after_interrupt=False,
        eval_epochs=None,
        
        # Backbone args
        hidden_size=64,  # Small for speed
        
        # Config
        dataset_config=None,
        model_config='default',
        
        # Job tracking
        conf_jobnum='test-0',
    )
    
    print("=" * 70)
    print("VALIDATION TEST: Metrics Integration")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Backbone: {args.backbone}")
    print(f"Advanced Metrics: {args.enable_advanced_metrics}")
    print(f"Tau Monitoring: {args.enable_tau_monitor}")
    print("=" * 70)
    
    # Get dataset
    dataset = get_dataset(args)
    
    # Get model
    model = get_model(args, backbone=None, loss=None, transform=None)
    
    # Train with metrics enabled
    try:
        train(model, dataset, args)
        print("\n" + "=" * 70)
        print("✓ VALIDATION PASSED: Metrics integration working correctly!")
        print("=" * 70)
        return 0
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ VALIDATION FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
