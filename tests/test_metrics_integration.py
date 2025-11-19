#!/usr/bin/env python3
"""
Test script to verify that advanced metrics and tau monitoring are properly integrated
into the Mammoth training loop.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mammoth'))

import argparse
from argparse import Namespace

def test_args_parsing():
    """Test that the new arguments are properly defined."""
    from utils.args import add_management_args
    
    parser = argparse.ArgumentParser()
    add_management_args(parser)
    
    # Test parsing with new flags
    args = parser.parse_args([
        '--enable_advanced_metrics', '1',
        '--enable_tau_monitor', '1',
        '--tau_log_interval', '50'
    ])
    
    assert args.enable_advanced_metrics == True, "Failed to parse enable_advanced_metrics"
    assert args.enable_tau_monitor == True, "Failed to parse enable_tau_monitor"
    assert args.tau_log_interval == 50, "Failed to parse tau_log_interval"
    
    print("✓ Argument parsing test passed")

def test_imports():
    """Test that all necessary imports work."""
    try:
        from utils.tau_monitor import get_tau_monitor
        from utils.advanced_metrics import AdvancedMetricsManager
        from utils.training import train
        print("✓ Import test passed")
    except ImportError as e:
        print(f"✗ Import test failed: {e}")
        raise

def test_tau_monitor_initialization():
    """Test that tau monitor can be initialized."""
    from utils.tau_monitor import get_tau_monitor
    
    tau_monitor = get_tau_monitor(enabled=True, log_every_n_steps=100)
    assert tau_monitor is not None, "Tau monitor should not be None"
    print("✓ Tau monitor initialization test passed")

def test_metrics_manager_initialization():
    """Test that metrics manager can be initialized."""
    from utils.advanced_metrics import AdvancedMetricsManager
    
    config = {
        'representational_stability': {'enabled': True},
        'weight_change': {'enabled': True},
        'gradient_interference': {'enabled': True}
    }
    metrics_manager = AdvancedMetricsManager(config)
    assert metrics_manager is not None, "Metrics manager should not be None"
    print("✓ Metrics manager initialization test passed")

def test_training_function_signature():
    """Test that the training function has the correct signature."""
    import inspect
    from utils.training import train_single_epoch
    
    sig = inspect.signature(train_single_epoch)
    params = list(sig.parameters.keys())
    
    assert 'tau_monitor' in params, "tau_monitor parameter missing from train_single_epoch"
    assert 'metrics_manager' in params, "metrics_manager parameter missing from train_single_epoch"
    assert 'cur_task' in params, "cur_task parameter missing from train_single_epoch"
    
    print("✓ Training function signature test passed")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Metrics Integration")
    print("=" * 60)
    
    try:
        test_imports()
        test_args_parsing()
        test_tau_monitor_initialization()
        test_metrics_manager_initialization()
        test_training_function_signature()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Tests failed: {e}")
        print("=" * 60)
        sys.exit(1)
