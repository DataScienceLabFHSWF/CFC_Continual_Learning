#!/usr/bin/env python3
"""
Gradient Boosting Baseline for Tennessee Eastman Process Fault Detection

Compares XGBoost and LightGBM against CfC continual learning methods.
Serves as traditional ML baseline for the TEP benchmark.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")


def load_tep_data(data_dir='data/tennessee_eastman'):
    """Load Tennessee Eastman Process dataset."""
    data_path = Path(data_dir)
    
    # Load all fault data files (d00.dat to d21.dat)
    all_data = []
    all_labels = []
    
    for fault_id in range(22):  # 0 = normal, 1-21 = faults
        file_path = data_path / f'd{fault_id:02d}.dat'
        
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        # Load data (assume CSV or space-separated)
        try:
            data = pd.read_csv(file_path, delim_whitespace=True, header=None)
        except:
            data = np.loadtxt(file_path)
            data = pd.DataFrame(data)
        
        # Add labels
        labels = np.full(len(data), fault_id)
        
        all_data.append(data.values)
        all_labels.append(labels)
        
        print(f"Loaded fault {fault_id}: {len(data)} samples, {data.shape[1]} features")
    
    # Combine all data
    X = np.vstack(all_data)
    y = np.concatenate(all_labels)
    
    return X, y


def create_temporal_features(X, window_size=10):
    """Create temporal features from time series."""
    # X shape: (samples, features)
    # Create features: [current, mean_window, std_window, max_window, min_window]
    
    n_samples, n_features = X.shape
    
    # Initialize with current values
    X_temporal = X.copy()
    
    # Add rolling statistics
    if window_size > 1:
        for i in range(n_samples):
            start_idx = max(0, i - window_size + 1)
            window = X[start_idx:i+1]
            
            # Compute statistics
            mean_feat = window.mean(axis=0)
            std_feat = window.std(axis=0)
            max_feat = window.max(axis=0)
            min_feat = window.min(axis=0)
            
            # Concatenate features
            X_temporal[i] = np.concatenate([
                X[i],  # Current values
                mean_feat,  # Mean over window
                std_feat,   # Std over window
                max_feat - min_feat  # Range
            ])
    
    return X_temporal


class GradientBoostingTEP:
    """Gradient Boosting classifier for TEP."""
    
    def __init__(self, model_type='xgboost', **params):
        self.model_type = model_type
        self.params = params
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        start_time = time.time()
        
        if self.model_type == 'xgboost' and HAS_XGB:
            # XGBoost parameters
            default_params = {
                'objective': 'multi:softmax',
                'num_class': len(np.unique(y_train)),
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
            }
            default_params.update(self.params)
            
            self.model = xgb.XGBClassifier(**default_params)
            
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
                self.model.fit(X_train_scaled, y_train, 
                             eval_set=eval_set,
                             verbose=False)
            else:
                self.model.fit(X_train_scaled, y_train)
        
        elif self.model_type == 'lightgbm' and HAS_LGB:
            # LightGBM parameters
            default_params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y_train)),
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1,
            }
            default_params.update(self.params)
            
            self.model = lgb.LGBMClassifier(**default_params)
            
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
                self.model.fit(X_train_scaled, y_train,
                             eval_set=eval_set,
                             verbose=False)
            else:
                self.model.fit(X_train_scaled, y_train)
        else:
            raise ValueError(f"Model type {self.model_type} not available or not installed")
        
        train_time = time.time() - start_time
        return train_time
    
    def predict(self, X_test):
        """Make predictions."""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        y_pred = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Per-class accuracy
        conf_matrix = confusion_matrix(y_test, y_pred)
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        results['per_class_accuracy'] = per_class_acc.tolist()
        
        return results, y_pred
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.model_type == 'xgboost':
            return self.model.feature_importances_
        elif self.model_type == 'lightgbm':
            return self.model.feature_importances_
        else:
            return None


def visualize_results(results, output_path='figures/tep_gb_results.pdf'):
    """Visualize results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy comparison
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in models]
    
    axes[0, 0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Overall Accuracy')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 100])
    
    # Plot 2: F1 scores
    f1_macro = [results[m]['f1_macro'] * 100 for m in models]
    f1_weighted = [results[m]['f1_weighted'] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    axes[0, 1].bar(x - width/2, f1_macro, width, label='F1 Macro', color='#2ca02c')
    axes[0, 1].bar(x + width/2, f1_weighted, width, label='F1 Weighted', color='#d62728')
    axes[0, 1].set_ylabel('F1 Score (%)')
    axes[0, 1].set_title('F1 Scores')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 100])
    
    # Plot 3: Training time
    train_times = [results[m]['train_time'] for m in models]
    axes[1, 0].bar(models, train_times, color=['#9467bd', '#8c564b'])
    axes[1, 0].set_ylabel('Training Time (s)')
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Per-class accuracy (first model)
    first_model = models[0]
    per_class = results[first_model]['per_class_accuracy']
    fault_ids = list(range(len(per_class)))
    
    axes[1, 1].plot(fault_ids, np.array(per_class) * 100, marker='o', linewidth=2)
    axes[1, 1].set_xlabel('Fault ID')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title(f'Per-Fault Accuracy ({first_model})')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Results visualization saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Gradient Boosting baseline for TEP')
    parser.add_argument('--data_dir', type=str, default='data/tennessee_eastman',
                       help='Path to TEP data directory')
    parser.add_argument('--models', nargs='+', default=['xgboost', 'lightgbm'],
                       choices=['xgboost', 'lightgbm'],
                       help='Models to evaluate')
    parser.add_argument('--window_size', type=int, default=10,
                       help='Window size for temporal features')
    parser.add_argument('--output', type=str, default='results/tep_gb_results.json',
                       help='Output JSON file')
    args = parser.parse_args()
    
    print("="*60)
    print("TEP Gradient Boosting Baseline")
    print("="*60)
    
    # Load data
    print("\nLoading TEP data...")
    X, y = load_tep_data(args.data_dir)
    print(f"Total samples: {len(X)}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Create temporal features
    if args.window_size > 1:
        print(f"\nCreating temporal features (window_size={args.window_size})...")
        # X = create_temporal_features(X, args.window_size)
        # Note: Commented out for now - needs proper time-series handling
        print(f"Features after temporal engineering: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train and evaluate models
    results = {}
    
    for model_type in args.models:
        if model_type == 'xgboost' and not HAS_XGB:
            print(f"\nSkipping {model_type} (not installed)")
            continue
        if model_type == 'lightgbm' and not HAS_LGB:
            print(f"\nSkipping {model_type} (not installed)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")
        
        # Initialize model
        gb = GradientBoostingTEP(model_type=model_type)
        
        # Train
        train_time = gb.train(X_train, y_train, X_val, y_val)
        print(f"Training time: {train_time:.2f}s")
        
        # Evaluate on validation set
        val_results, _ = gb.evaluate(X_val, y_val)
        print(f"Validation Accuracy: {val_results['accuracy']*100:.2f}%")
        print(f"Validation F1 (macro): {val_results['f1_macro']*100:.2f}%")
        
        # Evaluate on test set
        test_results, y_pred = gb.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_results['accuracy']*100:.2f}%")
        print(f"Test F1 (macro): {test_results['f1_macro']*100:.2f}%")
        
        # Store results
        results[model_type] = {
            'train_time': train_time,
            'accuracy': test_results['accuracy'],
            'f1_macro': test_results['f1_macro'],
            'f1_weighted': test_results['f1_weighted'],
            'per_class_accuracy': test_results['per_class_accuracy'],
            'val_accuracy': val_results['accuracy'],
        }
        
        # Feature importance
        importance = gb.get_feature_importance()
        if importance is not None:
            top_k = 10
            top_features = np.argsort(importance)[-top_k:][::-1]
            print(f"\nTop {top_k} features:")
            for i, feat_idx in enumerate(top_features):
                print(f"  {i+1}. Feature {feat_idx}: {importance[feat_idx]:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Visualize
    if results:
        visualize_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model_type, res in results.items():
        print(f"{model_type.upper()}:")
        print(f"  Accuracy: {res['accuracy']*100:.2f}%")
        print(f"  F1 Macro: {res['f1_macro']*100:.2f}%")
        print(f"  Train Time: {res['train_time']:.2f}s")


if __name__ == '__main__':
    main()
