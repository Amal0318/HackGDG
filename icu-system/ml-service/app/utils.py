"""
Utility functions for VitalX Sepsis Prediction Model
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import json
import os


def compute_utility_score(labels, predictions, dt_early=12, dt_optimal=6, dt_late=3.0, 
                         max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    """
    Compute utility score for sepsis prediction (PhysioNet 2019 metric)
    
    Args:
        labels: Ground truth labels
        predictions: Predicted labels (binary)
        dt_early: Early prediction window (hours before sepsis)
        dt_optimal: Optimal prediction window (hours before sepsis)
        dt_late: Late prediction window (hours after sepsis)
    
    Returns:
        Utility score
    """
    if len(labels) == 0:
        return 0.0
    
    # Find all patients with sepsis
    num_instances = len(labels)
    
    # Compute utility for each instance
    utilities = np.zeros(num_instances)
    
    for i in range(num_instances):
        if labels[i] == 1:  # Positive class (sepsis)
            if predictions[i] == 1:  # True Positive
                utilities[i] = max_u_tp
            else:  # False Negative
                utilities[i] = min_u_fn
        else:  # Negative class (no sepsis)
            if predictions[i] == 1:  # False Positive
                utilities[i] = u_fp
            else:  # True Negative
                utilities[i] = u_tn
    
    return np.sum(utilities)


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # AUROC and AUPRC
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Derived metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Utility score
    utility = compute_utility_score(y_true, y_pred)
    
    metrics = {
        'auroc': auroc,
        'auprc': auprc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'utility_score': utility,
        'threshold': threshold,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    return metrics


def save_config(config_dict, save_path):
    """Save configuration to JSON"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Configuration saved to {save_path}")


def load_config(config_path):
    """Load configuration from JSON"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
        
        return self.early_stop


def print_metrics(metrics, phase='Validation'):
    """Print metrics in a formatted way"""
    print(f"\n{'='*60}")
    print(f"{phase} Metrics")
    print(f"{'='*60}")
    print(f"AUROC:          {metrics['auroc']:.4f}")
    print(f"AUPRC:          {metrics['auprc']:.4f}")
    print(f"Sensitivity:    {metrics['sensitivity']:.4f}")
    print(f"Specificity:    {metrics['specificity']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"F1 Score:       {metrics['f1_score']:.4f}")
    print(f"Utility Score:  {metrics['utility_score']:.2f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']:6d}    FP: {metrics['fp']:6d}")
    print(f"  FN: {metrics['fn']:6d}    TN: {metrics['tn']:6d}")
    print(f"{'='*60}\n")
