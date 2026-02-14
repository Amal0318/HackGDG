"""
VitalX Evaluation Metrics
==========================

Comprehensive metrics for medical ML model evaluation.

Key metrics:
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- Confusion Matrix
- Sensitivity, Specificity
- Clinical metrics (FPR, FNR)
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import json


# ==============================
# CORE METRICS
# ==============================
def calculate_metrics(y_true, y_pred, y_proba=None, threshold=0.5):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted labels (0 or 1)
        y_proba: predicted probabilities (optional)
        threshold: classification threshold
    
    Returns:
        dict of metrics
    """
    # Convert to numpy if torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_proba is not None and isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if y_proba is not None:
        y_proba = y_proba.flatten()
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Sensitivity and Specificity
    sensitivity = recall  # Same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # NPV and PPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    ppv = precision  # Positive Predictive Value (same as precision)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fpr': fpr,  # False Positive Rate
        'fnr': fnr,  # False Negative Rate
        'npv': npv,  # Negative Predictive Value
        'ppv': ppv,  # Positive Predictive Value
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    # ROC-AUC and PR-AUC (require probabilities)
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            metrics['roc_auc'] = roc_auc
        except ValueError:
            metrics['roc_auc'] = None
        
        try:
            pr_auc = average_precision_score(y_true, y_proba)
            metrics['pr_auc'] = pr_auc
        except ValueError:
            metrics['pr_auc'] = None
    
    return metrics


# ==============================
# MEDICAL PRIORITY METRICS
# ==============================
def calculate_medical_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate medical-focused metrics.
    
    Priority: High Recall (minimize false negatives)
    
    Returns:
        dict with medical interpretations
    """
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Medical interpretation
    medical_report = {
        'sensitivity': metrics['recall'],  # Key metric: catch deteriorating patients
        'specificity': metrics['specificity'],  # Avoid false alarms
        'false_negative_rate': metrics['fnr'],  # Critical: missed deteriorations
        'false_positive_rate': metrics['fpr'],  # Alert fatigue concern
        'negative_predictive_value': metrics['npv'],  # Confidence in stable predictions
        'positive_predictive_value': metrics['ppv'],  # Confidence in deterioration predictions
        'total_alerts': metrics['tp'] + metrics['fp'],  # Total positive predictions
        'missed_deteriorations': metrics['fn'],  # Critical misses
        'correct_stable': metrics['tn'],  # Correct stable predictions
        'correct_deteriorations': metrics['tp']  # Caught deteriorations
    }
    
    # Risk assessment
    if medical_report['false_negative_rate'] > 0.10:
        medical_report['risk_level'] = 'HIGH - Too many missed deteriorations'
    elif medical_report['false_positive_rate'] > 0.30:
        medical_report['risk_level'] = 'MEDIUM - High false alarm rate'
    else:
        medical_report['risk_level'] = 'ACCEPTABLE'
    
    return medical_report


# ==============================
# PRINT METRICS
# ==============================
def print_metrics_report(metrics, title="Model Evaluation"):
    """
    Print formatted metrics report.
    
    Args:
        metrics: dict from calculate_metrics()
        title: report title
    """
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)
    
    # Classification metrics
    print("\nClassification Metrics:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics and metrics['pr_auc'] is not None:
        print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    
    # Medical metrics
    print("\nClinical Metrics:")
    print(f"  Sensitivity:  {metrics['sensitivity']:.4f} (Recall)")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    print(f"  NPV:          {metrics['npv']:.4f}")
    print(f"  PPV:          {metrics['ppv']:.4f}")
    
    # Error rates
    print("\nError Rates:")
    print(f"  False Positive Rate: {metrics['fpr']:.4f}")
    print(f"  False Negative Rate: {metrics['fnr']:.4f} ⚠️ Critical for medical use")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']:>6}  |  FP: {metrics['fp']:>6}")
    print(f"  FN: {metrics['fn']:>6}  |  TP: {metrics['tp']:>6}")
    
    print("=" * 70)


def print_medical_report(medical_metrics, title="Medical Safety Report"):
    """Print medical-focused report."""
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)
    
    print(f"\n🏥 Clinical Performance:")
    print(f"  Sensitivity (Catch Rate):        {medical_metrics['sensitivity']:.1%}")
    print(f"  Specificity (Stable Detection):  {medical_metrics['specificity']:.1%}")
    
    print(f"\n⚠️  Safety Metrics:")
    print(f"  Missed Deteriorations (FN):      {medical_metrics['missed_deteriorations']}")
    print(f"  False Negative Rate:             {medical_metrics['false_negative_rate']:.1%}")
    print(f"  False Positive Rate:             {medical_metrics['false_positive_rate']:.1%}")
    
    print(f"\n📊 Predictive Values:")
    print(f"  Positive Predictive Value (PPV): {medical_metrics['positive_predictive_value']:.1%}")
    print(f"  Negative Predictive Value (NPV): {medical_metrics['negative_predictive_value']:.1%}")
    
    print(f"\n📈 Summary:")
    print(f"  Total Alerts Generated:          {medical_metrics['total_alerts']}")
    print(f"  Correct Deterioration Catches:   {medical_metrics['correct_deteriorations']}")
    print(f"  Correct Stable Predictions:      {medical_metrics['correct_stable']}")
    
    print(f"\n🎯 Risk Assessment: {medical_metrics['risk_level']}")
    
    print("=" * 70)


# ==============================
# PLOTTING FUNCTIONS
# ==============================
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: ground truth
        y_pred: predictions
        save_path: path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stable', 'Deteriorating'],
                yticklabels=['Stable', 'Deteriorating'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: ground truth
        y_proba: predicted probabilities
        save_path: path to save figure (optional)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, save_path=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: ground truth
        y_proba: predicted probabilities
        save_path: path to save figure (optional)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_metrics(y_true, y_pred, y_proba, save_dir=None):
    """
    Generate all evaluation plots.
    
    Args:
        y_true: ground truth
        y_pred: predictions
        y_proba: probabilities
        save_dir: directory to save plots (optional)
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Confusion Matrix
    cm_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # ROC Curve
    roc_path = os.path.join(save_dir, 'roc_curve.png') if save_dir else None
    plot_roc_curve(y_true, y_proba, roc_path)
    
    # PR Curve
    pr_path = os.path.join(save_dir, 'pr_curve.png') if save_dir else None
    plot_precision_recall_curve(y_true, y_proba, pr_path)


# ==============================
# SAVE METRICS
# ==============================
def save_metrics_json(metrics, filepath):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: dict of metrics
        filepath: path to save JSON
    """
    # Convert numpy types to Python types
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_json[key] = float(value)
        else:
            metrics_json[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"Metrics saved to {filepath}")


# ==============================
# TESTING
# ==============================
if __name__ == "__main__":
    print("Testing VitalX Metrics\n")
    
    # Generate dummy predictions
    np.random.seed(42)
    n_samples = 500
    
    y_true = np.random.randint(0, 2, n_samples)
    y_proba = np.random.rand(n_samples)
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Add some correlation to make it realistic
    y_proba = (y_true * 0.7 + np.random.rand(n_samples) * 0.3)
    y_proba = np.clip(y_proba, 0, 1)
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    print_metrics_report(metrics, "Test Model Evaluation")
    
    # Medical metrics
    medical = calculate_medical_metrics(y_true, y_pred, y_proba)
    print_medical_report(medical, "Test Medical Report")
    
    # Test plotting (without saving)
    print("\nGenerating test plots...")
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_proba)
    plot_precision_recall_curve(y_true, y_proba)
    
    print("\n✓ All metric tests passed!")
