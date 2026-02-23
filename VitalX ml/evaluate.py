"""
Evaluation Script for VitalX Sepsis Prediction Model
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os
import json

import config
from dataset import SepsisDataset
from model import load_model
from utils import compute_metrics, print_metrics


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Computation device
        threshold: Classification threshold
    
    Returns:
        Dictionary of predictions and metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader):
            sequences = sequences.to(device)
            
            # Forward pass
            outputs = model(sequences)
            
            # Store predictions
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, threshold)
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'metrics': metrics
    }


def plot_roc_curve(labels, predictions, save_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_pr_curve(labels, predictions, save_path):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to {save_path}")


def plot_threshold_analysis(labels, predictions, save_path):
    """Plot performance metrics across different thresholds"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    sensitivities = []
    specificities = []
    f1_scores = []
    
    for thresh in thresholds:
        metrics = compute_metrics(labels, predictions, threshold=thresh)
        sensitivities.append(metrics['sensitivity'])
        specificities.append(metrics['specificity'])
        f1_scores.append(metrics['f1_score'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sensitivities, 'o-', label='Sensitivity', linewidth=2)
    plt.plot(thresholds, specificities, 's-', label='Specificity', linewidth=2)
    plt.plot(thresholds, f1_scores, '^-', label='F1 Score', linewidth=2)
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold analysis saved to {save_path}")


def analyze_early_detection(labels, predictions, window_hours=6, threshold=0.5):
    """
    Analyze early detection performance
    
    Focus on predictions made within the early warning window
    """
    # This is a simplified version
    # In practice, you would need to track time-to-sepsis for each patient
    
    print("\n" + "="*60)
    print("EARLY DETECTION ANALYSIS")
    print("="*60)
    print(f"Target: Predict sepsis {window_hours} hours before onset")
    print(f"Classification threshold: {threshold}")
    
    # Compute metrics at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\nPerformance at different thresholds:")
    print("-" * 60)
    print(f"{'Threshold':<12} {'Sensitivity':<12} {'Specificity':<12} {'F1 Score':<12}")
    print("-" * 60)
    
    for thresh in thresholds:
        metrics = compute_metrics(labels, predictions, threshold=thresh)
        print(f"{thresh:<12.2f} {metrics['sensitivity']:<12.4f} "
              f"{metrics['specificity']:<12.4f} {metrics['f1_score']:<12.4f}")
    
    print("="*60 + "\n")


def save_evaluation_report(results, save_path):
    """Save evaluation results to JSON"""
    # Convert numpy types to native Python types
    report = {
        'metrics': results['metrics'],
        'statistics': {
            'n_samples': int(len(results['labels'])),
            'n_positive': int(sum(results['labels'])),
            'n_negative': int(len(results['labels']) - sum(results['labels'])),
            'prevalence': float(sum(results['labels']) / len(results['labels']))
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Evaluation report saved to {save_path}")


def main():
    """Main evaluation pipeline"""
    print("\n" + "="*60)
    print("VitalX SEPSIS PREDICTION - EVALUATION")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"\nError: Model not found at {config.MODEL_SAVE_PATH}")
        print("Please train the model first using train.py")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(config.MODEL_SAVE_PATH, device)
    
    # Load scaler
    print("Loading scaler...")
    from dataset import SepsisDataset
    scaler = SepsisDataset.load_scaler(config.SCALER_SAVE_PATH)
    
    # Load test data
    print("\nLoading test data...")
    data_dirs = [config.DATA_DIR_A, config.DATA_DIR_B]
    existing_dirs = [d for d in data_dirs if os.path.exists(d)]
    
    test_dataset = SepsisDataset(
        data_dirs=existing_dirs,
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=scaler,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print_metrics(results['metrics'], phase='Test')
    
    # Create output directory for plots
    plots_dir = os.path.join(config.OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    
    plot_roc_curve(
        results['labels'], 
        results['predictions'],
        os.path.join(plots_dir, 'roc_curve.png')
    )
    
    plot_pr_curve(
        results['labels'],
        results['predictions'],
        os.path.join(plots_dir, 'pr_curve.png')
    )
    
    plot_threshold_analysis(
        results['labels'],
        results['predictions'],
        os.path.join(plots_dir, 'threshold_analysis.png')
    )
    
    # Early detection analysis
    analyze_early_detection(results['labels'], results['predictions'])
    
    # Save evaluation report
    report_path = os.path.join(config.OUTPUT_DIR, 'evaluation_report.json')
    save_evaluation_report(results, report_path)
    
    print("\n✓ Evaluation completed successfully!")
    print(f"Results saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
