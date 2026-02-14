"""
VitalX LSTM Training Script
============================

Train LSTM with Attention model for ICU deterioration prediction.

Features:
- Early stopping
- Gradient clipping
- Learning rate scheduling
- Model checkpointing
- Comprehensive evaluation
- Fallback logistic regression training
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMAttentionModel, LogisticRegressionFallback, create_model
from training.dataset import create_data_splits, create_dataloaders
from utils.metrics import (
    calculate_metrics, calculate_medical_metrics,
    print_metrics_report, print_medical_report,
    plot_all_metrics, save_metrics_json
)


# ==============================
# CONFIGURATION
# ==============================
class Config:
    """Training configuration."""
    
    # Data
    data_path = "data/"  # Changed from ../data/ to data/ since we run from training/
    X_file = "X.npy"
    y_file = "y.npy"
    
    # Model architecture
    input_size = 14
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    
    # Training
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-5
    
    # Optimization
    clip_norm = 1.0  # Gradient clipping
    patience = 10    # Early stopping patience
    
    # Data split
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15
    
    # Training options
    use_balanced_sampling = True
    use_augmentation = False
    
    # Device - CUDA GPU ONLY (no CPU fallback)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for training but not available!\n"
            "Please install PyTorch with CUDA support:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        )
    device = torch.device('cuda')
    
    # Paths
    save_dir = "saved_models/"
    plots_dir = "saved_models/plots/"
    
    # Random seed
    random_state = 42


# ==============================
# TRAINING LOOP
# ==============================
class Trainer:
    """Model trainer with all bells and whistles."""
    
    def __init__(self, model, config):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Config object
        """
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Loss function - BCELoss (model outputs probability via sigmoid)
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize ROC-AUC
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_roc_auc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_roc_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Move to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(X_batch)
            
            # Compute loss
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(X_batch)
                
                # Compute loss
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                # Store for metrics
                all_outputs.append(outputs.cpu())
                all_labels.append(y_batch.cpu())
        
        # Concatenate all batches
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        
        # Calculate metrics
        y_proba = all_outputs.numpy().flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        y_true = all_labels.numpy().flatten()
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader):
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: training data loader
            val_loader: validation data loader
        """
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Early stopping patience: {self.config.patience}")
        print("=" * 70 + "\n")
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_roc_auc'].append(val_metrics.get('roc_auc', 0.0))
            self.history['learning_rate'].append(current_lr)
            
            # Learning rate scheduling
            if val_metrics.get('roc_auc') is not None:
                self.scheduler.step(val_metrics['roc_auc'])
            
            # Print progress
            print(f"Epoch {epoch:>3}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val ROC-AUC: {val_metrics.get('roc_auc', 0.0):.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # Check for improvement
            current_roc_auc = val_metrics.get('roc_auc', 0.0)
            if current_roc_auc > self.best_roc_auc:
                self.best_roc_auc = current_roc_auc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint('best_model.pth')
                print(f"  → New best model saved! ROC-AUC: {self.best_roc_auc:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best ROC-AUC: {self.best_roc_auc:.4f} at epoch {self.best_epoch}")
                break
        
        print("\n" + "=" * 70)
        print("Training Complete")
        print("=" * 70)
        print(f"Best ROC-AUC: {self.best_roc_auc:.4f} at epoch {self.best_epoch}")
        print("=" * 70 + "\n")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_size': self.config.input_size,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout
            },
            'best_roc_auc': self.best_roc_auc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }
        
        filepath = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['val_accuracy'], label='Val Accuracy', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # ROC-AUC
        axes[1, 0].plot(self.history['val_roc_auc'], label='Val ROC-AUC', color='orange')
        axes[1, 0].axhline(y=self.best_roc_auc, color='r', linestyle='--', 
                           label=f'Best: {self.best_roc_auc:.4f}')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('ROC-AUC')
        axes[1, 0].set_title('Validation ROC-AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['learning_rate'], label='Learning Rate', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        os.makedirs(self.config.plots_dir, exist_ok=True)
        plt.savefig(os.path.join(self.config.plots_dir, 'training_history.png'), 
                    dpi=300, bbox_inches='tight')
        print(f"Training history saved to {self.config.plots_dir}/training_history.png")
        plt.close()


# ==============================
# EVALUATION
# ==============================
def evaluate_model(model, test_loader, config):
    """
    Evaluate model on test set.
    
    Args:
        model: trained model
        test_loader: test data loader
        config: Config object
    
    Returns:
        metrics dict
    """
    model.eval()
    all_outputs = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(config.device)
            outputs, _ = model(X_batch)
            
            all_outputs.append(outputs.cpu())
            all_labels.append(y_batch)
    
    # Concatenate
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    # Calculate metrics
    y_proba = all_outputs.numpy().flatten()
    y_pred = (y_proba >= 0.5).astype(int)
    y_true = all_labels.numpy().flatten()
    
    # Comprehensive metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    medical_metrics = calculate_medical_metrics(y_true, y_pred, y_proba)
    
    # Print reports
    print_metrics_report(metrics, "Test Set Evaluation")
    print_medical_report(medical_metrics, "Medical Safety Report")
    
    # Generate plots
    os.makedirs(config.plots_dir, exist_ok=True)
    plot_all_metrics(y_true, y_pred, y_proba, config.plots_dir)
    
    # Save metrics
    save_metrics_json(metrics, os.path.join(config.save_dir, 'test_metrics.json'))
    save_metrics_json(medical_metrics, os.path.join(config.save_dir, 'medical_metrics.json'))
    
    return metrics, medical_metrics


# ==============================
# MAIN TRAINING PIPELINE
# ==============================
def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("VitalX LSTM Training Pipeline")
    print("=" * 70)
    
    # Check CUDA availability first
    if not torch.cuda.is_available():
        print("\n" + "="*70)
        print("ERROR: CUDA GPU NOT AVAILABLE")
        print("="*70)
        print("This training script requires CUDA GPU.")
        print("\nYour current PyTorch version:", torch.__version__)
        print("\nTo install PyTorch with CUDA support, run:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("="*70)
        return
    
    print(f"\n✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    # Configuration
    config = Config()
    
    # Set random seeds
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    
    # Load data
    print("\n[1/7] Loading data...")
    X_path = os.path.join(config.data_path, config.X_file)
    y_path = os.path.join(config.data_path, config.y_file)
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Data files not found!")
        print(f"  Expected: {X_path}")
        print(f"  Expected: {y_path}")
        print(f"\nPlease run training/generate_dataset.py first to create the dataset.")
        return
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Positive cases: {y.sum()} ({100 * y.sum() / len(y):.2f}%)")
    
    # Split data
    print("\n[2/7] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
        X, y,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    # Create data loaders
    print("\n[3/7] Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config.batch_size,
        use_balanced_sampling=config.use_balanced_sampling,
        use_augmentation=config.use_augmentation
    )
    
    # Create model
    print("\n[4/7] Initializing LSTM model with attention...")
    model = LSTMAttentionModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n[5/7] Training model...")
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader)
    
    # Plot training history
    print("\n[6/7] Plotting training history...")
    trainer.plot_history()
    
    # Load best model for evaluation
    print("\n[7/7] Loading best model and evaluating...")
    best_model_path = os.path.join(config.save_dir, 'best_model.pth')
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    test_metrics, medical_metrics = evaluate_model(model, test_loader, config)
    
    # Train fallback logistic regression
    print("\n[BONUS] Training Logistic Regression fallback...")
    lr_model = LogisticRegressionFallback()
    lr_model.fit(X_train, y_train)
    
    # Evaluate fallback
    y_proba_lr = lr_model.predict_proba(X_test)
    y_pred_lr = (y_proba_lr >= 0.5).astype(int)
    lr_metrics = calculate_metrics(y_test, y_pred_lr, y_proba_lr)
    print_metrics_report(lr_metrics, "Logistic Regression Test Evaluation")
    
    # Save fallback model
    lr_model.save(os.path.join(config.save_dir, 'fallback_logistic.pkl'))
    
    # Final summary
    print("\n" + "=" * 70)
    print("Training Pipeline Complete!")
    print("=" * 70)
    print(f"\nModel Performance Summary:")
    print(f"  LSTM ROC-AUC:       {test_metrics.get('roc_auc', 0.0):.4f}")
    print(f"  LSTM Recall:        {test_metrics['recall']:.4f}")
    print(f"  LSTM Precision:     {test_metrics['precision']:.4f}")
    print(f"\nFallback Performance:")
    print(f"  LR ROC-AUC:         {lr_metrics.get('roc_auc', 0.0):.4f}")
    print(f"  LR Recall:          {lr_metrics['recall']:.4f}")
    print(f"  LR Precision:       {lr_metrics['precision']:.4f}")
    print(f"\nSaved artifacts:")
    print(f"  Model:              {config.save_dir}best_model.pth")
    print(f"  Fallback:           {config.save_dir}fallback_logistic.pkl")
    print(f"  Metrics:            {config.save_dir}test_metrics.json")
    print(f"  Plots:              {config.plots_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
