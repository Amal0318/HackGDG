"""
VitalX LSTM Training Script - QUICK VERSION
============================================
Reduced epochs for faster training/testing
REQUIRES: CUDA GPU (no CPU support)
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main training module
from train import *

# Override configuration for quick training
class QuickConfig(Config):
    """Quick training configuration."""
    epochs = 5  # Reduced from 50
    batch_size = 64  # Increased for faster training
    patience = 3  # Reduced early stopping patience

if __name__ == "__main__":
    print("=" * 70)
    print("VitalX LSTM QUICK Training Pipeline (5 epochs)")
    print("=" * 70)
    
    # Check CUDA availability first
    if not torch.cuda.is_available():
        print("\\n" + "="*70)
        print("ERROR: CUDA GPU NOT AVAILABLE")
        print("="*70)
        print("This training script requires CUDA GPU.")
        print("\\nYour current PyTorch version:", torch.__version__)
        print("\\nTo install PyTorch with CUDA support, run:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("="*70)
        sys.exit(1)
    
    print(f"\\n✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\\n\")
    
    # Use quick config
    config = QuickConfig()
    
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
        sys.exit(1)
    
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
    print("\n[5/7] Training model (QUICK - 5 epochs)...")
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
    print("QUICK Training Pipeline Complete!")
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
    print("\n  Note: This was a QUICK training (5 epochs)")
    print("  For production use, run full train.py (50 epochs)")
    print("=" * 70)
