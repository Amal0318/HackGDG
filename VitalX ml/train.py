"""
Training Script for VitalX Sepsis Prediction Model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import json

import config
from dataset import SepsisDataset
from model import create_model, save_model
from utils import (
    compute_metrics, 
    print_metrics, 
    save_config, 
    set_seed, 
    EarlyStopping
)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    return avg_loss, all_preds, all_labels


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for sequences, labels in pbar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    return avg_loss, all_preds, all_labels


def train_model(train_dataset, val_dataset, scaler=None, pos_weight=None, save_dir=config.OUTPUT_DIR):
    """
    Main training function
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        scaler: Fitted StandardScaler from full dataset
        pos_weight: Positive class weight for handling imbalance
        save_dir: Directory to save outputs
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    input_size = config.get_feature_count()
    model = create_model(input_size, device)
    
    # Loss function with class weights
    if pos_weight is None:
        pos_weight = 1.0
    print(f"\nPositive class weight: {pos_weight:.2f}")
    
    # Use Weighted BCE Loss
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # We need to modify model output for BCEWithLogitsLoss (remove sigmoid)
    # Or use regular BCELoss with current model
    criterion = nn.BCELoss()  # Model already has sigmoid
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auroc': [],
        'val_auprc': []
    }
    
    best_auroc = 0.0
    best_epoch = 0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Compute metrics
        val_metrics = compute_metrics(val_labels, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_auprc'].append(val_metrics['auprc'])
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val AUROC:  {val_metrics['auroc']:.4f}")
        print(f"Val AUPRC:  {val_metrics['auprc']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            best_epoch = epoch + 1
            save_model(model, config.MODEL_SAVE_PATH)
            print(f"[OK] Best model saved (AUROC: {best_auroc:.4f})")
        
        # Checkpoint
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, 
            f'checkpoint_epoch_{epoch + 1}.pth'
        )
        save_model(model, checkpoint_path)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best AUROC: {best_auroc:.4f} (Epoch {best_epoch})")
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")
    
    # Save scaler
    if scaler is not None:
        import pickle
        with open(config.SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {config.SCALER_SAVE_PATH}")
    
    # Save feature configuration
    feature_config = {
        'features': config.ALL_FEATURES,
        'n_features': len(config.ALL_FEATURES),
        'sequence_length': config.SEQUENCE_LENGTH,
        'base_features': config.BASE_FEATURES,
        'derived_features': config.DERIVED_FEATURES,
        'missingness_features': config.MISSINGNESS_FEATURES
    }
    save_config(feature_config, config.CONFIG_SAVE_PATH)
    
    print("\nAll artifacts saved successfully!")
    
    return model, history


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VitalX SEPSIS PREDICTION - TRAINING")
    print("="*60)
    
    # Print feature summary
    config.print_feature_summary()
    
    # Load data
    print("\nPreparing datasets...")
    data_dirs = [config.DATA_DIR_A, config.DATA_DIR_B]
    
    # Check if directories exist
    existing_dirs = [d for d in data_dirs if os.path.exists(d)]
    if not existing_dirs:
        print(f"Error: No data directories found!")
        print(f"Looking for: {data_dirs}")
        exit(1)
    
    print(f"Using data directories: {existing_dirs}")
    
    # Load data once and split efficiently
    print("\nLoading full dataset...")
    full_dataset = SepsisDataset(
        data_dirs=existing_dirs,
        sequence_length=config.SEQUENCE_LENGTH,
        mode='train'
    )
    
    # Calculate class weights from full dataset before splitting
    pos_weight = full_dataset.get_class_weights()
    print(f"\nClass imbalance - Positive weight: {pos_weight:.2f}")
    
    # Split dataset into train/val (80/20 split)
    print("\nSplitting dataset into train and validation...")
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Train model
    model, history = train_model(
        train_dataset, 
        val_dataset, 
        scaler=full_dataset.get_scaler(),
        pos_weight=pos_weight
    )
    
    print("\n[OK] Training pipeline completed successfully!")
