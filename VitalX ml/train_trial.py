"""
Trial Training Script - Uses small subset for quick testing
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import os
from tqdm import tqdm
import json
import glob

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


def train_model_trial(train_dataset, val_dataset, pos_weight=None, num_epochs=3):
    """
    Trial training function - only a few epochs
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
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
    
    # Model already has sigmoid, so use regular BCE
    criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRIAL TRAINING - {num_epochs} EPOCHS")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
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
        train_metrics = compute_metrics(train_labels, train_preds)
        val_metrics = compute_metrics(val_labels, val_preds)
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print_metrics(train_metrics, "Training")
        print_metrics(val_metrics, "Validation")
    
    print(f"\n{'='*60}")
    print("✓ TRIAL COMPLETED - NO ERRORS!")
    print(f"{'='*60}")
    
    return model


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VitalX SEPSIS PREDICTION - TRIAL TRAINING (SUBSET)")
    print("="*60)
    
    # Print feature summary
    config.print_feature_summary()
    
    # Load small subset of data
    print("\n" + "="*60)
    print("LOADING SMALL SUBSET FOR TRIAL")
    print("="*60)
    
    data_dirs = [config.DATA_DIR_A, config.DATA_DIR_B]
    existing_dirs = [d for d in data_dirs if os.path.exists(d)]
    
    if not existing_dirs:
        print(f"Error: No data directories found!")
        exit(1)
    
    # Get only first 500 patient files for quick trial
    all_files = []
    for data_dir in existing_dirs:
        pattern = os.path.join(data_dir, '*.psv')
        files = sorted(glob.glob(pattern))
        all_files.extend(files)
    
    # Use only first 500 files
    subset_files = all_files[:500]
    print(f"\nUsing {len(subset_files)} patient files for trial (out of {len(all_files)} total)")
    
    # Create temporary directory and copy subset
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp(prefix='vitalx_trial_')
    print(f"Temporary directory: {temp_dir}")
    
    for i, src_file in enumerate(subset_files):
        if (i + 1) % 100 == 0:
            print(f"  Copying file {i+1}/{len(subset_files)}...")
        dst_file = os.path.join(temp_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dst_file)
    
    print(f"\n✓ Subset prepared in {temp_dir}")
    
    # Load dataset from subset
    print("\nLoading full dataset from subset...")
    full_dataset = SepsisDataset(
        data_dirs=[temp_dir],
        sequence_length=config.SEQUENCE_LENGTH,
        mode='train'
    )
    
    # Calculate class weights
    pos_weight = full_dataset.get_class_weights()
    print(f"\nClass imbalance - Positive weight: {pos_weight:.2f}")
    
    # Split dataset into train/val (80/20)
    print("\nSplitting dataset...")
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
    
    # Train model for 3 epochs (trial)
    try:
        model = train_model_trial(train_dataset, val_dataset, pos_weight=pos_weight, num_epochs=3)
        print("\n" + "="*60)
        print("✓ TRIAL SUCCESSFUL - Ready for full training!")
        print("="*60)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR DETECTED: {type(e).__name__}")
        print(f"{'='*60}")
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temporary directory
        print(f"\nCleaning up temporary directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Done!")
