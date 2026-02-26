"""
Main Pipeline for VitalX Sepsis Prediction
Demonstrates complete workflow from data to inference
"""

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import config
from dataset import SepsisDataset
from model import create_model, save_model
from train import train_model
from evaluate import evaluate_model
from inference import SepsisPredictor
from utils import set_seed, print_metrics
from torch.utils.data import DataLoader
import torch


def prepare_data_splits(data_dirs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Prepare train/val/test splits at the patient level
    
    Returns:
        Lists of file paths for train, val, test
    """
    # Get all patient files
    all_files = []
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            pattern = os.path.join(data_dir, '*.psv')
            files = glob.glob(pattern)
            all_files.extend(files)
    
    if len(all_files) == 0:
        raise ValueError("No PSV files found in data directories!")
    
    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(all_files)
    
    # Split
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    print("\n" + "="*60)
    print("DATA SPLIT")
    print("="*60)
    print(f"Total patients: {n_total}")
    print(f"  Train: {len(train_files)} patients ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_files)} patients ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_files)} patients ({test_ratio*100:.0f}%)")
    print("="*60 + "\n")
    
    return train_files, val_files, test_files


def create_temp_dirs_for_splits(train_files, val_files, test_files, base_dir='temp_splits'):
    """
    Create temporary directories with split files
    (Creates symbolic links or copies)
    """
    import tempfile
    import shutil
    
    temp_base = tempfile.mkdtemp(prefix='vitalx_')
    
    train_dir = os.path.join(temp_base, 'train')
    val_dir = os.path.join(temp_base, 'val')
    test_dir = os.path.join(temp_base, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy files (or create symlinks on Unix)
    for f in train_files:
        dest = os.path.join(train_dir, os.path.basename(f))
        shutil.copy2(f, dest)
    
    for f in val_files:
        dest = os.path.join(val_dir, os.path.basename(f))
        shutil.copy2(f, dest)
    
    for f in test_files:
        dest = os.path.join(test_dir, os.path.basename(f))
        shutil.copy2(f, dest)
    
    return train_dir, val_dir, test_dir, temp_base


def run_complete_pipeline(use_subset=False):
    """
    Run the complete ML pipeline
    
    Args:
        use_subset: If True, use only a small subset for quick testing
    """
    print("\n" + "="*60)
    print("VitalX SEPSIS PREDICTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Set seed
    set_seed(42)
    
    # Print feature configuration
    config.print_feature_summary()
    
    # Step 1: Prepare data
    print("\n--- STEP 1: Prepare Data Splits ---")
    data_dirs = [config.DATA_DIR_A, config.DATA_DIR_B]
    existing_dirs = [d for d in data_dirs if os.path.exists(d)]
    
    if not existing_dirs:
        print("ERROR: No data directories found!")
        print(f"Looking for: {data_dirs}")
        return
    
    # Get all files and split
    train_files, val_files, test_files = prepare_data_splits(existing_dirs)
    
    # If using subset, limit files
    if use_subset:
        print("\n[WARNING] Using SUBSET mode (for quick testing)")
        train_files = train_files[:20]
        val_files = val_files[:5]
        test_files = test_files[:5]
        print(f"  Train: {len(train_files)} patients")
        print(f"  Val:   {len(val_files)} patients")
        print(f"  Test:  {len(test_files)} patients\n")
    
    # Create temporary directories for splits
    print("Creating temporary directories for data splits...")
    train_dir, val_dir, test_dir, temp_base = create_temp_dirs_for_splits(
        train_files, val_files, test_files
    )
    
    try:
        # Step 2: Load datasets
        print("\n--- STEP 2: Load and Preprocess Data ---")
        
        print("\nLoading training data...")
        train_dataset = SepsisDataset(
            data_dirs=[train_dir],
            sequence_length=config.SEQUENCE_LENGTH,
            mode='train'
        )
        
        print("\nLoading validation data...")
        val_dataset = SepsisDataset(
            data_dirs=[val_dir],
            sequence_length=config.SEQUENCE_LENGTH,
            scaler=train_dataset.get_scaler(),
            mode='val'
        )
        
        print("\nLoading test data...")
        test_dataset = SepsisDataset(
            data_dirs=[test_dir],
            sequence_length=config.SEQUENCE_LENGTH,
            scaler=train_dataset.get_scaler(),
            mode='test'
        )
        
        # Step 3: Train model
        print("\n--- STEP 3: Train Model ---")
        model, history = train_model(train_dataset, val_dataset)
        
        # Step 4: Evaluate on test set
        print("\n--- STEP 4: Evaluate on Test Set ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        from evaluate import evaluate_model, save_evaluation_report
        results = evaluate_model(model, test_loader, device)
        
        print_metrics(results['metrics'], phase='Test')
        
        # Save evaluation report
        report_path = os.path.join(config.OUTPUT_DIR, 'test_evaluation.json')
        save_evaluation_report(results, report_path)
        
        # Step 5: Inference example
        print("\n--- STEP 5: Inference Example ---")
        predictor = SepsisPredictor()
        
        # Test with a random sequence
        dummy_sequence = np.random.randn(config.SEQUENCE_LENGTH, config.get_feature_count())
        probability = predictor.predict(dummy_sequence)
        print(f"\nExample prediction on random data:")
        print(f"  Sepsis probability: {probability:.4f}")
        
        # Test with actual patient file if available
        if len(test_files) > 0:
            test_file = test_files[0]
            print(f"\nExample prediction on patient file:")
            print(f"  File: {os.path.basename(test_file)}")
            try:
                probability, sequence = predictor.predict_from_patient_file(test_file)
                print(f"  Sepsis probability: {probability:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\n" + "="*60)
        print("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
        print("  - model.pth")
        print("  - scaler.pkl")
        print("  - feature_config.json")
        print("  - training_history.json")
        print("  - test_evaluation.json")
        print("="*60 + "\n")
        
    finally:
        # Cleanup temporary directories
        import shutil
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_base)
        print("Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VitalX Sepsis Prediction Pipeline')
    parser.add_argument('--subset', action='store_true', 
                       help='Use only a small subset of data for quick testing')
    
    args = parser.parse_args()
    
    run_complete_pipeline(use_subset=args.subset)
