"""
Quick script to save the scaler without retraining
"""

import pickle
import os
import config
from dataset import SepsisDataset

print("\n" + "="*60)
print("SAVING SCALER FROM DATASET")
print("="*60)

# Load data
data_dirs = [config.DATA_DIR_A, config.DATA_DIR_B]
existing_dirs = [d for d in data_dirs if os.path.exists(d)]

if not existing_dirs:
    print("Error: No data directories found!")
    exit(1)

print(f"\nUsing data directories: {existing_dirs}")

# Load full dataset (this will fit the scaler)
print("\nLoading dataset to generate scaler...")
full_dataset = SepsisDataset(
    data_dirs=existing_dirs,
    sequence_length=config.SEQUENCE_LENGTH,
    mode='train'
)

# Get scaler
scaler = full_dataset.get_scaler()

# Save scaler
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
with open(config.SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n[OK] Scaler saved to {config.SCALER_SAVE_PATH}")

# Also save feature config if missing
feature_config = {
    'features': config.ALL_FEATURES,
    'n_features': len(config.ALL_FEATURES),
    'sequence_length': config.SEQUENCE_LENGTH,
    'base_features': config.BASE_FEATURES,
    'derived_features': config.DERIVED_FEATURES,
    'missingness_features': config.MISSINGNESS_FEATURES
}

import json
with open(config.CONFIG_SAVE_PATH, 'w') as f:
    json.dump(feature_config, f, indent=4)

print(f"[OK] Feature config saved to {config.CONFIG_SAVE_PATH}")

print("\n" + "="*60)
print("ALL ARTIFACTS READY!")
print("="*60)
print("\nYour trained model is ready to use:")
print(f"  Model:     {config.MODEL_SAVE_PATH}")
print(f"  Scaler:    {config.SCALER_SAVE_PATH}")
print(f"  Config:    {config.CONFIG_SAVE_PATH}")
print("\nYou can now use: python inference.py")
