"""
Dataset class for VitalX Sepsis Prediction
Handles loading, preprocessing, and windowing of patient time-series data
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle
import glob
import os
from typing import List, Tuple, Optional

import config


class SepsisDataset(Dataset):
    """
    PyTorch Dataset for sepsis prediction from PhysioNet 2019 format
    
    Handles:
    - Loading PSV files
    - Forward-filling missing values per patient
    - Creating missingness mask features
    - Deriving additional features
    - Normalization
    - Creating sliding windows
    """
    
    def __init__(self, 
                 data_dirs: List[str],
                 sequence_length: int = config.SEQUENCE_LENGTH,
                 scaler: Optional[StandardScaler] = None,
                 mode: str = 'train'):
        """
        Args:
            data_dirs: List of directories containing PSV files
            sequence_length: Length of time-series sequences (hours)
            scaler: Pre-fitted StandardScaler (for val/test)
            mode: 'train', 'val', or 'test'
        """
        self.data_dirs = data_dirs
        self.sequence_length = sequence_length
        self.mode = mode
        self.scaler = scaler
        
        # Load all patient data
        self.patient_data = []
        self.sequences = []
        self.labels = []
        
        print(f"\n{'='*60}")
        print(f"Loading {mode.upper()} data...")
        print(f"{'='*60}")
        
        # Load raw data
        self._load_patients()
        
        # Preprocess
        self._preprocess()
        
        # Create sliding windows
        self._create_windows()
        
        print(f"Total sequences: {len(self.sequences)}")
        print(f"Positive samples: {sum(self.labels)}")
        print(f"Negative samples: {len(self.labels) - sum(self.labels)}")
        print(f"Class imbalance ratio: {len(self.labels) / max(sum(self.labels), 1):.2f}:1")
        print(f"{'='*60}\n")
    
    def _load_patients(self):
        """Load all patient PSV files"""
        all_files = []
        for data_dir in self.data_dirs:
            pattern = os.path.join(data_dir, '*.psv')
            files = glob.glob(pattern)
            all_files.extend(files)
        
        print(f"Found {len(all_files)} patient files")
        
        for i, filepath in enumerate(all_files):
            if (i + 1) % 100 == 0:
                print(f"  Loading patient {i + 1}/{len(all_files)}...")
            
            try:
                # Load PSV file
                df = pd.read_csv(filepath, sep='|')
                
                # Keep only if patient has data
                if len(df) > 0:
                    self.patient_data.append(df)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        print(f"Successfully loaded {len(self.patient_data)} patients")
    
    def _preprocess(self):
        """Preprocess all patient data"""
        print("Preprocessing patients...")
        
        processed_patients = []
        
        for i, patient_df in enumerate(self.patient_data):
            if (i + 1) % 100 == 0:
                print(f"  Processing patient {i + 1}/{len(self.patient_data)}...")
            
            # Create a copy
            df = patient_df.copy()
            
            # Extract base features
            base_features = [f for f in config.BASE_FEATURES if f in df.columns]
            
            # Create missingness indicators BEFORE forward-filling
            for feat in config.CORE_VITALS + config.KEY_LABS:
                if feat in df.columns:
                    df[f'{feat}_missing'] = df[feat].isna().astype(int)
            
            # Forward-fill missing values per patient
            df[base_features] = df[base_features].ffill()
            
            # Backward-fill for any remaining NaNs at the start
            df[base_features] = df[base_features].bfill()
            
            # Fill any remaining NaNs with 0
            df[base_features] = df[base_features].fillna(0)
            
            # Derive additional features
            df = self._derive_features(df)
            
            # Keep only required columns
            feature_cols = config.ALL_FEATURES
            available_features = [f for f in feature_cols if f in df.columns]
            
            # Add target column
            if config.TARGET_COLUMN in df.columns:
                df_processed = df[available_features + [config.TARGET_COLUMN]].copy()
                processed_patients.append(df_processed)
        
        self.patient_data = processed_patients
        print(f"Preprocessing complete")
    
    def _derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive additional features from base vitals"""
        
        # Shock Index = HR / SBP
        if 'HR' in df.columns and 'SBP' in df.columns:
            df['ShockIndex'] = df['HR'] / (df['SBP'] + 1e-6)  # Add small epsilon to avoid division by zero
            df['ShockIndex'] = df['ShockIndex'].replace([np.inf, -np.inf], 0)
        else:
            df['ShockIndex'] = 0
        
        # HR delta (change from previous hour)
        if 'HR' in df.columns:
            df['HR_delta'] = df['HR'].diff().fillna(0)
        else:
            df['HR_delta'] = 0
        
        # SBP delta
        if 'SBP' in df.columns:
            df['SBP_delta'] = df['SBP'].diff().fillna(0)
        else:
            df['SBP_delta'] = 0
        
        # ShockIndex delta
        if 'ShockIndex' in df.columns:
            df['ShockIndex_delta'] = df['ShockIndex'].diff().fillna(0)
        else:
            df['ShockIndex_delta'] = 0
        
        # Rolling mean features
        if 'HR' in df.columns:
            df['RollingMean_HR'] = df['HR'].rolling(window=config.ROLLING_WINDOW, min_periods=1).mean()
        else:
            df['RollingMean_HR'] = 0
        
        if 'SBP' in df.columns:
            df['RollingMean_SBP'] = df['SBP'].rolling(window=config.ROLLING_WINDOW, min_periods=1).mean()
        else:
            df['RollingMean_SBP'] = 0
        
        return df
    
    def _create_windows(self):
        """Create sliding windows from patient time-series"""
        print("Creating sliding windows...")
        
        for patient_df in self.patient_data:
            n_hours = len(patient_df)
            
            # Create windows with sufficient history
            for end_idx in range(self.sequence_length, n_hours + 1):
                start_idx = end_idx - self.sequence_length
                
                # Extract sequence
                window = patient_df.iloc[start_idx:end_idx]
                
                # Features
                feature_cols = config.ALL_FEATURES
                available_cols = [c for c in feature_cols if c in window.columns]
                
                sequence_features = window[available_cols].values
                
                # Ensure all features are present (pad with zeros if missing)
                if len(available_cols) < len(config.ALL_FEATURES):
                    padded = np.zeros((self.sequence_length, len(config.ALL_FEATURES)))
                    padded[:, :len(available_cols)] = sequence_features
                    sequence_features = padded
                
                # Target (label at current time point)
                label = window[config.TARGET_COLUMN].iloc[-1]
                
                self.sequences.append(sequence_features)
                self.labels.append(label)
        
        # Convert to numpy arrays
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        
        # Normalize features
        if self.mode == 'train':
            # Fit scaler on training data
            self.scaler = StandardScaler()
            # Reshape for fitting: (n_samples * sequence_length, n_features)
            n_samples, seq_len, n_features = self.sequences.shape
            sequences_reshaped = self.sequences.reshape(-1, n_features)
            self.scaler.fit(sequences_reshaped)
        
        # Transform sequences
        if self.scaler is not None:
            n_samples, seq_len, n_features = self.sequences.shape
            sequences_reshaped = self.sequences.reshape(-1, n_features)
            sequences_normalized = self.scaler.transform(sequences_reshaped)
            self.sequences = sequences_normalized.reshape(n_samples, seq_len, n_features)
    
    def get_scaler(self):
        """Return the fitted scaler"""
        return self.scaler
    
    def get_class_weights(self):
        """Compute class weights for handling imbalance"""
        n_samples = len(self.labels)
        n_positive = sum(self.labels)
        n_negative = n_samples - n_positive
        
        if n_positive == 0:
            return 1.0
        
        # Weight for positive class
        pos_weight = n_negative / n_positive
        return pos_weight
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            sequence: (sequence_length, n_features)
            label: scalar (0 or 1)
        """
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.FloatTensor([self.labels[idx]])
        
        return sequence, label
    
    def save_scaler(self, filepath):
        """Save the fitted scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {filepath}")
    
    @staticmethod
    def load_scaler(filepath):
        """Load a saved scaler"""
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        return scaler


def split_patient_files(data_dirs: List[str], 
                       train_ratio: float = config.TRAIN_RATIO,
                       val_ratio: float = config.VAL_RATIO,
                       test_ratio: float = config.TEST_RATIO,
                       seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patient files into train/val/test sets
    
    Returns:
        train_files, val_files, test_files
    """
    # Get all patient files
    all_files = []
    for data_dir in data_dirs:
        pattern = os.path.join(data_dir, '*.psv')
        files = glob.glob(pattern)
        all_files.extend(files)
    
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
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} patients")
    print(f"  Val:   {len(val_files)} patients")
    print(f"  Test:  {len(test_files)} patients")
    print(f"  Total: {len(all_files)} patients\n")
    
    return train_files, val_files, test_files


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    config.print_feature_summary()
    
    # Just test with a small subset
    test_files = glob.glob(os.path.join(config.DATA_DIR_A, '*.psv'))[:10]
    
    # Create temporary directory structure
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    
    # Copy test files
    for f in test_files:
        shutil.copy(f, temp_dir)
    
    # Create dataset
    dataset = SepsisDataset(
        data_dirs=[temp_dir],
        sequence_length=24,
        mode='train'
    )
    
    print(f"\nDataset created successfully!")
    print(f"Number of sequences: {len(dataset)}")
    print(f"Feature shape: {dataset[0][0].shape}")
    print(f"Label shape: {dataset[0][1].shape}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
