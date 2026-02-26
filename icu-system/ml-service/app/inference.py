"""
Inference Wrapper for VitalX Sepsis Prediction Model
Simple, clean interface for making predictions
"""

import numpy as np
import pandas as pd
import torch
import pickle
import json
import os

import config
from model import load_model


class SepsisPredictor:
    """
    Inference wrapper for sepsis prediction
    
    Usage:
        predictor = SepsisPredictor()
        probability = predictor.predict(patient_sequence)
    """
    
    def __init__(self, 
                 model_path=config.MODEL_SAVE_PATH,
                 scaler_path=config.SCALER_SAVE_PATH,
                 config_path=config.CONFIG_SAVE_PATH):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            config_path: Path to feature configuration
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path, self.device)
        self.model.eval()
        
        # Load scaler
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature config
        print(f"Loading feature configuration from {config_path}...")
        with open(config_path, 'r') as f:
            self.feature_config = json.load(f)
        
        self.features = self.feature_config['features']
        self.n_features = self.feature_config['n_features']
        self.sequence_length = self.feature_config['sequence_length']
        
        print(f"\nPredictor initialized successfully!")
        print(f"  Device: {self.device}")
        print(f"  Features: {self.n_features}")
        print(f"  Sequence length: {self.sequence_length} hours")
    
    def preprocess_sequence(self, sequence):
        """
        Preprocess input sequence
        
        Args:
            sequence: numpy array of shape (sequence_length, n_features) or
                     pandas DataFrame with feature columns
        
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert DataFrame to numpy if needed
        if isinstance(sequence, pd.DataFrame):
            sequence = sequence[self.features].values
        
        # Ensure correct shape
        if sequence.shape != (self.sequence_length, self.n_features):
            raise ValueError(
                f"Expected shape ({self.sequence_length}, {self.n_features}), "
                f"got {sequence.shape}"
            )
        
        # Normalize using fitted scaler
        sequence_normalized = self.scaler.transform(sequence)
        
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0)
        sequence_tensor = sequence_tensor.to(self.device)
        
        return sequence_tensor
    
    def predict(self, sequence):
        """
        Predict sepsis probability
        
        Args:
            sequence: Input sequence (sequence_length, n_features)
                     Can be numpy array or pandas DataFrame
        
        Returns:
            probability: Float between 0 and 1
        """
        # Preprocess
        sequence_tensor = self.preprocess_sequence(sequence)
        
        # Predict
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probability = output.item()
        
        return probability
    
    def predict_batch(self, sequences):
        """
        Predict for multiple sequences
        
        Args:
            sequences: List of sequences or numpy array (batch_size, seq_len, n_features)
        
        Returns:
            probabilities: numpy array of probabilities
        """
        # Convert list of sequences to batch
        if isinstance(sequences, list):
            sequences = np.array(sequences)
        
        # Normalize
        batch_size, seq_len, n_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, n_features)
        sequences_normalized = self.scaler.transform(sequences_reshaped)
        sequences_normalized = sequences_normalized.reshape(batch_size, seq_len, n_features)
        
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences_normalized).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(sequences_tensor)
            probabilities = outputs.cpu().numpy().flatten()
        
        return probabilities
    
    def predict_from_patient_file(self, filepath, start_hour=None):
        """
        Predict from a patient PSV file
        
        Args:
            filepath: Path to patient PSV file
            start_hour: Starting hour for sequence (default: latest 24 hours)
        
        Returns:
            probability: Sepsis probability
            sequence: The input sequence used
        """
        # Load patient data
        df = pd.read_csv(filepath, sep='|')
        
        # Use latest sequence if start_hour not specified
        if start_hour is None:
            start_hour = max(0, len(df) - self.sequence_length)
        
        end_hour = start_hour + self.sequence_length
        
        if end_hour > len(df):
            raise ValueError(
                f"Not enough data. Patient has {len(df)} hours, "
                f"need at least {end_hour} for prediction."
            )
        
        # Extract sequence
        sequence_df = df.iloc[start_hour:end_hour]
        
        # Preprocess sequence (similar to dataset.py)
        sequence_df = self._preprocess_patient_sequence(sequence_df)
        
        # Predict
        probability = self.predict(sequence_df)
        
        return probability, sequence_df
    
    def _preprocess_patient_sequence(self, df):
        """Preprocess a raw patient sequence (mimics dataset preprocessing)"""
        df = df.copy()
        
        # Extract base features
        base_features = [f for f in config.BASE_FEATURES if f in df.columns]
        
        # Create missingness indicators
        for feat in config.CORE_VITALS + config.KEY_LABS:
            if feat in df.columns:
                df[f'{feat}_missing'] = df[feat].isna().astype(int)
        
        # Forward-fill
        df[base_features] = df[base_features].ffill()
        df[base_features] = df[base_features].bfill()
        df[base_features] = df[base_features].fillna(0)
        
        # Derive features
        df = self._derive_features(df)
        
        # Select features
        available_features = [f for f in self.features if f in df.columns]
        
        return df[available_features]
    
    def _derive_features(self, df):
        """Derive additional features (same as dataset.py)"""
        # Shock Index
        if 'HR' in df.columns and 'SBP' in df.columns:
            df['ShockIndex'] = df['HR'] / (df['SBP'] + 1e-6)
            df['ShockIndex'] = df['ShockIndex'].replace([np.inf, -np.inf], 0)
        else:
            df['ShockIndex'] = 0
        
        # Deltas
        if 'HR' in df.columns:
            df['HR_delta'] = df['HR'].diff().fillna(0)
        else:
            df['HR_delta'] = 0
        
        if 'SBP' in df.columns:
            df['SBP_delta'] = df['SBP'].diff().fillna(0)
        else:
            df['SBP_delta'] = 0
        
        if 'ShockIndex' in df.columns:
            df['ShockIndex_delta'] = df['ShockIndex'].diff().fillna(0)
        else:
            df['ShockIndex_delta'] = 0
        
        # Rolling means
        if 'HR' in df.columns:
            df['RollingMean_HR'] = df['HR'].rolling(window=config.ROLLING_WINDOW, min_periods=1).mean()
        else:
            df['RollingMean_HR'] = 0
        
        if 'SBP' in df.columns:
            df['RollingMean_SBP'] = df['SBP'].rolling(window=config.ROLLING_WINDOW, min_periods=1).mean()
        else:
            df['RollingMean_SBP'] = 0
        
        return df


def predict(sequence_tensor):
    """
    Simple prediction function (standalone)
    
    Args:
        sequence_tensor: Preprocessed tensor (1, sequence_length, n_features)
    
    Returns:
        probability: Float between 0 and 1
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config.MODEL_SAVE_PATH, device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        sequence_tensor = sequence_tensor.to(device)
        output = model(sequence_tensor)
        probability = output.item()
    
    return probability


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("VitalX SEPSIS PREDICTION - INFERENCE")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("\nError: Model not found. Please train the model first.")
        exit(1)
    
    # Initialize predictor
    predictor = SepsisPredictor()
    
    # Example 1: Predict from random data
    print("\n" + "-"*60)
    print("Example 1: Predict from random sequence")
    print("-"*60)
    
    # Create dummy sequence
    dummy_sequence = np.random.randn(config.SEQUENCE_LENGTH, config.get_feature_count())
    
    probability = predictor.predict(dummy_sequence)
    print(f"Predicted sepsis probability: {probability:.4f}")
    
    # Example 2: Predict from patient file
    print("\n" + "-"*60)
    print("Example 2: Predict from patient file")
    print("-"*60)
    
    # Get first patient file
    import glob
    patient_files = glob.glob(os.path.join(config.DATA_DIR_A, '*.psv'))
    
    if len(patient_files) > 0:
        test_file = patient_files[0]
        print(f"Using file: {os.path.basename(test_file)}")
        
        try:
            probability, sequence = predictor.predict_from_patient_file(test_file)
            print(f"Predicted sepsis probability: {probability:.4f}")
            print(f"Sequence shape: {sequence.shape}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Inference example completed!")
    print("="*60 + "\n")
