"""
VitalX PyTorch Dataset Classes
===============================

Custom dataset classes for time-series ICU data.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import random


# ==============================
# MAIN DATASET CLASS
# ==============================
class VitalXDataset(Dataset):
    """
    PyTorch Dataset for VitalX time-series data.
    
    Args:
        X: numpy array of shape (samples, seq_len, features)
        y: numpy array of shape (samples,) - binary labels
        transform: optional transform to apply
    
    Returns:
        (sequence_tensor, label_tensor)
    """
    
    def __init__(self, X, y, transform=None):
        """
        Initialize dataset.
        
        Args:
            X: (samples, 60, 14) sequences
            y: (samples,) labels
            transform: optional data augmentation
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # (samples, 1)
        self.transform = transform
        
        # Validate shapes
        assert len(self.X) == len(self.y), "X and y must have same length"
        assert self.X.dim() == 3, "X must be 3D (samples, seq_len, features)"
        
        print(f"Dataset initialized: {len(self)} samples")
        print(f"  Sequence shape: {self.X.shape}")
        print(f"  Label shape: {self.y.shape}")
        print(f"  Positive cases: {torch.sum(self.y).item():.0f} ({100 * torch.mean(self.y).item():.2f}%)")
    
    def __len__(self):
        """Return number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: index
        
        Returns:
            sequence: (seq_len, features)
            label: scalar tensor
        """
        sequence = self.X[idx]
        label = self.y[idx]
        
        # Apply transform if provided
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label
    
    def get_batch(self, indices):
        """Get multiple samples by indices."""
        return self.X[indices], self.y[indices]
    
    def get_class_weights(self):
        """
        Calculate class weights for imbalanced data.
        
        Returns:
            weights: tensor of shape (2,) for [class_0, class_1]
        """
        n_samples = len(self.y)
        n_positive = torch.sum(self.y).item()
        n_negative = n_samples - n_positive
        
        # Weight inversely proportional to class frequency
        weight_positive = n_samples / (2 * n_positive) if n_positive > 0 else 1.0
        weight_negative = n_samples / (2 * n_negative) if n_negative > 0 else 1.0
        
        return torch.FloatTensor([weight_negative, weight_positive])


# ==============================
# BALANCED BATCH SAMPLER
# ==============================
class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures balanced batches.
    
    Each batch contains equal number of positive and negative samples.
    Useful for handling imbalanced datasets.
    """
    
    def __init__(self, labels, batch_size):
        """
        Initialize sampler.
        
        Args:
            labels: binary labels (0 or 1)
            batch_size: size of each batch (should be even)
        """
        self.labels = labels.squeeze() if isinstance(labels, torch.Tensor) else torch.FloatTensor(labels).squeeze()
        self.batch_size = batch_size
        
        # Get indices for each class
        self.positive_indices = torch.where(self.labels == 1)[0].tolist()
        self.negative_indices = torch.where(self.labels == 0)[0].tolist()
        
        # Number of samples per class in each batch
        self.n_per_class = batch_size // 2
        
        # Calculate number of batches
        self.n_batches = min(
            len(self.positive_indices) // self.n_per_class,
            len(self.negative_indices) // self.n_per_class
        )
        
        print(f"BalancedBatchSampler initialized:")
        print(f"  Batch size: {batch_size} ({self.n_per_class} per class)")
        print(f"  Positive samples: {len(self.positive_indices)}")
        print(f"  Negative samples: {len(self.negative_indices)}")
        print(f"  Number of batches: {self.n_batches}")
    
    def __iter__(self):
        """Generate balanced batches."""
        # Shuffle indices
        random.shuffle(self.positive_indices)
        random.shuffle(self.negative_indices)
        
        for i in range(self.n_batches):
            # Get balanced batch
            pos_batch = self.positive_indices[i * self.n_per_class:(i + 1) * self.n_per_class]
            neg_batch = self.negative_indices[i * self.n_per_class:(i + 1) * self.n_per_class]
            
            # Combine and shuffle
            batch = pos_batch + neg_batch
            random.shuffle(batch)
            
            yield batch
    
    def __len__(self):
        """Return number of batches."""
        return self.n_batches


# ==============================
# DATA AUGMENTATION
# ==============================
class TimeSeriesAugmentation:
    """
    Data augmentation for time-series data.
    
    Techniques:
    - Gaussian noise injection
    - Scaling
    - Time warping (future enhancement)
    """
    
    def __init__(self, noise_level=0.01, scale_range=(0.95, 1.05)):
        """
        Args:
            noise_level: standard deviation of Gaussian noise
            scale_range: (min, max) scaling factors
        """
        self.noise_level = noise_level
        self.scale_range = scale_range
    
    def __call__(self, sequence):
        """
        Apply augmentation to sequence.
        
        Args:
            sequence: (seq_len, features)
        
        Returns:
            augmented_sequence: (seq_len, features)
        """
        # Add Gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(sequence) * self.noise_level
            sequence = sequence + noise
        
        # Random scaling
        if random.random() > 0.5:
            scale = random.uniform(*self.scale_range)
            sequence = sequence * scale
        
        return sequence


# ==============================
# DATA SPLITTING
# ==============================
def create_data_splits(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: (samples, seq_len, features)
        y: (samples,)
        train_size: proportion for training
        val_size: proportion for validation
        test_size: proportion for testing
        random_state: random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=random_state
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(y_train)} samples ({100 * len(y_train) / len(y):.1f}%)")
    print(f"  Val:   {len(y_val)} samples ({100 * len(y_val) / len(y):.1f}%)")
    print(f"  Test:  {len(y_test)} samples ({100 * len(y_test) / len(y):.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, 
                       batch_size=32, use_balanced_sampling=False, 
                       use_augmentation=False, num_workers=0):
    """
    Create PyTorch DataLoaders for train, val, and test sets.
    
    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        X_test, y_test: test data
        batch_size: batch size
        use_balanced_sampling: use balanced batch sampler for training
        use_augmentation: apply data augmentation to training
        num_workers: number of worker processes
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Data augmentation for training
    transform = TimeSeriesAugmentation() if use_augmentation else None
    
    # Create datasets
    train_dataset = VitalXDataset(X_train, y_train, transform=transform)
    val_dataset = VitalXDataset(X_val, y_val)
    test_dataset = VitalXDataset(X_test, y_test)
    
    # Create data loaders
    if use_balanced_sampling:
        # Use balanced sampler for training
        train_sampler = BalancedBatchSampler(y_train, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers
        )
    else:
        # Standard random sampling
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    
    # Validation and test loaders (no shuffling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Balanced sampling: {use_balanced_sampling}")
    print(f"  Data augmentation: {use_augmentation}")
    
    return train_loader, val_loader, test_loader


# ==============================
# TESTING
# ==============================
if __name__ == "__main__":
    print("Testing VitalX Dataset\n")
    
    # Create dummy data
    n_samples = 1000
    seq_len = 60
    n_features = 14
    
    X = np.random.randn(n_samples, seq_len, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Imbalance the data (20% positive)
    n_positive = int(0.2 * n_samples)
    y[:n_positive] = 1
    y[n_positive:] = 0
    np.random.shuffle(y)
    
    print("=" * 60)
    print("1. Testing VitalXDataset")
    print("=" * 60)
    dataset = VitalXDataset(X, y)
    
    # Test single sample
    seq, label = dataset[0]
    print(f"\nSample 0:")
    print(f"  Sequence shape: {seq.shape}")
    print(f"  Label: {label.item()}")
    
    # Test class weights
    weights = dataset.get_class_weights()
    print(f"\nClass weights: {weights}")
    
    # Test data splitting
    print("\n" + "=" * 60)
    print("2. Testing Data Splitting")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
    
    # Test DataLoaders
    print("\n" + "=" * 60)
    print("3. Testing DataLoaders (Standard)")
    print("=" * 60)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32,
        use_balanced_sampling=False
    )
    
    # Test batch
    for batch_X, batch_y in train_loader:
        print(f"\nFirst batch:")
        print(f"  X shape: {batch_X.shape}")
        print(f"  y shape: {batch_y.shape}")
        print(f"  Positive in batch: {torch.sum(batch_y).item():.0f}")
        break
    
    # Test balanced sampling
    print("\n" + "=" * 60)
    print("4. Testing Balanced Sampling")
    print("=" * 60)
    train_loader_balanced, _, _ = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32,
        use_balanced_sampling=True
    )
    
    for batch_X, batch_y in train_loader_balanced:
        print(f"\nFirst balanced batch:")
        print(f"  X shape: {batch_X.shape}")
        print(f"  y shape: {batch_y.shape}")
        print(f"  Positive in batch: {torch.sum(batch_y).item():.0f}")
        break
    
    print("\n✓ All dataset tests passed!")
