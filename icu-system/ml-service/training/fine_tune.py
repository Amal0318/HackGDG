"""
Fine-tuning script for ICU LSTM model
Allows hyperparameter tuning, transfer learning, and model optimization
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple

from models.lstm_model import LSTMAttentionModel
from utils.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FineTuneConfig:
    """Configuration for fine-tuning with lower learning rate and enhanced options"""
    
    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        batch_size: int = 32,
        num_epochs: int = 30,
        learning_rate: float = 0.0001,  # 10x lower for fine-tuning
        weight_decay: float = 0.0001,
        patience: int = 8,
        freeze_layers: bool = False,  # Option to freeze early layers
        freeze_n_layers: int = 1,  # Number of LSTM layers to freeze
        use_augmentation: bool = True,
        pretrained_path: Optional[str] = None
    ):
        # Check GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for fine-tuning!")
        self.device = torch.device('cuda')
        
        # Model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Training parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        
        # Fine-tuning specific
        self.freeze_layers = freeze_layers
        self.freeze_n_layers = freeze_n_layers
        self.use_augmentation = use_augmentation
        self.pretrained_path = pretrained_path or "saved_models/best_model.pth"
        
        # Paths
        self.data_path = "data"
        self.save_dir = "fine_tuned_models"
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info(f"Fine-tuning Configuration:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model: {num_layers}x{hidden_size}, dropout={dropout}")
        logger.info(f"  Learning Rate: {learning_rate} (fine-tuning)")
        logger.info(f"  Freeze Layers: {freeze_layers} ({freeze_n_layers} layers)")
        logger.info(f"  Augmentation: {use_augmentation}")
        logger.info(f"  Pretrained: {pretrained_path}")


class FineTuneTrainer:
    """Trainer for fine-tuning pretrained LSTM models"""
    
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.device = config.device
        self.best_auc = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': []
        }
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load training data"""
        logger.info("Loading data...")
        
        X = np.load(os.path.join(self.config.data_path, 'X.npy'))
        y = np.load(os.path.join(self.config.data_path, 'y.npy'))
        
        # Split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Data augmentation for fine-tuning
        if self.config.use_augmentation:
            X_train, y_train = self._augment_data(X_train, y_train)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)  # (samples, 1)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)  # (samples, 1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        # Balanced sampling for positive class
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        sample_weights = torch.where(
            y_train_t.squeeze() == 1,  # Squeeze to (samples,) for comparison
            torch.tensor(pos_weight, device=self.device),
            torch.tensor(1.0, device=self.device)
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, sampler=sampler
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
        logger.info(f"Augmentation: {self.config.use_augmentation}")
        
        return train_loader, val_loader
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple data augmentation: add Gaussian noise to positive samples"""
        augmented_X = []
        augmented_y = []
        for i, label in enumerate(y):
            augmented_X.append(X[i])
            augmented_y.append(label)
            # Augment only positive class samples
            if label == 1:
                noise = np.random.normal(0, 0.01, X[i].shape)
                augmented_X.append(X[i] + noise)
                augmented_y.append(label)
        
        logger.info(f"Augmented from {len(X)} to {len(augmented_X)} samples")
        return np.array(augmented_X), np.array(augmented_y)
    
    def load_pretrained_model(self) -> LSTMAttentionModel:
        """Load pretrained model for fine-tuning"""
        logger.info(f"Loading pretrained model: {self.config.pretrained_path}")
        
        # Create model
        model = LSTMAttentionModel(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Load pretrained weights
        checkpoint = torch.load(self.config.pretrained_path, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_auc' in checkpoint:
                logger.info(f"Pretrained AUC: {checkpoint['val_auc']:.4f}")
            else:
                logger.info("Pretrained AUC: unknown")
        else:
            model.load_state_dict(checkpoint)
        
        # Freeze layers if requested
        if self.config.freeze_layers:
            self._freeze_lstm_layers(model)
        
        return model
    
    def _freeze_lstm_layers(self, model: LSTMAttentionModel):
        """Freeze early LSTM layers for transfer learning"""
        logger.info(f"Freezing first {self.config.freeze_n_layers} LSTM layer(s)")
        
        # LSTM parameters are stored in lstm.weight_ih_l{layer}, lstm.weight_hh_l{layer}
        for name, param in model.named_parameters():
            if 'lstm' in name:
                # Extract layer number
                if '_l0' in name or (self.config.freeze_n_layers > 1 and '_l1' in name):
                    param.requires_grad = False
                    logger.info(f"  Frozen: {name}")
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs, _ = model(batch_X)  # Unpack (output, attention_weights)
            loss = criterion(outputs, batch_y)  # Both are (batch_size, 1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())  # Already sigmoid probabilities
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        
        # Convert probabilities to binary predictions
        y_proba = np.array(all_preds).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        y_true = np.array(all_labels).flatten()
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        return avg_loss, metrics['roc_auc']
    
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float, Dict]:
        """Validate the model"""
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs, _ = model(batch_X)  # Unpack (output, attention_weights)
                loss = criterion(outputs, batch_y)  # Both are (batch_size, 1)
                
                total_loss += loss.item()
                all_preds.extend(outputs.detach().cpu().numpy())  # Already sigmoid probabilities
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Convert probabilities to binary predictions
        y_proba = np.array(all_preds).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        y_true = np.array(all_labels).flatten()
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        return avg_loss, metrics['roc_auc'], metrics
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict:
        """Full fine-tuning loop"""
        logger.info("Starting fine-tuning...")
        
        # Loss and optimizer
        criterion = nn.BCELoss()  # Model already applies sigmoid
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_auc = self.train_epoch(
                model, train_loader, criterion, optimizer
            )
            
            # Validate
            val_loss, val_auc, val_metrics = self.validate(
                model, val_loader, criterion
            )
            
            # Update scheduler
            scheduler.step(val_auc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
                f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                self.save_model(model, optimizer, epoch, val_auc, val_metrics)
                logger.info(f"  ✓ New best model! AUC: {val_auc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            'best_auc': self.best_auc,
            'history': self.history,
            'final_metrics': val_metrics
        }
    
    def save_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_auc: float,
        metrics: Dict
    ):
        """Save fine-tuned model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fine_tuned_model_auc{val_auc:.4f}_{timestamp}.pth"
        filepath = os.path.join(self.config.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
            'metrics': metrics,
            'config': {
                'input_size': self.config.input_size,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout,
                'learning_rate': self.config.learning_rate,
                'freeze_layers': self.config.freeze_layers,
                'use_augmentation': self.config.use_augmentation
            },
            'history': self.history,
            'timestamp': timestamp
        }
        
        torch.save(checkpoint, filepath)
        
        # Also save as 'best_fine_tuned.pth'
        best_path = os.path.join(self.config.save_dir, "best_fine_tuned.pth")
        torch.save(checkpoint, best_path)
        
        logger.info(f"Saved fine-tuned model: {filepath}")


def hyperparameter_search(base_config: FineTuneConfig):
    """Run hyperparameter search for fine-tuning"""
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER SEARCH FOR FINE-TUNING")
    logger.info("=" * 60)
    
    # Define search space
    search_configs = [
        # Original architecture variations
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'name': 'Original'},
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.5, 'name': 'Higher Dropout'},
        {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3, 'name': 'Wider Network'},
        {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.3, 'name': 'Deeper Network'},
        {'hidden_size': 256, 'num_layers': 3, 'dropout': 0.4, 'name': 'Wider + Deeper'},
        
        # With layer freezing
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'freeze_layers': True, 'name': 'Frozen Layer 1'},
    ]
    
    results = []
    
    for i, params in enumerate(search_configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuration {i+1}/{len(search_configs)}: {params['name']}")
        logger.info(f"{'='*60}")
        
        # Create config
        config = FineTuneConfig(
            hidden_size=params.get('hidden_size', base_config.hidden_size),
            num_layers=params.get('num_layers', base_config.num_layers),
            dropout=params.get('dropout', base_config.dropout),
            freeze_layers=params.get('freeze_layers', False),
            freeze_n_layers=params.get('freeze_n_layers', 1),
            learning_rate=base_config.learning_rate,
            batch_size=base_config.batch_size,
            num_epochs=20,  # Shorter for search
            patience=6,
            use_augmentation=base_config.use_augmentation,
            pretrained_path=base_config.pretrained_path
        )
        
        # Train
        trainer = FineTuneTrainer(config)
        train_loader, val_loader = trainer.load_data()
        model = trainer.load_pretrained_model()
        result = trainer.train(model, train_loader, val_loader)
        
        # Store results
        results.append({
            'name': params['name'],
            'params': params,
            'best_auc': result['best_auc'],
            'final_metrics': result['final_metrics']
        })
        
        logger.info(f"Result: AUC = {result['best_auc']:.4f}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("HYPERPARAMETER SEARCH RESULTS")
    logger.info("="*60)
    
    results.sort(key=lambda x: x['best_auc'], reverse=True)
    
    for i, result in enumerate(results):
        logger.info(
            f"{i+1}. {result['name']:<20} | "
            f"AUC: {result['best_auc']:.4f} | "
            f"Params: {result['params']}"
        )
    
    logger.info(f"\nBest configuration: {results[0]['name']}")
    logger.info(f"Best AUC: {results[0]['best_auc']:.4f}")
    
    # Save results
    results_file = os.path.join(base_config.save_dir, "hyperparameter_search_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_file}")
    
    return results


def main():
    """Main fine-tuning entry point"""
    print("="*60)
    print("ICU LSTM MODEL FINE-TUNING")
    print("="*60)
    print("\nOptions:")
    print("1. Fine-tune with default configuration")
    print("2. Fine-tune with custom configuration")
    print("3. Run hyperparameter search")
    print("4. Fine-tune with layer freezing")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        # Default fine-tuning
        config = FineTuneConfig(
            learning_rate=0.0001,
            num_epochs=30,
            patience=8,
            use_augmentation=True
        )
        
        trainer = FineTuneTrainer(config)
        train_loader, val_loader = trainer.load_data()
        model = trainer.load_pretrained_model()
        result = trainer.train(model, train_loader, val_loader)
        
        logger.info("\n" + "="*60)
        logger.info("FINE-TUNING COMPLETE")
        logger.info("="*60)
        logger.info(f"Best AUC: {result['best_auc']:.4f}")
        logger.info(f"Final Metrics: {result['final_metrics']}")
    
    elif choice == '2':
        # Custom configuration
        print("\nCustom Configuration:")
        hidden_size = int(input("Hidden size (default 128): ") or "128")
        num_layers = int(input("Number of layers (default 2): ") or "2")
        dropout = float(input("Dropout (default 0.3): ") or "0.3")
        lr = float(input("Learning rate (default 0.0001): ") or "0.0001")
        
        config = FineTuneConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=lr,
            use_augmentation=True
        )
        
        trainer = FineTuneTrainer(config)
        train_loader, val_loader = trainer.load_data()
        model = trainer.load_pretrained_model()
        result = trainer.train(model, train_loader, val_loader)
        
        logger.info("\n" + "="*60)
        logger.info("FINE-TUNING COMPLETE")
        logger.info("="*60)
        logger.info(f"Best AUC: {result['best_auc']:.4f}")
    
    elif choice == '3':
        # Hyperparameter search
        base_config = FineTuneConfig(
            learning_rate=0.0001,
            use_augmentation=True
        )
        hyperparameter_search(base_config)
    
    elif choice == '4':
        # Transfer learning with frozen layers
        n_freeze = int(input("Number of LSTM layers to freeze (1 or 2): ") or "1")
        
        config = FineTuneConfig(
            learning_rate=0.0001,
            num_epochs=30,
            patience=8,
            freeze_layers=True,
            freeze_n_layers=n_freeze,
            use_augmentation=True
        )
        
        trainer = FineTuneTrainer(config)
        train_loader, val_loader = trainer.load_data()
        model = trainer.load_pretrained_model()
        result = trainer.train(model, train_loader, val_loader)
        
        logger.info("\n" + "="*60)
        logger.info("FINE-TUNING COMPLETE (with frozen layers)")
        logger.info("="*60)
        logger.info(f"Best AUC: {result['best_auc']:.4f}")
    
    else:
        logger.error("Invalid choice!")


if __name__ == "__main__":
    main()
