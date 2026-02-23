"""
LSTM Model with Attention for Sepsis Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence modeling
    Computes weighted sum of LSTM outputs based on learned attention weights
    """
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        
        # Attention weights
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, sequence_length, hidden_size)
        
        Returns:
            context: (batch_size, hidden_size)
            attention_weights: (batch_size, sequence_length)
        """
        # Compute attention scores
        # (batch_size, sequence_length, 1)
        attention_scores = self.attention(lstm_output)
        
        # Apply softmax to get attention weights
        # (batch_size, sequence_length, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute weighted sum (context vector)
        # (batch_size, hidden_size)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context, attention_weights.squeeze(-1)


class SepsisLSTM(nn.Module):
    """
    LSTM-based model for sepsis prediction
    
    Architecture:
        Input → LSTM (2 layers) → Attention → FC → Sigmoid → Probability
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT,
                 bidirectional: bool = config.BIDIRECTIONAL,
                 use_attention: bool = config.USE_ATTENTION):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
        """
        super(SepsisLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Adjust hidden size for bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(lstm_output_size)
            fc_input_size = lstm_output_size
        else:
            self.attention = None
            fc_input_size = lstm_output_size
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            output: (batch_size, 1) - probability of sepsis
        """
        # LSTM
        # lstm_out: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention or use last time step
        if self.use_attention:
            # Apply attention
            context, attention_weights = self.attention(lstm_out)
            features = context
        else:
            # Use last time step
            features = lstm_out[:, -1, :]
        
        # Dropout
        features = self.dropout_layer(features)
        
        # Fully connected layers
        out = self.fc1(features)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        # Sigmoid activation for probability
        prob = self.sigmoid(out)
        
        return prob
    
    def predict(self, x):
        """
        Prediction mode (no gradient computation)
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            probabilities: (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            prob = self.forward(x)
        return prob


def create_model(input_size: int, device: str = 'cpu'):
    """
    Factory function to create model
    
    Args:
        input_size: Number of input features
        device: 'cpu' or 'cuda'
    
    Returns:
        model: SepsisLSTM model
    """
    model = SepsisLSTM(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
        use_attention=config.USE_ATTENTION
    )
    
    model = model.to(device)
    
    # Print model summary
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    print("="*60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60 + "\n")
    
    return model


def save_model(model, filepath):
    """Save model state dict"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'dropout': model.dropout,
            'bidirectional': model.bidirectional,
            'use_attention': model.use_attention
        }
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device='cpu'):
    """Load model from checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model_config = checkpoint['model_config']
    model = SepsisLSTM(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model...")
    
    # Create dummy input
    batch_size = 4
    sequence_length = 24
    input_size = config.get_feature_count()
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = create_model(input_size, device)
    
    # Create dummy data
    x = torch.randn(batch_size, sequence_length, input_size).to(device)
    
    # Forward pass
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (probabilities): {output.squeeze().detach().cpu().numpy()}")
    
    print("\nModel test successful!")
