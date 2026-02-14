"""
VitalX LSTM Model with Attention Mechanism
===========================================

Architecture:
- LSTM layers for temporal pattern learning
- Attention mechanism for focusing on critical timesteps
- Fully connected layers for classification
- Sigmoid output for risk probability

Fallback: Logistic Regression for simple baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle


# ==============================
# ATTENTION MECHANISM
# ==============================
class Attention(nn.Module):
    """
    Attention layer to weight LSTM outputs.
    
    Learns which timesteps are most important for prediction.
    """
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
        # Attention weights
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_size)
        
        Returns:
            context: (batch_size, hidden_size) - weighted sum
            attention_weights: (batch_size, seq_len) - weights
        """
        # Calculate attention scores
        # (batch_size, seq_len, 1)
        attention_scores = self.attention(lstm_output)
        
        # Apply softmax to get weights
        # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum of LSTM outputs
        # (batch_size, hidden_size)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        
        # Return context and weights
        return context, attention_weights.squeeze(-1)


# ==============================
# LSTM MODEL WITH ATTENTION
# ==============================
class LSTMAttentionModel(nn.Module):
    """
    LSTM-based deterioration prediction model with attention.
    
    Architecture:
        Input: (batch_size, seq_len=60, input_size=14)
        └─> LSTM (2 layers, hidden_size=128, dropout=0.3)
        └─> Attention mechanism
        └─> FC1 (128 -> 64)
        └─> Dropout (0.3)
        └─> FC2 (64 -> 32)
        └─> Dropout (0.2)
        └─> Output (32 -> 1)
        └─> Sigmoid
    
    Output: (batch_size, 1) - risk probability [0, 1]
    """
    
    def __init__(self, input_size=14, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMAttentionModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = Attention(hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc_out = nn.Linear(32, 1)
        
        # Activation
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            output: (batch_size, 1) - risk probability
            attention_weights: (batch_size, seq_len) - attention weights
        """
        # LSTM forward
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        # context: (batch_size, hidden_size)
        context, attention_weights = self.attention(lstm_out)
        
        # Fully connected layers
        out = F.relu(self.fc1(context))
        out = self.dropout1(out)
        
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        
        out = self.fc_out(out)
        
        # Sigmoid for probability
        output = self.sigmoid(out)
        
        return output, attention_weights
    
    def predict_risk(self, x):
        """
        Predict risk without returning attention weights.
        
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            risk_scores: (batch_size, 1)
        """
        output, _ = self.forward(x)
        return output


# ==============================
# SIMPLE LSTM MODEL (No Attention)
# ==============================
class SimpleLSTMModel(nn.Module):
    """
    Simplified LSTM model without attention.
    Used as baseline or when attention is not needed.
    """
    
    def __init__(self, input_size=14, hidden_size=128, num_layers=2, dropout=0.3):
        super(SimpleLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            output: (batch_size, 1)
        """
        # Get last hidden state
        _, (h_n, _) = self.lstm(x)
        
        # Use last hidden state
        out = self.fc(h_n[-1])
        output = self.sigmoid(out)
        
        return output


# ==============================
# LOGISTIC REGRESSION FALLBACK
# ==============================
class LogisticRegressionFallback:
    """
    Logistic Regression fallback model.
    
    Used when:
    - LSTM model fails to load
    - Quick inference needed
    - Limited computational resources
    
    Flattens sequence data and uses traditional ML.
    """
    
    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Train logistic regression model.
        
        Args:
            X: (samples, seq_len, features) - will be flattened
            y: (samples,) - binary labels
        """
        # Flatten sequences
        X_flat = X.reshape(X.shape[0], -1)
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Train
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"Logistic Regression trained on {len(y)} samples")
        print(f"Input shape (flattened): {X_flat.shape}")
    
    def predict_proba(self, X):
        """
        Predict risk probabilities.
        
        Args:
            X: (samples, seq_len, features)
        
        Returns:
            proba: (samples,) - probability of deterioration
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Flatten
        X_flat = X.reshape(X.shape[0], -1)
        
        # Scale
        X_scaled = self.scaler.transform(X_flat)
        
        # Predict probability of class 1 (deterioration)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        
        return proba
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary labels.
        
        Args:
            X: (samples, seq_len, features)
            threshold: decision threshold
        
        Returns:
            predictions: (samples,) - binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def save(self, path):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Logistic Regression model saved to {path}")
    
    def load(self, path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        
        print(f"Logistic Regression model loaded from {path}")


# ==============================
# MODEL FACTORY
# ==============================
def create_model(model_type='lstm_attention', **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: 'lstm_attention', 'lstm_simple', or 'logistic'
        **kwargs: model-specific parameters
    
    Returns:
        model instance
    """
    if model_type == 'lstm_attention':
        return LSTMAttentionModel(**kwargs)
    elif model_type == 'lstm_simple':
        return SimpleLSTMModel(**kwargs)
    elif model_type == 'logistic':
        return LogisticRegressionFallback()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ==============================
# MODEL INFO
# ==============================
def count_parameters(model):
    """Count trainable parameters in PyTorch model."""
    if isinstance(model, (LSTMAttentionModel, SimpleLSTMModel)):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return None


def print_model_summary(model):
    """Print model summary."""
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    if isinstance(model, LSTMAttentionModel):
        print("Type: LSTM with Attention")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Input size: {model.input_size}")
        print(f"Hidden size: {model.hidden_size}")
        print(f"Num layers: {model.num_layers}")
    elif isinstance(model, SimpleLSTMModel):
        print("Type: Simple LSTM")
        print(f"Parameters: {count_parameters(model):,}")
    elif isinstance(model, LogisticRegressionFallback):
        print("Type: Logistic Regression (Fallback)")
        print(f"Fitted: {model.is_fitted}")
    
    print("=" * 60)


# ==============================
# TESTING
# ==============================
if __name__ == "__main__":
    print("Testing VitalX Models\n")
    
    # Test LSTM with Attention
    print("1. LSTM with Attention:")
    model = create_model('lstm_attention', input_size=14, hidden_size=128, num_layers=2)
    print_model_summary(model)
    
    # Test input
    batch_size = 4
    seq_len = 60
    input_size = 14
    
    x = torch.randn(batch_size, seq_len, input_size)
    output, attention = model(x)
    
    print(f"\nTest input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test Logistic Regression
    print("\n2. Logistic Regression Fallback:")
    lr_model = create_model('logistic')
    
    # Dummy training data
    X_train = np.random.randn(100, 60, 14)
    y_train = np.random.randint(0, 2, 100)
    
    lr_model.fit(X_train, y_train)
    
    # Test prediction
    X_test = np.random.randn(10, 60, 14)
    proba = lr_model.predict_proba(X_test)
    
    print(f"Test predictions: {proba[:5]}")
    
    print("\n✓ All models tested successfully!")
