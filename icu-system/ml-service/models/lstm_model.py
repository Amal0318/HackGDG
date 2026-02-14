"""
LSTM Model Implementation for ICU Deterioration Prediction
"""

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Attention mechanism wrapper"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_out):
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return context


class LSTMAttentionModel(nn.Module):
    """
    LSTM with Attention for ICU Patient Deterioration Prediction
    
    Architecture:
    - Input: (batch_size, 60, 14) - 60 timesteps x 14 features
    - LSTM: 2 layers, hidden_size=128
    - Attention mechanism
    - FC layers: 128 -> 64 -> 32 -> 1
    - Output: (batch_size, 1) - Risk probability [0, 1]
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
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism (nested structure to match saved model)
        self.attention = AttentionLayer(hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)  # Changed from fc3 to fc_out
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        
        # Attention mechanism
        context = self.attention(lstm_out)
        # context shape: (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out = self.fc_out(out)  # Changed from fc3 to fc_out
        out = self.sigmoid(out)
        
        return out


class LogisticRegressionFallback:
    """Simple fallback model for when LSTM is not available"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def predict(self, X):
        """Predict using logistic regression"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        
        # Take only the last timestep for logistic regression
        if len(X.shape) == 3:
            X = X[:, -1, :]
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)[:, 1]
