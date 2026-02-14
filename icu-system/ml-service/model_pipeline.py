import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# -------------------------
# CONFIGURATION
# -------------------------
BATCH_SIZE = 64
INPUT_SIZE = 5
SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 256
NUM_LAYERS = 1
RANDOM_STATE = 42

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------
# 1. LOAD DATA
# -------------------------
print("\n=== Loading Data ===")
X = np.load("X_lstm_scaled.npy")
y = np.load("y_lstm.npy")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Positive cases: {y.sum()} ({100*y.sum()/len(y):.2f}%)")

# -------------------------
# 2. STRATIFIED SPLIT
# -------------------------
print("\n=== Splitting Data ===")

# First split: 70% train, 30% temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.30, 
    stratify=y, 
    random_state=RANDOM_STATE
)

# Second split: Split temp into 50-50 (15% val, 15% test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.50, 
    stratify=y_temp, 
    random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape[0]} samples ({100*len(X_train)/len(X):.1f}%)")
print(f"Val:   {X_val.shape[0]} samples ({100*len(X_val)/len(X):.1f}%)")
print(f"Test:  {X_test.shape[0]} samples ({100*len(X_test)/len(X):.1f}%)")

print(f"\nTrain positive rate: {100*y_train.sum()/len(y_train):.2f}%")
print(f"Val positive rate:   {100*y_val.sum()/len(y_val):.2f}%")
print(f"Test positive rate:  {100*y_test.sum()/len(y_test):.2f}%")

# -------------------------
# 3. CONVERT TO PYTORCH TENSORS
# -------------------------
print("\n=== Converting to PyTorch Tensors ===")

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

print(f"Train tensors: X={X_train_tensor.shape}, y={y_train_tensor.shape}")
print(f"Val tensors:   X={X_val_tensor.shape}, y={y_val_tensor.shape}")
print(f"Test tensors:  X={X_test_tensor.shape}, y={y_test_tensor.shape}")

# -------------------------
# 4. CREATE DATASETS AND DATALOADERS
# -------------------------
print("\n=== Creating DataLoaders ===")

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    drop_last=False
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    drop_last=False
)

print(f"Train loader: {len(train_loader)} batches")
print(f"Val loader:   {len(val_loader)} batches")
print(f"Test loader:  {len(test_loader)} batches")

# -------------------------
# 5. ATTENTION MECHANISM
# -------------------------
class AttentionLayer(nn.Module):
    """Attention mechanism over time dimension"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size)
            attention_weights: (batch_size, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, seq_len)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Compute weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden)
        
        return context, attention_weights


# -------------------------
# 6. LSTM + ATTENTION MODEL
# -------------------------
class LSTMAttentionModel(nn.Module):
    """LSTM with Attention for ICU deterioration prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMAttentionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.2  # No dropout with 1 layer
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_size)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, 1) - probability scores
            attention_weights: (batch_size, seq_len)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        # context: (batch, hidden_size)
        
        # Fully connected layer
        output = self.fc(context)  # (batch, 1)
        
        return output, attention_weights


# -------------------------
# 7. INITIALIZE MODEL
# -------------------------
print("\n=== Initializing Model ===")

model = LSTMAttentionModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
)

# Move model to device
model = model.to(device)

print(f"\nModel Architecture:")
print(model)

# -------------------------
# 8. MODEL SUMMARY
# -------------------------
print("\n=== Model Summary ===")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# -------------------------
# 9. TEST FORWARD PASS
# -------------------------
print("\n=== Testing Forward Pass ===")

# Get a sample batch
sample_batch_X, sample_batch_y = next(iter(train_loader))
sample_batch_X = sample_batch_X.to(device)
sample_batch_y = sample_batch_y.to(device)

print(f"Sample batch X shape: {sample_batch_X.shape}")
print(f"Sample batch y shape: {sample_batch_y.shape}")

# Forward pass
with torch.no_grad():
    predictions, attention_weights = model(sample_batch_X)

print(f"\nPredictions shape: {predictions.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"Sample predictions: {predictions[:5].squeeze().cpu().numpy()}")
print(f"Sample labels: {sample_batch_y[:5].cpu().numpy()}")

print("\n=== Pipeline Ready ===")
print("✓ Data loaded and split")
print("✓ Tensors and DataLoaders created")
print("✓ Model initialized and moved to device")
print("✓ Forward pass validated")

# -------------------------
# 10. LOSS FUNCTION AND OPTIMIZER
# -------------------------
print("\n=== Setting up Training ===")

LEARNING_RATE = 0.0005
NUM_EPOCHS = 30

# Loss function - BCEWithLogitsLoss (more numerically stable)
criterion = nn.BCEWithLogitsLoss()

# Optimizer - Adam
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Learning rate: {LEARNING_RATE}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Loss function: BCEWithLogitsLoss")
print(f"Optimizer: Adam")

# -------------------------
# 11. TRAINING LOOP WITH EARLY STOPPING
# -------------------------
print("\n=== Starting Training ===\n")

# Early stopping setup
best_val_loss = float("inf")
patience = 5
counter = 0

for epoch in range(NUM_EPOCHS):
    
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # Move data to device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, attention_weights = model(batch_X)
        outputs = outputs.squeeze()  # (batch_size,)
        
        # Compute loss
        loss = criterion(outputs, batch_y)
        
        # Backpropagation
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate loss
        train_loss += loss.item()
    
    # Average training loss
    avg_train_loss = train_loss / len(train_loader)
    
    # -------------------------
    # 12. VALIDATION LOOP
    # -------------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs, attention_weights = model(batch_X)
            outputs = outputs.squeeze()  # (batch_size,)
            
            # Compute loss
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            # Compute accuracy
            predictions = torch.sigmoid(outputs) > 0.5  # Apply sigmoid for predictions
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    # Average validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    # Print metrics
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_accuracy:.2f}%", end="")
    
    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "best_lstm_model.pth")
        print(" ✅ Model improved. Saved.")
    else:
        counter += 1
        print(f" No improvement. EarlyStopping counter: {counter}/{patience}")
        
        if counter >= patience:
            print("⛔ Early stopping triggered.")
            break

print("\n=== Training Complete ===\n")

# -------------------------
# 13. SAVE FINAL MODEL
# -------------------------
print("Saving final model...")
torch.save(model.state_dict(), "lstm_model.pth")
print("✓ Final model saved to lstm_model.pth")
print("✓ Best model saved to best_lstm_model.pth")

print("\n=== All Done! ===\n")
