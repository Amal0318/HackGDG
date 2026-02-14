import torch
import numpy as np
import torch.nn as nn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Define model architecture (must match training) ----
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        scores = self.attention(lstm_output).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        out = self.fc(context)
        return out


# ---- Load Model ----
model = LSTMAttentionModel().to(device)
model.load_state_dict(torch.load("best_lstm_model.pth"))
model.eval()

print("Model loaded successfully!")

# ---- Example Input (Replace with real data) ----
# Shape must be (1, 20, 5)
sample_input = np.random.randn(1, 20, 5)

sample_tensor = torch.tensor(sample_input, dtype=torch.float32).to(device)

# ---- Prediction ----
with torch.no_grad():
    output = model(sample_tensor)
    probability = torch.sigmoid(output)

risk_score = probability.item()

print(f"Predicted Risk: {risk_score:.4f}")

if risk_score > 0.5:
    print("⚠ High Risk Detected")
else:
    print("✓ Stable")
