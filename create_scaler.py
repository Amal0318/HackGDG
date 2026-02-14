import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

print("Creating feature scaler from training data...")

# Load the training data
X = np.load("X_lstm_scaled.npy")
print(f"Data shape: {X.shape}")

# The data is already scaled, but we need to create the scaler object
# We'll fit it on the original unscaled data
X_unscaled = np.load("X_lstm.npy")

# Reshape for scaling: (samples, timesteps, features) -> (samples*timesteps, features)
n_samples, n_timesteps, n_features = X_unscaled.shape
X_reshaped = X_unscaled.reshape(-1, n_features)

# Fit scaler
scaler = StandardScaler()
scaler.fit(X_reshaped)

# Save scaler
with open("feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"✓ Scaler saved to feature_scaler.pkl")
print(f"✓ Mean values: {scaler.mean_}")
print(f"✓ Std values: {scaler.scale_}")
