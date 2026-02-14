import numpy as np
from sklearn.preprocessing import StandardScaler

print("Loading data...")
X = np.load("X_lstm.npy")

print(f"Original X shape: {X.shape}")
print(f"Original X range: [{X.min():.2f}, {X.max():.2f}]")

# Reshape for scaling: (samples * timesteps, features)
n_samples, n_timesteps, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)

print(f"\nScaling features...")
scaler = StandardScaler()
X_scaled_reshaped = scaler.fit_transform(X_reshaped)

# Reshape back to original: (samples, timesteps, features)
X_scaled = X_scaled_reshaped.reshape(n_samples, n_timesteps, n_features)

print(f"Scaled X shape: {X_scaled.shape}")
print(f"Scaled X range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

# Save scaled data
np.save("X_lstm_scaled.npy", X_scaled)
print("\n✓ Saved X_lstm_scaled.npy")
