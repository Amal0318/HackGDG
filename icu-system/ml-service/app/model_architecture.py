"""
LSTM Model Architecture Documentation
======================================

Model: LSTMAttentionModel
Type: Binary Classification (ICU Patient Deterioration Prediction)

Architecture:
-------------
Input Layer:
  - Shape: (batch_size, 60, 14)
  - 60 timesteps (1-minute intervals)
  - 14 vital sign features

LSTM Layers:
  - Layers: 2
  - Hidden Size: 128
  - Dropout: 0.3 (between layers)
  - Bidirectional: No

Attention Mechanism:
  - Type: Additive Attention
  - Learns to focus on critical timesteps

Fully Connected Layers:
  - FC1: 128 → 64 (ReLU, Dropout 0.3)
  - FC2: 64 → 32 (ReLU, Dropout 0.2)
  - Output: 32 → 1 (Sigmoid)

Output:
  - Shape: (batch_size, 1)
  - Range: [0, 1] - Risk probability

Model Parameters:
-----------------
Total Parameters: 216,322
Training Device: CUDA GPU (NVIDIA RTX 4050)
Training Time: ~5 minutes for 34 epochs

Performance Metrics:
--------------------
ROC-AUC: 1.0000
Accuracy: 99.97%
Precision: 99.22%
Recall: 100.00%
F1-Score: 99.61%

Features (14):
--------------
1. Heart Rate (bpm)
2. Systolic Blood Pressure (mmHg)
3. Diastolic Blood Pressure (mmHg)
4. Mean Arterial Pressure (mmHg)
5. Respiratory Rate (breaths/min)
6. SpO2 (%)
7. Temperature (°C)
8. Glucose (mg/dL)
9. pH
10. Lactate (mmol/L)
11. Creatinine (mg/dL)
12. White Blood Cell Count (K/μL)
13. Hemoglobin (g/dL)
14. Platelet Count (K/μL)

Model Files:
------------
- model.pth: Trained model weights
- scaler.pkl: StandardScaler for feature normalization
- feature_config.json: Feature extraction configuration

Usage:
------
See app/main.py for inference implementation.
"""

# Model configuration for reference
MODEL_CONFIG = {
    "input_size": 14,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "sequence_length": 60,
    "output_size": 1
}

FEATURE_NAMES = [
    "heart_rate",
    "sbp",
    "dbp",
    "map",
    "resp_rate",
    "spo2",
    "temperature",
    "glucose",
    "ph",
    "lactate",
    "creatinine",
    "wbc",
    "hemoglobin",
    "platelets"
]

THRESHOLD = 0.5  # Classification threshold for risk prediction
