"""
Quick Reference Guide for VitalX Sepsis Prediction

FEATURE COUNT: 33 features total
- Core Vitals: 8
- Key Labs: 5
- Demographics: 2
- Derived Features: 6
- Missingness Indicators: 13
"""

# ============================================================
# QUICK START
# ============================================================

# 1. Install dependencies
# pip install -r requirements.txt

# 2. Run complete pipeline
# python main.py

# 3. Run with small subset (for testing)
# python main.py --subset

# ============================================================
# INDIVIDUAL SCRIPTS
# ============================================================

# View feature configuration
# python config.py

# Train model only
# python train.py

# Evaluate model
# python evaluate.py

# Run inference
# python inference.py

# ============================================================
# PROGRAMMATIC USAGE
# ============================================================

"""
# Example 1: Load and preprocess data
from dataset import SepsisDataset

dataset = SepsisDataset(
    data_dirs=['training/training_setA'],
    sequence_length=24,
    mode='train'
)

# Example 2: Create model
from model import create_model
import config

model = create_model(
    input_size=config.get_feature_count(),
    device='cpu'
)

# Example 3: Make predictions
from inference import SepsisPredictor
import numpy as np

predictor = SepsisPredictor()
sequence = np.random.randn(24, 33)
probability = predictor.predict(sequence)
print(f"Sepsis probability: {probability:.4f}")

# Example 4: Batch predictions
sequences = np.random.randn(10, 24, 33)  # 10 patients
probabilities = predictor.predict_batch(sequences)

# Example 5: Predict from patient file
probability, sequence = predictor.predict_from_patient_file(
    'training/training_setA/p000001.psv'
)
"""

# ============================================================
# KEY PARAMETERS (config.py)
# ============================================================

PARAMETERS = {
    'Sequence Length': 24,  # hours
    'Hidden Size': 128,
    'Num LSTM Layers': 2,
    'Dropout': 0.3,
    'Batch Size': 32,
    'Learning Rate': 0.001,
    'Max Epochs': 50,
    'Early Stopping Patience': 7,
    'Use Attention': True
}

# ============================================================
# DATA FORMAT
# ============================================================

"""
PhysioNet 2019 PSV Format:
- Pipe-separated values (|)
- One file per patient
- Each row = one hour of ICU stay
- Last column = SepsisLabel (0 or 1)
- Many missing values (NaN)

Example:
HR|O2Sat|Temp|SBP|...|SepsisLabel
97|95|NaN|98|...|0
89|99|NaN|122|...|0
"""

# ============================================================
# MODEL OUTPUTS
# ============================================================

"""
outputs/
├── model.pth                   # Trained model weights
├── scaler.pkl                  # Feature normalization scaler
├── feature_config.json         # Feature list and config
├── training_history.json       # Loss and metrics per epoch
├── test_evaluation.json        # Test set performance
├── plots/
│   ├── roc_curve.png          # ROC curve
│   ├── pr_curve.png           # Precision-Recall curve
│   └── threshold_analysis.png  # Threshold sensitivity
└── checkpoints/                # Model checkpoints
"""

# ============================================================
# EVALUATION METRICS
# ============================================================

"""
Primary Metrics:
- AUROC: Area Under ROC Curve
- AUPRC: Area Under Precision-Recall Curve

Secondary Metrics:
- Sensitivity (Recall): TP / (TP + FN)
- Specificity: TN / (TN + FP)
- Precision: TP / (TP + FP)
- F1 Score: 2 * Precision * Recall / (Precision + Recall)
- Utility Score: PhysioNet 2019 challenge metric

Confusion Matrix:
    Predicted
         0    1
Actual 0  TN   FP
       1  FN   TP
"""

# ============================================================
# TROUBLESHOOTING
# ============================================================

"""
Issue: "No data directories found"
Fix: Ensure training/training_setA or training/training_setB exists

Issue: "Model not found"
Fix: Run python train.py first

Issue: Out of memory
Fix: Reduce BATCH_SIZE in config.py

Issue: Poor performance
Fix: 
- Check class imbalance weights
- Increase training data
- Tune hyperparameters
- Try different model architectures
"""

# ============================================================
# FEATURE ENGINEERING DETAILS
# ============================================================

"""
1. Missing Value Handling:
   - Forward-fill per patient
   - Backward-fill for initial NaNs
   - Fill remaining with 0

2. Missingness Indicators:
   - Created BEFORE forward-fill
   - Binary mask (1 = missing, 0 = present)
   - Critical for clinical interpretation

3. Derived Features:
   - ShockIndex = HR / SBP (cardiovascular stress)
   - HR_delta = diff(HR) (trend detection)
   - SBP_delta = diff(SBP)
   - ShockIndex_delta = diff(ShockIndex)
   - RollingMean_HR = 6-hour rolling mean
   - RollingMean_SBP = 6-hour rolling mean

4. Normalization:
   - StandardScaler (zero mean, unit variance)
   - Fit on training data only
   - Applied to all splits
"""

# ============================================================
# MODEL ARCHITECTURE DETAILS
# ============================================================

"""
Input: (batch_size, 24, 33)
   ↓
LSTM Layer 1 (input_size=33, hidden_size=128)
   ↓
Dropout (p=0.3)
   ↓
LSTM Layer 2 (input_size=128, hidden_size=128)
   ↓
Dropout (p=0.3)
   ↓
Attention Layer (learns temporal importance)
   ↓
Fully Connected (128 → 64)
   ↓
ReLU + Dropout
   ↓
Fully Connected (64 → 1)
   ↓
Sigmoid
   ↓
Output: (batch_size, 1) [probability]

Total Parameters: ~300K
"""

print(__doc__)
