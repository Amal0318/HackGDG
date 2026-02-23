# VitalX - Sepsis Prediction Model

Clinical time-series ML model for predicting sepsis 6 hours before clinical onset.  
Aligned with **PhysioNet 2019 Sepsis Challenge**.

---

## 📋 Overview

**Objective:** Predict probability of sepsis at every hour, with 6-hour early warning window.

**Key Features:**
- ✅ No future data leakage
- ✅ Hour-by-hour predictions
- ✅ LSTM with attention mechanism
- ✅ Comprehensive feature engineering
- ✅ Class imbalance handling
- ✅ Clean modular architecture

---

## 🏗️ Project Structure

```
VitalX ml/
│
├── config.py              # Configuration and feature definitions
├── dataset.py             # Dataset class with preprocessing
├── model.py               # LSTM model with attention
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Inference wrapper
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
│
├── training/
│   ├── training_setA/     # Patient data files (PSV format)
│   └── training_setB/     # Additional patient data
│
└── outputs/               # Generated during training
    ├── model.pth          # Trained model
    ├── scaler.pkl         # Feature scaler
    ├── feature_config.json # Feature configuration
    ├── training_history.json
    ├── evaluation_report.json
    └── plots/             # Evaluation visualizations
```

---

## 📊 Features

### Total: **33 Features**

**Core Vitals (8):**
- HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2

**Key Labs (5):**
- Lactate, WBC, Creatinine, Platelets, Bilirubin_total

**Demographics (2):**
- Age, Gender

**Derived Features (6):**
- ShockIndex (HR/SBP)
- HR_delta, SBP_delta, ShockIndex_delta
- RollingMean_HR, RollingMean_SBP

**Missingness Indicators (13):**
- Binary masks for missing vitals and labs

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Paths

Check [`config.py`](config.py) and update paths if needed:

```python
DATA_DIR_A = 'training/training_setA'
DATA_DIR_B = 'training/training_setB'
OUTPUT_DIR = 'outputs'
```

### 3. View Feature Configuration

```bash
python config.py
```

### 4. Train Model

```bash
python train.py
```

**Training includes:**
- Automatic train/val split
- Forward-filling missing values
- Missingness mask creation
- Feature derivation
- Normalization
- Sliding window creation (24-hour sequences)
- Class weight computation
- Early stopping
- Model checkpointing

### 5. Evaluate Model

```bash
python evaluate.py
```

**Outputs:**
- AUROC, AUPRC, Sensitivity, Specificity
- ROC curve
- Precision-Recall curve
- Threshold analysis
- Confusion matrix
- Utility score

### 6. Run Inference

```bash
python inference.py
```

---

## 🔧 Model Architecture

```
Input (24 hours × 33 features)
    ↓
LSTM (2 layers, hidden=128, dropout=0.3)
    ↓
Attention Layer
    ↓
Fully Connected (64 units)
    ↓
ReLU + Dropout
    ↓
Output Layer (1 unit)
    ↓
Sigmoid → Probability [0, 1]
```

**Parameters:**
- Hidden size: 128
- Num layers: 2
- Dropout: 0.3
- Attention: ✅ Enabled
- Bidirectional: ❌ Disabled

---

## 📈 Training Configuration

| Parameter | Value |
|-----------|-------|
| Sequence length | 24 hours |
| Batch size | 32 |
| Learning rate | 0.001 |
| Max epochs | 50 |
| Early stopping patience | 7 |
| Optimizer | Adam |
| Loss function | Weighted BCE |
| LR scheduler | ReduceLROnPlateau |

---

## 🎯 Usage Examples

### Example 1: Using SepsisPredictor Class

```python
from inference import SepsisPredictor
import numpy as np

# Initialize predictor
predictor = SepsisPredictor()

# Create sequence (24 hours × 33 features)
sequence = np.random.randn(24, 33)

# Predict
probability = predictor.predict(sequence)
print(f"Sepsis probability: {probability:.4f}")
```

### Example 2: Predict from Patient File

```python
from inference import SepsisPredictor

predictor = SepsisPredictor()

# Predict from PSV file
probability, sequence = predictor.predict_from_patient_file(
    'training/training_setA/p000001.psv'
)

print(f"Sepsis probability: {probability:.4f}")
```

### Example 3: Batch Prediction

```python
from inference import SepsisPredictor
import numpy as np

predictor = SepsisPredictor()

# Multiple sequences
sequences = np.random.randn(10, 24, 33)  # 10 patients

# Batch predict
probabilities = predictor.predict_batch(sequences)
print(probabilities)
```

### Example 4: Standalone Prediction Function

```python
import torch
from inference import predict

# Preprocessed tensor
sequence_tensor = torch.randn(1, 24, 33)

# Predict
probability = predict(sequence_tensor)
print(f"Sepsis probability: {probability:.4f}")
```

---

## 📁 Outputs

After training, the following files are saved:

### Model Artifacts
- **`outputs/model.pth`** - Trained PyTorch model
- **`outputs/scaler.pkl`** - Fitted StandardScaler
- **`outputs/feature_config.json`** - Feature configuration

### Training Results
- **`outputs/training_history.json`** - Loss and metrics per epoch
- **`outputs/checkpoints/`** - Model checkpoints

### Evaluation Results
- **`outputs/evaluation_report.json`** - Test metrics
- **`outputs/plots/roc_curve.png`** - ROC curve
- **`outputs/plots/pr_curve.png`** - Precision-Recall curve
- **`outputs/plots/threshold_analysis.png`** - Threshold sensitivity

---

## 🔄 Data Preprocessing Pipeline

1. **Load PSV files** (pipe-separated format)
2. **Create missingness masks** (before forward-fill)
3. **Forward-fill missing values** per patient
4. **Derive additional features** (shock index, deltas, rolling means)
5. **Normalize features** (StandardScaler fit on training data)
6. **Create sliding windows** (24-hour sequences with 1-hour stride)
7. **Handle class imbalance** (compute positive class weights)

---

## 📊 Evaluation Metrics

- **AUROC** - Area Under ROC Curve
- **AUPRC** - Area Under Precision-Recall Curve
- **Sensitivity** (Recall) - True Positive Rate
- **Specificity** - True Negative Rate
- **Precision** - Positive Predictive Value
- **F1 Score** - Harmonic mean of precision and recall
- **Utility Score** - PhysioNet 2019 challenge metric

---

## 🛠️ Troubleshooting

### Issue: "No data directories found"
**Solution:** Ensure `training/training_setA` and/or `training/training_setB` exist with PSV files.

### Issue: "Model not found"
**Solution:** Train the model first using `python train.py`

### Issue: Out of memory during training
**Solution:** Reduce `BATCH_SIZE` in [`config.py`](config.py)

### Issue: Poor performance
**Solutions:**
- Increase training data
- Adjust class weights
- Tune hyperparameters in [`config.py`](config.py)
- Try different architectures (increase `HIDDEN_SIZE` or `NUM_LAYERS`)

---

## 🔬 Technical Details

### Loss Function
**Weighted Binary Cross Entropy** - Handles class imbalance by weighting positive samples.

```python
pos_weight = n_negative / n_positive
criterion = nn.BCELoss()  # Applied after sigmoid
```

### Attention Mechanism
Learns to focus on important time steps in the 24-hour window.

```python
attention_scores = Linear(lstm_output)
attention_weights = Softmax(attention_scores)
context = Sum(attention_weights * lstm_output)
```

### Feature Derivation
- **ShockIndex**: HR / SBP (cardiovascular stress indicator)
- **Delta features**: Hour-to-hour changes capture trends
- **Rolling means**: 6-hour windows smooth vital signs

---

## 📝 Notes

- **Data Format**: PhysioNet 2019 PSV (pipe-separated values)
- **Time Resolution**: Hourly measurements
- **Prediction Window**: 24-hour lookback
- **Target**: SepsisLabel at current hour
- **No API/Kafka**: Model-only implementation as requested

---

## 🎓 References

- PhysioNet 2019 Sepsis Challenge
- Clinical early warning systems
- LSTM for time-series classification
- Attention mechanisms in sequential models

---

## 📄 License

This is a clinical ML research project. Ensure proper validation before clinical deployment.

---

## ✅ Checklist

- [x] Data preprocessing with forward-fill
- [x] Missingness mask features
- [x] Feature engineering (derived + rolling)
- [x] LSTM model with attention
- [x] Weighted BCE for class imbalance
- [x] Early stopping & checkpointing
- [x] AUROC & AUPRC metrics
- [x] Model export (pth, pkl, json)
- [x] Clean inference wrapper
- [x] Modular structure

---

**Built with PyTorch • Designed for Clinical ML • VitalX 2026**
