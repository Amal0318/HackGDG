# VitalX ML Service

LSTM-based time-series deterioration prediction model for ICU Digital Twin system.

## 📋 Overview

This service trains and deploys an LSTM model with attention mechanism to predict patient deterioration 3-5 minutes in advance using real-time vital sign telemetry.

**Model Type**: Binary Classification (Time-Series)  
**Input**: 60-second sliding window of 14 vital features  
**Output**: Risk probability [0, 1]  
**Architecture**: LSTM + Attention + Logistic Regression Fallback

---

## 🏗️ Project Structure

```
ml-service/
├── data/                          # Training data (generated)
│   ├── X.npy                      # Sequences (samples, 60, 14)
│   └── y.npy                      # Labels (samples,)
│
├── models/                        # Model architectures
│   ├── lstm_model.py             # LSTM with Attention
│   └── __init__.py
│
├── training/                      # Training pipeline
│   ├── generate_dataset.py       # Dataset generation from JSONL
│   ├── dataset.py                # PyTorch Dataset classes
│   ├── train.py                  # Main training script
│   └── __init__.py
│
├── utils/                         # Utilities
│   ├── metrics.py                # Evaluation metrics
│   └── __init__.py
│
├── saved_models/                  # Trained models & artifacts
│   ├── best_model.pth            # Best LSTM checkpoint
│   ├── fallback_logistic.pkl     # Fallback model
│   ├── scaler.pkl                # Feature scaler
│   ├── feature_config.json       # Feature metadata
│   ├── test_metrics.json         # Test performance
│   ├── medical_metrics.json      # Clinical metrics
│   └── plots/                    # Evaluation plots
│
├── app/                           # FastAPI inference service
│   └── main.py
│
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Generate Dataset

**Prepare raw telemetry:**
- Place `vitals.jsonl` in `../data/` directory
- Format: JSONL with fields: `patient_id`, `timestamp`, vital signs

**Generate training dataset:**
```bash
cd training
python generate_dataset.py
```

**Output:**
- `data/X.npy` - Sequences (samples, 60, 14)
- `data/y.npy` - Binary labels
- `saved_models/scaler.pkl` - Feature scaler
- `saved_models/feature_config.json` - Metadata

### 3️⃣ Train Model

```bash
cd training
python train.py
```

**Training includes:**
- LSTM with Attention training
- Logistic Regression fallback training
- Early stopping
- Learning rate scheduling
- Comprehensive evaluation

**Output:**
- `saved_models/best_model.pth` - Best LSTM model
- `saved_models/fallback_logistic.pkl` - Fallback model
- `saved_models/plots/` - Training curves & evaluation plots
- `saved_models/test_metrics.json` - Performance metrics

---

## 📊 Model Architecture

### LSTM with Attention

```
Input: (batch, 60, 14)
    ↓
LSTM (2 layers, hidden=128, dropout=0.3)
    ↓
Attention Mechanism (learns important timesteps)
    ↓
FC1 (128 → 64) + Dropout(0.3)
    ↓
FC2 (64 → 32) + Dropout(0.2)
    ↓
Output (32 → 1) + Sigmoid
    ↓
Risk Probability [0, 1]
```

### Features (14 per timestep)

**Raw Vitals:**
1. Heart Rate
2. Systolic BP
3. Diastolic BP
4. SpO2
5. Respiratory Rate
6. Temperature
7. Shock Index

**Engineered Features:**
8. HR Delta (rate of change)
9. SBP Delta
10. SpO2 Delta
11. Shock Index Delta
12. HR Rolling Mean (10-sec window)
13. SBP Rolling Mean
14. SpO2 Rolling Mean

---

## 📈 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 50 (with early stopping) |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | BCELoss |
| Gradient Clipping | 1.0 |
| Early Stopping Patience | 10 epochs |
| Train/Val/Test Split | 70% / 15% / 15% |
| Balanced Sampling | Enabled |

---

## 🔬 Evaluation Metrics

### Standard Metrics
- **Accuracy**: Overall correctness
- **Precision**: Reliability of positive predictions
- **Recall**: Ability to catch deteriorations
- **F1 Score**: Harmonic mean
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve

### Medical Metrics
- **Sensitivity**: Same as Recall (key metric)
- **Specificity**: True negative rate
- **PPV**: Positive Predictive Value
- **NPV**: Negative Predictive Value
- **FNR**: False Negative Rate (critical!)
- **FPR**: False Positive Rate (alert fatigue)

### Priority
✅ **High Recall** (minimize false negatives)  
⚠️ Acceptable precision (balance alert fatigue)

---

## 📦 Saved Models

### LSTM Model (`best_model.pth`)

```python
import torch
from models.lstm_model import LSTMAttentionModel

# Load model
model = LSTMAttentionModel(input_size=14, hidden_size=128, num_layers=2)
checkpoint = torch.load('saved_models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    risk, attention = model(sequence)  # sequence: (batch, 60, 14)
```

### Logistic Regression Fallback (`fallback_logistic.pkl`)

```python
import pickle

# Load model
with open('saved_models/fallback_logistic.pkl', 'rb') as f:
    data = pickle.load(f)
    lr_model = data['model']
    scaler = data['scaler']

# Inference
X_flat = X.reshape(X.shape[0], -1)
X_scaled = scaler.transform(X_flat)
risk = lr_model.predict_proba(X_scaled)[:, 1]
```

---

## 🧪 Testing Models

### Test LSTM Model
```bash
cd models
python lstm_model.py
```

### Test Dataset
```bash
cd training
python dataset.py
```

### Test Metrics
```bash
cd utils
python metrics.py
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t vitalx-ml-service .

# Run container
docker run -p 8000:8000 vitalx-ml-service
```

---

## 📝 Usage Examples

### Generate Dataset
```python
from training.generate_dataset import load_jsonl, build_dataset, normalize_sequences

# Load raw data
df = load_jsonl('../data/vitals.jsonl')

# Build dataset
X, y = build_dataset(df)

# Normalize
X_scaled, scaler = normalize_sequences(X)
```

### Train Model
```python
from training.train import main

# Run full training pipeline
main()
```

### Evaluate Model
```python
from utils.metrics import calculate_metrics, print_metrics_report

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
y_proba = [0.2, 0.8, 0.4, 0.1, 0.9]

metrics = calculate_metrics(y_true, y_pred, y_proba)
print_metrics_report(metrics)
```

---

## 🔧 Configuration

Edit `training/train.py` → `Config` class:

```python
class Config:
    # Model
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    
    # Training
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    
    # Early stopping
    patience = 10
```

---

## 📊 Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| ROC-AUC | > 0.85 | Primary metric |
| Recall | > 0.90 | Critical: catch deteriorations |
| Precision | > 0.70 | Balance alert fatigue |
| F1 Score | > 0.75 | Overall balance |
| FNR | < 0.10 | Max 10% missed cases |

---

## 🔍 Troubleshooting

### Issue: Data files not found
**Solution**: Run `python training/generate_dataset.py` first

### Issue: Out of memory during training
**Solution**: Reduce `batch_size` in `Config`

### Issue: Model overfitting
**Solution**: 
- Increase `dropout`
- Enable data augmentation: `use_augmentation = True`
- Reduce `hidden_size`

### Issue: Poor recall
**Solution**:
- Enable balanced sampling: `use_balanced_sampling = True`
- Adjust classification threshold (default: 0.5)
- Check class distribution in dataset

---

## 📚 Dependencies

See `requirements.txt` for full list:
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Scikit-learn >= 1.3.0
- Matplotlib >= 3.7.0
- FastAPI == 0.104.1
- Uvicorn == 0.24.0

---

## 🤝 Integration with VitalX

This ML service integrates with:
- **Vital Simulator**: Generates telemetry data
- **Pathway Engine**: Real-time stream processing
- **Backend API**: Serves predictions to frontend
- **Digital Twin**: Provides risk scores for dashboard

---

## 📄 License

Part of VitalX Real-Time ICU Digital Twin System

---

## 👨‍💻 Development

### Add New Features
1. Update `FEATURES` list in `generate_dataset.py`
2. Adjust `input_size` in model config
3. Regenerate dataset
4. Retrain model

### Modify Architecture
1. Edit `models/lstm_model.py`
2. Update `Config` in `training/train.py`
3. Retrain from scratch

---

## ✅ Checklist

- [x] Dataset generation script
- [x] LSTM model with attention
- [x] Logistic regression fallback
- [x] PyTorch dataset classes
- [x] Training pipeline with early stopping
- [x] Comprehensive evaluation metrics
- [x] Medical-focused reporting
- [x] Model checkpointing
- [x] Visualization plots
- [x] Docker support
- [ ] FastAPI inference endpoint (TODO)
- [ ] Real-time streaming integration (TODO)

---

**Built for VitalX Digital Twin**  
Real-Time ICU Patient Deterioration Prediction 🏥
