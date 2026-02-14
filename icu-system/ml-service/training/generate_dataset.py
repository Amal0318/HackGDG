"""
VitalX Dataset Generation Script (Revised Production Version)
==============================================================

Generates time-series training data for LSTM-based deterioration prediction.

Input:
    data/vitals.jsonl

Output:
    data/X.npy
    data/y.npy
    saved_models/scaler.pkl
    saved_models/feature_config.json
"""

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle
import os

# ==========================================================
# SAFE BASE PATH SETUP (CRITICAL FIX)
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)  # ml-service/
ICU_SYSTEM_DIR = os.path.dirname(BASE_DIR)  # icu-system/
INPUT_DATA_DIR = os.path.join(ICU_SYSTEM_DIR, "data")  # icu-system/data/ (for vitals.jsonl)
DATA_DIR = os.path.join(BASE_DIR, "training", "data")  # ml-service/training/data/ (for X.npy, y.npy)
MODEL_DIR = os.path.join(BASE_DIR, "training", "saved_models")  # ml-service/training/saved_models/

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# CONFIG
# ==========================================================

WINDOW_SIZE = 60
FUTURE_WINDOW = 20

SHOCK_THRESHOLD = 1.3
SHOCK_DURATION = 5
MIN_SPO2 = 88
MAX_HR = 130
MIN_BP = 90

FEATURES = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature",
    "shock_index",
    "hr_delta",
    "sbp_delta",
    "spo2_delta",
    "shock_delta",
    "hr_roll_mean",
    "sbp_roll_mean",
    "spo2_roll_mean",
]

# ==========================================================
# LOAD RAW DATA
# ==========================================================

def load_jsonl(path):
    print(f"Loading data from: {path}")
    records = []

    with open(path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Loaded {len(df)} records")
    print(f"Patients: {df['patient_id'].nunique()}")

    return df

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def engineer_features(df):
    df = df.sort_values("timestamp").copy()

    df["hr_delta"] = df["heart_rate"].diff().fillna(0)
    df["sbp_delta"] = df["systolic_bp"].diff().fillna(0)
    df["spo2_delta"] = df["spo2"].diff().fillna(0)
    df["shock_delta"] = df["shock_index"].diff().fillna(0)

    df["hr_roll_mean"] = df["heart_rate"].rolling(10, min_periods=1).mean()
    df["sbp_roll_mean"] = df["systolic_bp"].rolling(10, min_periods=1).mean()
    df["spo2_roll_mean"] = df["spo2"].rolling(10, min_periods=1).mean()

    # FIXED pandas deprecation
    df = df.bfill().ffill()

    return df

# ==========================================================
# LABELING
# ==========================================================

def check_deterioration(future_df):

    shock_high = (future_df["shock_index"] > SHOCK_THRESHOLD).astype(int)
    if shock_high.rolling(SHOCK_DURATION, min_periods=SHOCK_DURATION).sum().max() >= SHOCK_DURATION:
        return 1

    if "state" in future_df.columns:
        if (future_df["state"] == "CRITICAL").any():
            return 1

    if (future_df["spo2"] < MIN_SPO2).any():
        return 1

    if (future_df["heart_rate"] > MAX_HR).any():
        return 1

    if (future_df["systolic_bp"] < MIN_BP).any():
        return 1

    return 0

# ==========================================================
# BUILD DATASET
# ==========================================================

def build_dataset(df):

    X_sequences = []
    y_labels = []

    patients = df["patient_id"].unique()
    print(f"Processing {len(patients)} patients...")

    for pid in patients:

        patient_df = df[df["patient_id"] == pid].copy()
        patient_df = engineer_features(patient_df)
        patient_df = patient_df.reset_index(drop=True)

        max_idx = len(patient_df) - WINDOW_SIZE - FUTURE_WINDOW

        for i in range(max_idx):

            window_df = patient_df.iloc[i:i + WINDOW_SIZE]
            future_df = patient_df.iloc[i + WINDOW_SIZE:i + WINDOW_SIZE + FUTURE_WINDOW]

            label = check_deterioration(future_df)
            sequence = window_df[FEATURES].values

            if sequence.shape == (WINDOW_SIZE, len(FEATURES)):
                X_sequences.append(sequence)
                y_labels.append(label)

    X = np.array(X_sequences)
    y = np.array(y_labels)

    print("Dataset created:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y

# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_sequences(X):

    samples, seq_len, n_features = X.shape

    X_reshaped = X.reshape(-1, n_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    X_scaled = X_scaled.reshape(samples, seq_len, n_features)

    return X_scaled, scaler

# ==========================================================
# SAVE ARTIFACTS
# ==========================================================

def save_artifacts(X, y, scaler):

    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    config = {
        "features": FEATURES,
        "sequence_length": WINDOW_SIZE,
        "future_window": FUTURE_WINDOW,
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(MODEL_DIR, "feature_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Artifacts saved successfully.")

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("VitalX Dataset Generation (Production Safe)")
    print("=" * 60)

    input_file = os.path.join(INPUT_DATA_DIR, "vitals.jsonl")

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        exit(1)

    print("[1/4] Loading raw telemetry...")
    df = load_jsonl(input_file)

    print("[2/4] Building dataset...")
    X, y = build_dataset(df)

    print("[3/4] Normalizing...")
    X_scaled, scaler = normalize_sequences(X)

    print("[4/4] Saving artifacts...")
    save_artifacts(X_scaled, y, scaler)

    print("=" * 60)
    print("Dataset generation complete.")
    print("Total samples:", len(y))
    print("Final shape:", X_scaled.shape)
    print("=" * 60)
