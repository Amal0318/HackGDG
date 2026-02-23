"""
Configuration file for VitalX Sepsis Prediction Model
Aligned with PhysioNet 2019 Sepsis Challenge
"""

# ==================== FEATURE CONFIGURATION ====================

# Core Vitals
CORE_VITALS = [
    'HR',      # Heart Rate
    'O2Sat',   # Oxygen Saturation
    'Temp',    # Temperature
    'SBP',     # Systolic Blood Pressure
    'MAP',     # Mean Arterial Pressure
    'DBP',     # Diastolic Blood Pressure
    'Resp',    # Respiration Rate
    'EtCO2'    # End-tidal CO2
]

# Key Labs
KEY_LABS = [
    'Lactate',
    'WBC',            # White Blood Cell Count
    'Creatinine',
    'Platelets',
    'Bilirubin_total'
]

# Demographics
DEMOGRAPHICS = [
    'Age',
    'Gender'
]

# Derived Features (will be computed)
DERIVED_FEATURES = [
    'ShockIndex',          # HR / SBP
    'HR_delta',            # Change in HR
    'SBP_delta',           # Change in SBP
    'ShockIndex_delta',    # Change in ShockIndex
    'RollingMean_HR',      # Rolling mean of HR
    'RollingMean_SBP'      # Rolling mean of SBP
]

# Base features from dataset
BASE_FEATURES = CORE_VITALS + KEY_LABS + DEMOGRAPHICS

# Missingness indicators for critical features
MISSINGNESS_FEATURES = [f'{feat}_missing' for feat in CORE_VITALS + KEY_LABS]

# All features (base + derived + missingness)
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES + MISSINGNESS_FEATURES

# ==================== DATA PROCESSING CONFIGURATION ====================

# Sequence parameters
SEQUENCE_LENGTH = 24  # 24 hours lookback window
TARGET_COLUMN = 'SepsisLabel'
TIME_COLUMN = 'ICULOS'

# Rolling window parameters for derived features
ROLLING_WINDOW = 6  # 6-hour rolling window

# Train/Val/Test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==================== MODEL CONFIGURATION ====================

# Model architecture
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = False
USE_ATTENTION = True

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
RANDOM_SEED = 42

# Class imbalance handling
# Positive class weight (will be computed from data if None)
POS_CLASS_WEIGHT = None  # Auto-compute from training data

# ==================== PATHS ====================

DATA_DIR_A = 'training/training_setA'
DATA_DIR_B = 'training/training_setB'
OUTPUT_DIR = 'outputs'
MODEL_SAVE_PATH = 'outputs/model.pth'
SCALER_SAVE_PATH = 'outputs/scaler.pkl'
CONFIG_SAVE_PATH = 'outputs/feature_config.json'
CHECKPOINT_DIR = 'outputs/checkpoints'

# ==================== FEATURE COUNT ====================

def get_feature_count():
    """Returns the total number of features"""
    return len(ALL_FEATURES)

def print_feature_summary():
    """Print summary of all features"""
    print("\n" + "="*60)
    print("FEATURE CONFIGURATION SUMMARY")
    print("="*60)
    print(f"\nCore Vitals ({len(CORE_VITALS)}):")
    for feat in CORE_VITALS:
        print(f"  - {feat}")
    
    print(f"\nKey Labs ({len(KEY_LABS)}):")
    for feat in KEY_LABS:
        print(f"  - {feat}")
    
    print(f"\nDemographics ({len(DEMOGRAPHICS)}):")
    for feat in DEMOGRAPHICS:
        print(f"  - {feat}")
    
    print(f"\nDerived Features ({len(DERIVED_FEATURES)}):")
    for feat in DERIVED_FEATURES:
        print(f"  - {feat}")
    
    print(f"\nMissingness Indicators ({len(MISSINGNESS_FEATURES)}):")
    print(f"  - {len(MISSINGNESS_FEATURES)} missingness masks")
    
    print("\n" + "-"*60)
    print(f"TOTAL FEATURE COUNT: {get_feature_count()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_feature_summary()
