"""
Convert Synthetic CSV to JSONL Format
======================================

Converts synthetic_mimic_style_icU.csv to vitals.jsonl format with all required fields.
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta

def calculate_shock_index(hr, sbp):
    """Calculate shock index (HR / SBP)."""
    return hr / sbp if sbp > 0 else 0

def assign_state(row):
    """Assign patient state based on vitals."""
    # Criteria for CRITICAL state
    if row['spo2'] < 88 or row['heart_rate'] > 130 or row['systolic_bp'] < 90:
        return 'CRITICAL'
    # Criteria for ACUTE state
    elif row['spo2'] < 92 or row['heart_rate'] > 110 or row['systolic_bp'] < 100:
        return 'ACUTE'
    else:
        return 'STABLE'

def convert_csv_to_jsonl(csv_file, output_file, max_records=None):
    """
    Convert CSV to JSONL format.
    
    Args:
        csv_file: path to CSV file
        output_file: path to output JSONL file
        max_records: maximum number of records to convert (None = all)
    """
    print(f"Loading CSV: {csv_file}")
    
    # Read CSV
    if max_records:
        df = pd.read_csv(csv_file, nrows=max_records)
    else:
        df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Add missing columns
    print("\nAdding missing fields...")
    
    # Add diastolic_bp (approximate as 60% of systolic)
    if 'diastolic_bp' not in df.columns:
        df['diastolic_bp'] = df['systolic_bp'] * 0.6
    
    # Calculate shock index
    df['shock_index'] = df.apply(
        lambda row: calculate_shock_index(row['heart_rate'], row['systolic_bp']), 
        axis=1
    )
    
    # Assign state
    df['state'] = df.apply(assign_state, axis=1)
    
    # Convert timestamp to datetime
    base_time = datetime.now()
    df['timestamp'] = df['timestamp'].apply(
        lambda t: (base_time + timedelta(seconds=int(t))).isoformat()
    )
    
    # Convert patient_id to string format
    df['patient_id'] = 'P' + df['patient_id'].astype(str)
    
    # Select and reorder columns
    output_columns = [
        'patient_id', 
        'timestamp', 
        'heart_rate', 
        'systolic_bp', 
        'diastolic_bp',
        'spo2', 
        'respiratory_rate', 
        'temperature',
        'shock_index',
        'state'
    ]
    
    df_output = df[output_columns]
    
    # Write to JSONL
    print(f"\nWriting to JSONL: {output_file}")
    records_written = 0
    
    with open(output_file, 'w') as f:
        for _, row in df_output.iterrows():
            record = row.to_dict()
            json.dump(record, f)
            f.write('\n')
            records_written += 1
            
            if records_written % 10000 == 0:
                print(f"  Written {records_written} records...")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total records: {records_written}")
    print(f"   Patients: {df['patient_id'].nunique()}")
    print(f"   Output: {output_file}")
    
    # Sample stats
    print(f"\n📊 Data Statistics:")
    print(f"   States - STABLE: {(df['state'] == 'STABLE').sum()}, "
          f"ACUTE: {(df['state'] == 'ACUTE').sum()}, "
          f"CRITICAL: {(df['state'] == 'CRITICAL').sum()}")
    print(f"   Heart Rate: {df['heart_rate'].min():.1f} - {df['heart_rate'].max():.1f}")
    print(f"   SpO2: {df['spo2'].min():.1f} - {df['spo2'].max():.1f}")
    print(f"   Systolic BP: {df['systolic_bp'].min():.1f} - {df['systolic_bp'].max():.1f}")


if __name__ == "__main__":
    import os
    
    # Paths
    csv_file = "../synthetic_mimic_style_icU.csv"
    output_file = "../../data/vitals.jsonl"
    
    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found: {csv_file}")
        print("Please ensure synthetic_mimic_style_icU.csv exists in the ml-service directory")
        exit(1)
    
    # Convert (use subset for faster testing, or None for all)
    # Use max_records=50000 for faster testing or None for full dataset
    convert_csv_to_jsonl(csv_file, output_file, max_records=None)
    
    print("\n✅ Ready to generate dataset!")
    print("Next step: Run 'python generate_dataset.py'")
