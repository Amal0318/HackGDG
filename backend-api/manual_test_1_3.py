"""
Manual verification of Phase 1.3 implementation
"""

import requests
import json

BASE = "http://localhost:8000"

print("\n=== Phase 1.3 Manual Verification ===\n")

# 1. Health check
print("[1] Health check:")
r = requests.get(f"{BASE}/health")
print(f"    Status: {r.json()['status']}")
print(f"    Active patients: {r.json()['active_patients']}")

# 2. Admit new patient
patient_id = "MANUAL_TEST"
print(f"\n[2] Admitting patient {patient_id}...")
try:
    requests.delete(f"{BASE}/patients/{patient_id}/discharge")
except:
    pass

r = requests.post(f"{BASE}/patients/{patient_id}/admit")
print(f"    Response: {r.status_code}")

# 3. Send vitals for calibration
print(f"\n[3] Sending 12 vitals for baseline calibration...")
for i in range(12):
    vitals = [75, 98, 120, 16, 37]
    r = requests.post(
        f"{BASE}/patients/{patient_id}/vitals",
        json={"vitals": vitals}
    )
    data = r.json()
    if data["status"] == "collecting":
        print(f"    Sample {i+1}: Collecting ({data['vitals_collected']} collected)")
    elif data["status"] == "calibrated":
        print(f"    Sample {i+1}: ✓ CALIBRATED!")
        print(f"    Confidence: {data['stability_confidence']:.2f}")
        print(f"    Sample count: {data['vitals_collected']}")
        break

# 4. Get baseline
print(f"\n[4] Retrieving baseline...")
r = requests.get(f"{BASE}/patients/{patient_id}/baseline")
if r.status_code == 200:
    baseline = r.json()
    hr_baseline = baseline['baseline_vitals']['HR']
    print(f"    HR: {hr_baseline['mean']:.1f} ± {hr_baseline['std']:.2f}")
    print(f"    Green Zone: [{hr_baseline['green_zone_min']:.1f}, {hr_baseline['green_zone_max']:.1f}]")

# 5. Monitoring mode
print(f"\n[5] Monitoring mode - sending 32 vitals to trigger stability...")
for i in range(32):
    r = requests.post(
        f"{BASE}/patients/{patient_id}/vitals",
        json={"vitals": [75, 98, 120, 16, 37]}
    )
    data = r.json()
    
    if i == 0:
        print(f"    Entered monitoring mode")
    elif i == 29:
        print(f"    After 30 samples: is_stable = {data.get('is_stable', False)}")
    elif i == 31:
        stable_dur = data.get('stable_duration_minutes', 0)
        print(f"    After 32 samples: stable_duration = {stable_dur:.2f} min")

# 6. Get patient state
print(f"\n[6] Patient state:")
r = requests.get(f"{BASE}/patients/{patient_id}")
if r.status_code == 200:
    state = r.json()
    print(f"    Calibration: {state['calibration_status']}")
    print(f"    Risk scores: {len(state.get('recent_risk_scores', []))}")
    print(f"    Stable buffer: {len(state.get('stable_vitals_buffer', []))}")
    print(f"    Stable period: {'Active' if state.get('stable_period_start') else 'Inactive'}")

# 7. Discharge
print(f"\n[7] Discharging patient...")
r = requests.delete(f"{BASE}/patients/{patient_id}/discharge")
print(f"    Response: {r.status_code}")

print("\n=== Phase 1.3 Verification Complete ===\n")
