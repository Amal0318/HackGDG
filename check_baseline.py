"""Check baseline format after calibration"""
import requests
import json

BACKEND = "http://localhost:8000"

print("\n1. Admit CHECK_BASELINE patient...")
r = requests.post(f"{BACKEND}/patients/CHECK_BASELINE/admit")
print(f"  {r.json()['message']}")

print("\n2. Send 12 vitals for calibration...")
for i in range(12):
    vitals = [75, 98, 120, 16, 37.0]
    r = requests.post(f"{BACKEND}/patients/CHECK_BASELINE/vitals", json={"vitals": vitals})
    if r.json().get('status') == 'calibrated':
        print(f"  ✓ Calibrated after {i+1} vitals")
        baseline = r.json().get('baseline_vitals')
        print(f"\n3. Baseline format:")
        print(json.dumps(baseline, indent=2))
        break

print("\n4. Check patient state...")
r = requests.get(f"{BACKEND}/patients/CHECK_BASELINE")
if r.status_code == 200:
    patient = r.json()['patient']
    print(f"  Calibration: {patient['calibration_status']}")
    print(f"  Has baseline: {patient.get('baseline_vitals') is not None}")
    if patient.get('baseline_vitals'):
        print(f"  Baseline keys: {list(patient['baseline_vitals'].keys())}")

print("\n5. Discharge...")
requests.delete(f"{BACKEND}/patients/CHECK_BASELINE/discharge")
