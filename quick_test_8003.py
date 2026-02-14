"""Quick single test"""
import requests

url = "http://localhost:8003"

# Admit patient
r = requests.post(f"{url}/patients/PT-QUICK/admit")
print(f"Admit: {r.status_code}")

# Build baseline (10)
for i in range(10):
    r = requests.post(f"{url}/patients/PT-QUICK/vitals", json={"vitals": [75,98,120,16,37]})
    print(f"Baseline {i+1}: {r.status_code}")
    if r.status_code != 200:
        print(f"ERROR: {r.text}")
        break

# Enter monitoring
print("\nEntering monitoring phase...")
r = requests.post(f"{url}/patients/PT-QUICK/vitals", json={"vitals": [75,98,120,16,37]})
print(f"Monitoring: {r.status_code}")
if r.status_code != 200:
    print(f"ERROR: {r.text}")
else:
    print(f"SUCCESS: {r.json()}")
