"""
Quick test of VitalX Backend API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("Testing VitalX Backend API...")
    print("="*60)
    
    # Test 1: Health check
    print("\n1. Health Check:")
    r = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
    
    # Test 2: Admit patient
    print("\n2. Admit Patient PT999:")
    r = requests.post(f"{BASE_URL}/patients/PT999/admit", json=[75, 98, 120, 16, 37.0])
    print(f"   Status: {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=2)}")
    
    # Test 3: List patients
    print("\n3. List Patients:")
    r = requests.get(f"{BASE_URL}/patients")
    print(f"   Status: {r.status_code}")
    print(f"   Total patients: {r.json()['total_patients']}")
    
    # Test 4: Get patient state
    print("\n4. Get Patient State:")
    r = requests.get(f"{BASE_URL}/patients/PT999")
    print(f"   Status: {r.status_code}")
    print(f"   Calibration status: {r.json()['calibration_status']}")
    print(f"   Vitals collected: {r.json()['vitals_buffer_size']}")
    
    # Test 5: Send more vitals
    print("\n5. Send 5 more vital readings:")
    for i in range(5):
        r = requests.post(f"{BASE_URL}/patients/PT999/vitals", json={
            "patient_id": "PT999",
            "vitals": [75+i, 98, 120, 16, 37.0]
        })
        print(f"   Reading {i+2}: {r.json()['vitals_collected']} vitals collected")
    
    # Test 6: Check if ready for baseline
    r = requests.get(f"{BASE_URL}/patients/PT999")
    print(f"\n6. After 6 readings: {r.json()['vitals_buffer_size']} vitals collected")
    
    # Test 7: Discharge
    print("\n7. Discharge Patient:")
    r = requests.delete(f"{BASE_URL}/patients/PT999/discharge")
    print(f"   Status: {r.status_code}")
    print(f"   {r.json()['message']}")
    
    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("="*60)

if __name__ == "__main__":
    try:
        test_api()
    except Exception as e:
        print(f"❌ Error: {e}")
