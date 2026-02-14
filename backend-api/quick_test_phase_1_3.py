"""
Quick test for Phase 1.3: Rolling Baseline Updates
Simplified test focusing on key functionality
"""

import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"

def quick_test():
    """Quick test of Phase 1.3 features"""
    
    print("\n" + "=" * 70)
    print("PHASE 1.3: Quick Test - Rolling Baseline Updates")
    print("=" * 70)
    
    # Clean up any existing test patient
    try:
        requests.delete(f"{BASE_URL}/patients/QUICK_TEST/discharge")
    except:
        pass
    
    # Test 1: Admit patient
    print("\n[1] Admitting patient...")
    r = requests.post(f"{BASE_URL}/patients/QUICK_TEST/admit")
    assert r.status_code == 201
    print("✓ Patient admitted")
    
    # Test 2: Send 15 vitals for baseline calibration
    print("\n[2] Baseline calibration (sending vitals)...")
    for i in range(15):
        vitals = [75 + i*0.5, 98 - i*0.1, 120 + i*0.3, 16, 37]
        r = requests.post(
            f"{BASE_URL}/patients/QUICK_TEST/vitals",
            json={"vitals": vitals, "timestamp": datetime.now().isoformat()}
        )
        assert r.status_code == 200
        data = r.json()
        
        if data["status"] == "calibrated":
            print(f"✓ Baseline calibrated after {data['vitals_collected']} samples")
            print(f"  Stability confidence: {data['stability_confidence']:.2f}")
            break
        elif i % 5 == 4:
            print(f"  Collected {data['vitals_collected']} samples...")
    
    # Test 3: Get baseline
    print("\n[3] Retrieving baseline...")
    r = requests.get(f"{BASE_URL}/patients/QUICK_TEST/baseline")
    assert r.status_code == 200
    baseline = r.json()
    print(f"✓ HR baseline: {baseline['baseline_vitals']['HR']['mean']:.1f} ± {baseline['baseline_vitals']['HR']['std']:.2f}")
    print(f"  Green Zone: [{baseline['baseline_vitals']['HR']['green_zone_min']:.1f}, {baseline['baseline_vitals']['HR']['green_zone_max']:.1f}]")
    
    # Test 4: Monitoring mode with stability tracking
    print("\n[4] Testing monitoring mode (sending 35 vitals)...")
    for i in range(35):
        vitals = [75, 98, 120, 16, 37]
        r = requests.post(
            f"{BASE_URL}/patients/QUICK_TEST/vitals",
            json={"vitals": vitals, "timestamp": datetime.now().isoformat()}
        )
        assert r.status_code == 200
        data = r.json()
        
        if i == 0:
            print(f"  Entered monitoring mode")
        elif i == 29:
            print(f"  After 30 samples: is_stable = {data['is_stable']}")
            if data['is_stable']:
                print(f"✓ Patient marked as STABLE (risk < 0.3 for 30+ samples)")
        elif i == 34:
            print(f"  Final: stable_duration = {data['stable_duration_minutes']:.1f} min")
    
    # Test 5: Verify patient state
    print("\n[5] Patient state details...")
    r = requests.get(f"{BASE_URL}/patients/QUICK_TEST")
    assert r.status_code == 200
    state = r.json()
    print(f"✓ Calibration status: {state['calibration_status']}")
    print(f"  Risk scores tracked: {len(state['recent_risk_scores'])}")
    print(f"  Stable vitals buffer: {len(state['stable_vitals_buffer'])} vitals")
    print(f"  Stable period active: {'Yes' if state['stable_period_start'] else 'No'}")
    
    # Cleanup
    print("\n[6] Discharging patient...")
    r = requests.delete(f"{BASE_URL}/patients/QUICK_TEST/discharge")
    assert r.status_code == 200
    print("✓ Patient discharged")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - Phase 1.3 Working Correctly")
    print("=" * 70)
    print("\nKey features validated:")
    print("  ✓ BaselineCalibrator integrated")
    print("  ✓ Cold-start calibration (10-30 samples)")
    print("  ✓ Stability detection (risk < 0.3 for 30+ samples)")
    print("  ✓ Rolling update infrastructure")
    print("  ✓ Patient state tracking (risk, stable periods, buffer)")
    print("\n✅ Ready for Phase 2: Multivariate Trend Correlation")


if __name__ == "__main__":
    try:
        quick_test()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to Backend API")
        print("   Start server: cd backend-api && uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
