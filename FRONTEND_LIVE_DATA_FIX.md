# Frontend Live Data Fix - Diagnosis & Solution

## Problem
UI cards showing static data - not updating in real-time

## Root Cause Analysis

### ✅ What's Working
1. **Backend data IS updating** - Tested via API calls, vitals change every second
2. **Kafka pipeline working** - Vitals → Enriched → Predictions flowing correctly
3. **API endpoints working** - `/floors/ICU-1/patients` returning live data with nested vitals
4. **Vite proxy working** - Frontend can access backend via `/api/*` routes

### ❌ What's Not Working  
The React components may not be receiving/processing the updates properly due to:
1. **Silent polling** - useWebSocket hook may be polling but not triggering re-renders
2. **State mutation** - React not detecting state changes
3. **Missing subscriptions** - Floor subscriptions may not be registered correctly

## Solution Applied

### 1. Added Console Logging
Modified the following files with debug logs:
- `frontend/src/hooks/useWebSocket.ts` - Poll cycle tracking
- `frontend/src/hooks/usePatients.ts` - Message handling tracking

### 2. Created Test Page  
**File:** `frontend/public/live-test.html`

**Access:** http://localhost:3000/live-test.html

This pure JavaScript page polls the same API endpoints the React app uses.
- If THIS works → Problem is in React state management
- If THIS doesn't work → Problem is in backend/API

## How to Debug

### Step 1: Test with Live Test Page
```
Open: http://localhost:3000/live-test.html
```

**Expected behavior:**
- Cards update every 2 seconds
- Update counter increments
- Vitals change (HR, BP, SpO₂)
- Timestamp updates

**If this works:** Problem is in React app  
**If this doesn't work:** Problem is in backend

### Step 2: Check Browser Console
Open your React app (http://localhost:3000) and check DevTools Console:

**Look for:**
```
[useWebSocket] Starting polling with interval: 3000
[useWebSocket] Polling tick
[useWebSocket] Fetched 1 floors
[useWebSocket] Fetched 8 patients from floor ICU-1
[useWebSocket] Total patients fetched: 8
[usePatients] Received message: patient_update
[usePatients] Upserting patient: P7
[usePatients] Transformed patient: P7 HR: 82
```

**If you see these logs:**
- Polling IS working
- Data IS being fetched
- Issue is in React re-rendering

**If you DON'T see these logs:**
- Polling may not be starting
- Check if useWebSocket hook is being called
- Check if autoReconnect is true

### Step 3: Network Tab Analysis
Open DevTools → Network tab → Filter XHR

**Expected:**
- Regular calls to `/api/floors` every ~3 seconds
- Regular calls to `/api/floors/ICU-1/patients` every ~3 seconds
- Response shows updated vitals each time

**If NOT seeing regular calls:**
- Polling interval not set up
- useWebSocket not connecting

## Quick Fixes

### Fix 1: Force Polling (if not happening)

Edit `frontend/src/pages/NurseDashboard.tsx`:

```tsx
// Line 26 - Increase logging visibility
const { patients, error } = usePatients({ 
  refreshInterval: 2000  // Reduce to 2 seconds for testing
});

// Add this after the hook call to verify
useEffect(() => {
  console.log('[NurseDashboard] Patients updated:', patients.length);
  patients.forEach(p => {
    console.log(`  ${p.patient_id}: HR=${p.vitals.heart_rate}, Risk=${p.latest_risk_score}`);
  });
}, [patients]);
```

### Fix 2: Verify Data Structure

Check if vitals are being read correctly:

```tsx
// In PatientCard.tsx, add temporary logging
console.log('Patient data:', {
  id: patient.patient_id,
  hr: patient.vitals?.heart_rate,
  rawVitals: patient.vitals
});
```

### Fix 3: Force Re-render with Key

In `NurseDashboard.tsx`, when mapping patients:

```tsx
{sortedPatients.map((patient) => (
  <PatientCard
    key={`${patient.patient_id}-${patient.last_updated}`}  // Force re-render on update
    patient={patient}
    onClick={() => setSelectedPatient(patient)}
  />
))}
```

## API Response Format

### What Backend Returns:
```json
{
  "floor_id": "ICU-1",
  "patients": [
    {
      "patient_id": "P7",
      "floor_id": "ICU-1",
      "risk_score": 0.000004,
      "last_updated": "2026-02-24T11:25:50.180797+00:00",
      "anomaly_flag": false,
      "vitals": {
        "heart_rate": 76.2,
        "systolic_bp": 127.8,
        "diastolic_bp": 91.7,
        "spo2": 97.3
      }
    }
  ]
}
```

### What transformPatientData() Returns:
```typescript
{
  patient_id: "P7",
  name: "Patient P7",
  bed_number: "P7",
  floor: 1,
  latest_risk_score: 0.0004,  // Converted to percentage
  vitals: {
    heart_rate: 76,    // Rounded
    systolic_bp: 128,
    diastolic_bp: 92,
    spo2: 97
    // ... other vitals
  }
}
```

## Testing Commands

### Test 1: Backend Data Updating
```powershell
# Run twice, 3 seconds apart
$p1 = Invoke-RestMethod 'http://localhost:8000/floors/ICU-1/patients'
$p1.patients[0].vitals

# Wait 3 seconds

$p2 = Invoke-RestMethod 'http://localhost:8000/floors/ICU-1/patients'
$p2.patients[0].vitals

# Compare - should be different
```

### Test 2: Frontend Proxy
```powershell
Invoke-RestMethod 'http://localhost:3000/api/floors/ICU-1/patients'
```

### Test 3: Watch Live Updates
```powershell
while ($true) {
  $p = Invoke-RestMethod 'http://localhost:8000/floors/ICU-1/patients'
  Clear-Host
  Write-Host "Patient P7 - HR: $($p.patients[0].vitals heart_rate)"
  Start-Sleep -Seconds 2
}
```

## Expected Console Output (React App)

When working correctly, you should see this every 3 seconds:

```
[useWebSocket] Polling tick
[useWebSocket] Starting poll...
[useWebSocket] Fetched floors: 1
[useWebSocket] Fetched 8 patients from floor ICU-1
[useWebSocket] Total patients fetched: 8
[useWebSocket] Subscriptions: { patients: [], floors: ['ICU-1'] }
[useWebSocket] Poll complete - emitted updates for 8 patients
[usePatients] Received message: floor_update Object
[usePatients] Processing update for: P7
[usePatients] Upserting patient: P7
[usePatients] Transformed patient: P7 HR: 82
[usePatients] Updating existing patient: P7
... (repeated for all 8 patients)
```

## Next Steps

1. **Open test page** → http://localhost:3000/live-test.html
2. **Verify it updates** every 2 seconds
3. **Open main app** → http://localhost:3000
4. **Open console** → Check for polling logs
5. **Open network tab** → Check for API calls

If test page works but main app doesn't, the issue is React-specific (likely state not triggering re-renders).

If test page also doesn't work, there's an API/backend issue.

## For Presentation

The live test page (`live-test.html`) is perfect for demonstrations:
- Clean UI
- Shows real-time updates
- Displays update counter
- Shows all patients‣
- Uses same backend as React app
- No React complexity - pure JavaScript

✅ Use this during your presentation to show live data streaming!
