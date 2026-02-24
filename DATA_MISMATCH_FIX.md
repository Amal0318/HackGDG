# Data Mismatch Fix - Current Vitals vs Chart

## Problem
In the Patient Detail panel:
- **Current Vitals** section shows HR = 102 bpm
- **Vital Signs Trend chart** shows HR = 83.5 bpm

Two different values for the SAME patient at the SAME time!

## Root Cause

### Data Flow:
```
Backend API
    ↓
[Patient Prop] ────────────> Current Vitals Section (HR: 102) ✅
    |
    ↓
[WebSocket Updates] ──────>  Vitals History Hook ────> Chart (HR: 83.5) ❌
```

### The Issue:
The **vitals history hook** expects flat structure:
```javascript
{
  heart_rate: 102,
  systolic_bp: 120,
  spo2: 98
}
```

But backend returns **nested structure**:
```javascript
{
  vitals: {
    heart_rate: 102,
    systolic_bp: 120,
    spo2: 98
  },
  last_updated: "2026-02-24T12:00:00Z"
}
```

**Result:** The chart hook wasn't extracting vitals from the nested object!

## Fix Applied

### Modified Files:

#### 1. `frontend/src/hooks/usePatientVitalsHistory.ts` (Line 26-50)
**Before:**
```typescript
if (pid && data && (data.heart_rate !== undefined || data.rolling_hr !== undefined)) {
  const newPoint: VitalsDataPoint = {
    timestamp: data.timestamp || ...,
    heart_rate: data.heart_rate || data.rolling_hr || 0,
    systolic_bp: data.systolic_bp || data.rolling_sbp || 0,
    ...
  };
}
```

**After:**
```typescript
if (pid && data) {
  // Extract vitals from nested structure OR flat structure
  const vitals = data.vitals || data;
  
  if (vitals.heart_rate !== undefined || vitals.rolling_hr !== undefined) {
    const newPoint: VitalsDataPoint = {
      timestamp: data.last_updated || data.timestamp || ...,
      heart_rate: vitals.heart_rate || vitals.rolling_hr || 0,
      systolic_bp: vitals.systolic_bp || vitals.rolling_sbp || 0,
      ...
    };
  }
}
```

#### 2. `frontend/src/hooks/usePatientRiskHistory.ts` (Line 20-40)
**Added:**
- Support for `last_updated` timestamp field
- Cleaner console logging

## How To Verify The Fix

### Step 1: Check Browser Console
After refresh, you should see:
```
💓 Vitals update for P3: { hr: 102, sbp: 105, spo2: 93, timestamp: "2026-02-24T12:00:00Z" }
```

### Step 2: Watch The Chart
- Open patient detail panel
- The chart values should NOW match the Current Vitals section
- Both should update together every 2-3 seconds

### Step 3: Visual Verification
```
Current Vitals: HR = 102 bpm
Chart Latest:   HR = 102 bpm  ✅ NOW MATCHING!
```

## Why This Happened

The backend API structure changed to return nested vitals:
```json
{
  "patient_id": "P3",
  "floor_id": "ICU-1",
  "risk_score": 0.00001,
  "last_updated": "2026-02-24T12:00:00Z",
  "vitals": {               ← NESTED HERE
    "heart_rate": 102,
    "systolic_bp": 105,
    "spo2": 93
  }
}
```

But the vitals history hook was still expecting the OLD flat structure where vitals were at the root level.

## Additional Benefits

The fix also handles:
- ✅ Both `last_updated` and `timestamp` fields
- ✅ Backward compatibility with flat structure
- ✅ Better null/undefined handling
- ✅ Cleaner logging for debugging

## Related Components

Components that were affected:
- ✅ **PatientDetailDrawer** - Shows current vitals
- ✅ **TrendsView** - Shows charts
- ✅ **VitalsTrendChart** - Renders the actual chart
- ✅ **RiskTrendChart** - Risk score over time

All now use consistent data sources and should show synchronized values.

## Testing Checklist

- [ ] Open Patient Detail panel
- [ ] Check Current Vitals section
- [ ] Check Vital Signs Trend chart
- [ ] Verify values match
- [ ] Wait 3 seconds
- [ ] Verify both update together
- [ ] Check browser console for logs
- [ ] Verify no errors

## Live Monitoring

After the fix, open browser console and watch:
```
[useWebSocket] Polling tick
💓 Vitals update for P3: { hr: 102, sbp: 105, spo2: 93 }
📊 Risk update for P3: 0.00%
[usePatients] Upserting patient: P3
```

Every 3 seconds, all three systems should update together!
