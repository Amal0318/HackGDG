# LangChain Alert System - Fixed and Enabled

## Problem
Patient P8 had high risk but no LangChain alerts were being triggered.

## Root Cause
The **alert-system service was commented out** in `docker-compose.yml`, so it wasn't running at all!

## What Was Fixed

### 1. Uncommented Alert System in Docker Compose
**File:** `icu-system/docker-compose.yml`
- Uncommented the `alert-system` service (lines 138-166)
- Set `ENABLE_CONSOLE_ALERTS: "true"` for visible alerts
- Configured Gemini API for LangChain integration
- Set `HIGH_RISK_THRESHOLD: 0.7` (triggers for risk > 70%)

### 2. Fixed Missing Imports
**File:** `icu-system/alert-engine/app/main.py`
- Added missing `import os` statement

### 3. Created Module Entry Point
**File:** `icu-system/alert-system/app/__main__.py`
- Created entry point for running as module

### 4. Updated Root Docker Compose
**File:** `docker-compose.yml` (root level)
- Also added alert-system service for consistency

### 5. Added API Key to Environment
**File:** `.env`
- Added `GEMINI_API_KEY` for LangChain alerts

## How It Works

```
Patient Data → Kafka (vitals_predictions) → Alert System
                                              ↓
                                         LangChain (Gemini)
                                              ↓
                                    AI-Generated Alert
                                              ↓
                           Console / Email / Webhook
```

### Trigger Conditions
Alerts are sent when ANY of these conditions are met:
- ✅ `is_high_risk == true` (from ML model)
- ✅ `computed_risk > 0.7`
- ✅ `shock_index > 1.0` (CRITICAL)
- ✅ `spo2 < 90%` (CRITICAL)
- ✅ `anomaly_flag == 1`

### Alert Severity Levels
- **CRITICAL**: shock_index > 1.5 OR spo2 < 85 OR risk > 0.9
- **HIGH**: shock_index > 1.0 OR spo2 < 90 OR risk > 0.7
- **MEDIUM**: Other high-risk conditions

## How to Use

### Start the System
```powershell
cd icu-system
docker compose up -d
```

### Monitor Alerts in Real-Time
```powershell
cd icu-system
.\monitor_alerts.ps1
```

### Check Container Status
```powershell
docker ps | Select-String "alert"
```

### View Recent Alerts
```powershell
docker logs icu-alert-system --tail=50
```

## What You Should See

When P8 (or any patient) has high risk, you'll see alerts like:

```
================================================================================
🟠 HIGH ALERT - 2026-02-24 11:45:32 UTC
================================================================================
Patient: P8 | Floor: ICU-2B
--------------------------------------------------------------------------------
SEVERITY: HIGH

KEY FINDINGS:
• Elevated heart rate with declining blood pressure trend
• Shock index 1.15 indicating hemodynamic instability  
• SpO2 at 92% - borderline hypoxemia

RECOMMENDED ACTION:
Immediate bedside assessment. Consider fluid resuscitation and oxygen 
supplementation. Monitor for signs of early sepsis or cardiogenic shock.

CLINICAL REASONING:
Rising shock index with declining BP suggests inadequate tissue perfusion.
Current trajectory indicates potential deterioration within 2-4 hours.
--------------------------------------------------------------------------------
LLM Provider: GEMINI
================================================================================
```

## Rate Limiting
- Alerts for the same patient are **rate-limited to 5 minutes** (300 seconds)
- This prevents alert spam while ensuring timely notifications

## Email Notifications (Optional)
To enable email alerts, set these environment variables:
```bash
ENABLE_EMAIL_ALERTS=true
SMTP_USERNAME=your-gmail@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_TO=doctor@hospital.com
```

## Troubleshooting

### Container Not Starting?
```powershell
docker logs icu-alert-system
```

### No Alerts Appearing?
1. Check if ML service is predicting high risk:
   ```powershell
   docker logs icu-ml-service --tail=20
   ```

2. Check Kafka messages:
   ```powershell
   docker logs icu-kafka --tail=20
   ```

3. Verify Gemini API key is valid in `.env`

### Build Errors?
```powershell
docker compose build alert-system --no-cache
docker compose up -d alert-system
```

## Next Steps

1. ✅ **System is now configured** - alerts will trigger automatically
2. ✅ **Monitor using** `monitor_alerts.ps1` script
3. 🔧 **Customize thresholds** in `docker-compose.yml` if needed
4. 📧 **Enable email** alerts for doctor notifications (optional)

---

**Status:** ✅ FIXED - LangChain alerts are now active!
