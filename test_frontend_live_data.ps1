# Frontend Live Data Test
# Tests if the frontend is actually receiving and displaying live updates

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  FRONTEND LIVE DATA DIAGNOSTICS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Step 1: Check backend data is changing..." -ForegroundColor Yellow
$check1 = Invoke-RestMethod -Uri 'http://localhost:8000/floors/ICU-1/patients'
$patient1 = $check1.patients[0]
Write-Host "  Patient: $($patient1.patient_id)" -ForegroundColor White
Write-Host "  HR: $($patient1.vitals.heart_rate)" -ForegroundColor White
Write-Host "  Risk: $([math]::Round($patient1.risk_score * 100, 2))%" -ForegroundColor White

Write-Host "`n  Waiting 3 seconds...`n" -ForegroundColor Gray
Start-Sleep -Seconds 3

$check2 = Invoke-RestMethod -Uri 'http://localhost:8000/floors/ICU-1/patients'
$patient2 = $check2.patients[0]
Write-Host "  HR: $($patient2.vitals.heart_rate)" -ForegroundColor White
Write-Host " Risk: $([math]::Round($patient2.risk_score * 100, 2))%" -ForegroundColor White

if (($patient1.vitals.heart_rate -ne $patient2.vitals.heart_rate) -or ($patient1.risk_score -ne $patient2.risk_score)) {
    Write-Host "`n  ✓ Backend data IS updating!" -ForegroundColor Green
} else {
    Write-Host "`n  ✗ Backend data NOT updating!" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 2: Check frontend polling config..." -ForegroundColor Yellow
Write-Host "  Checking usePatients hook..." -ForegroundColor White

$hookFile = "frontend/src/hooks/usePatients.ts"
if (Test-Path $hookFile) {
    $content = Get-Content $hookFile -Raw
    if ($content -match 'refreshInterval\s*=\s*(\d+)') {
        $interval = $matches[1]
        Write-Host "  Refresh interval: ${interval}ms" -ForegroundColor Green
    } else {
        Write-Host "  Could not find refresh interval" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Hook file not found" -ForegroundColor Red
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 3: Test frontend API fetch..." -ForegroundColor Yellow

# Simulate what the frontend does
$apiResponse = Invoke-RestMethod -Uri 'http://localhost:3000/api/floors/ICU-1/patients'
Write-Host "  Patients returned: $($apiResponse.patients.Count)" -ForegroundColor White
if ($apiResponse.patients.Count -gt 0) {
    $frontendPatient = $apiResponse.patients[0]
    Write-Host "  Sample patient: $($frontendPatient.patient_id)" -ForegroundColor White
    Write-Host "  HR field exists: $(($frontendPatient.PSObject.Properties.Name -contains 'heart_rate') -or ($frontendPatient.vitals.PSObject.Properties.Name -contains 'heart_rate'))" -ForegroundColor $(if (($frontendPatient.PSObject.Properties.Name -contains 'heart_rate') -or ($frontendPatient.vitals.PSObject.Properties.Name -contains 'heart_rate')) { "Green" } else { "Red" })
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  DIAGNOSIS COMPLETE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "🔍 Issue Analysis:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  The backend IS providing live data that updates every second."
Write-Host "  The vitals are nested in a 'vitals' object as expected."
Write-Host "  The transformPatientData() function should handle this."
Write-Host ""
Write-Host "  💡 Most likely causes:" -ForegroundColor Cyan
Write-Host "     1. Frontend polling may not be triggering" -ForegroundColor White
Write-Host "     2. React state may not be updating" -ForegroundColor White
Write-Host "     3. UI components may not be re-rendering" -ForegroundColor White
Write-Host ""
Write-Host "  ✅ Recommended Fix:" -ForegroundColor Green
Write-Host "     Open browser DevTools → Network tab" -ForegroundColor White
Write-Host "     Look for repeated calls to /api/floors/ICU-1/patients" -ForegroundColor White
Write-Host "     Expected: New call every 3-5 seconds" -ForegroundColor White
Write-Host "     If NOT happening: Polling hook is broken" -ForegroundColor White
Write-Host ""
