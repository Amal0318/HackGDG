# Alert System Monitor
# This script checks if the LangChain alert-system is running and shows real-time alerts

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  LANGCHAIN ALERT SYSTEM MONITOR" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Checking alert-system container status..." -ForegroundColor Yellow

# Check if container exists
$containerStatus = docker container inspect icu-alert-system --format '{{.State.Status}}' 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Container Status: $containerStatus" -ForegroundColor Green
    
    Write-Host "`nShowing last 30 log lines:" -ForegroundColor Yellow
    Write-Host "----------------------------------------`n" -ForegroundColor Gray
    
    docker logs icu-alert-system --tail=30
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  LIVE ALERT MONITORING (Ctrl+C to stop)" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    # Follow logs in real-time
    docker logs icu-alert-system --follow
    
} else {
    Write-Host "❌ Container 'icu-alert-system' not found!" -ForegroundColor Red
    Write-Host "`nContainer might still be building. Run this command to check:" -ForegroundColor Yellow
    Write-Host "    docker compose ps" -ForegroundColor White
    Write-Host "`nOnce built, run this script again to monitor alerts.`n" -ForegroundColor Gray
}
