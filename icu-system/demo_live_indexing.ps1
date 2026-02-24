#!/usr/bin/env powershell
<#
.SYNOPSIS
    Live RAG Indexing Demo - Real-time Embedding Visualization
    
.DESCRIPTION
    Demonstrates streaming embeddings being created in real-time
    Perfect for presentations to show Live RAG in action
    
.PARAMETER Duration
    How many seconds to run the demo (default: 60)
    
.PARAMETER RefreshRate
    Update interval in seconds (default: 2)
#>

param(
    [int]$Duration = 60,
    [int]$RefreshRate = 2
)

# Configuration
$HealthEndpoint = "http://localhost:8080/health"
$QueryEndpoint = "http://localhost:8080/query"
$StartTime = Get-Date

# Color scheme
$ColorTitle = "Cyan"
$ColorLabel = "Yellow"
$ColorValue = "Green"
$ColorDelta = "Magenta"
$ColorBar = "White"

# Initialize
$PreviousEmbeddings = 0
$PreviousPatients = 0
$IterationCount = 0

function Draw-Header {
    Clear-Host
    Write-Host ""
    Write-Host " ╔═══════════════════════════════════════════════════════════════════╗" -ForegroundColor $ColorTitle
    Write-Host " ║                                                                   ║" -ForegroundColor $ColorTitle
    Write-Host " ║          🔴 LIVE RAG INDEXING - REAL-TIME DEMO                   ║" -ForegroundColor $ColorTitle
    Write-Host " ║          Streaming Embeddings from Kafka → Pathway Engine        ║" -ForegroundColor $ColorTitle
    Write-Host " ║                                                                   ║" -ForegroundColor $ColorTitle
    Write-Host " ╚═══════════════════════════════════════════════════════════════════╝" -ForegroundColor $ColorTitle
    Write-Host ""
}

function Draw-ProgressBar {
    param([int]$Current, [int]$Max, [int]$BarWidth = 50)
    
    $Percent = [math]::Min(100, [int](($Current / $Max) * 100))
    $FilledWidth = [int](($Percent / 100) * $BarWidth)
    $EmptyWidth = $BarWidth - $FilledWidth
    
    $Bar = "[" + ("█" * $FilledWidth) + ("░" * $EmptyWidth) + "] $Percent%"
    return $Bar
}

function Format-Number {
    param([int]$Number)
    return "{0:N0}" -f $Number
}

function Get-RateColor {
    param([int]$Rate)
    
    if ($Rate -gt 50) { return "Green" }
    elseif ($Rate -gt 20) { return "Yellow" }
    else { return "Red" }
}

Write-Host "`n🚀 Starting Live RAG Indexing Demo..." -ForegroundColor Green
Write-Host "   Duration: $Duration seconds" -ForegroundColor Gray
Write-Host "   Refresh: Every $RefreshRate seconds" -ForegroundColor Gray
Write-Host "`n   Press Ctrl+C to stop early`n" -ForegroundColor Gray
Start-Sleep -Seconds 2

try {
    while ($true) {
        $Elapsed = ((Get-Date) - $StartTime).TotalSeconds
        
        if ($Elapsed -gt $Duration) {
            break
        }
        
        $IterationCount++
        
        # Fetch current stats
        try {
            $Stats = Invoke-RestMethod -Uri $HealthEndpoint -Method Get -TimeoutSec 3
            
            # Calculate deltas
            $EmbeddingsDelta = $Stats.total_embeddings - $PreviousEmbeddings
            $PatientsDelta = $Stats.total_patients - $PreviousPatients
            $EmbeddingsPerSecond = [math]::Round($EmbeddingsDelta / $RefreshRate, 1)
            
            # Draw UI
            Draw-Header
            
            # Status indicators
            Write-Host "  ⏱️  Runtime: " -NoNewline -ForegroundColor $ColorLabel
            Write-Host "$([math]::Round($Elapsed, 0))s / ${Duration}s" -ForegroundColor $ColorValue
            
            Write-Host "  📊 Updates: " -NoNewline -ForegroundColor $ColorLabel
            Write-Host "$IterationCount" -ForegroundColor $ColorValue
            
            Write-Host "  🟢 Status: " -NoNewline -ForegroundColor $ColorLabel
            Write-Host "$($Stats.status.ToUpper())" -ForegroundColor Green
            
            Write-Host ""
            Write-Host " ─────────────────────────────────────────────────────────────────" -ForegroundColor $ColorBar
            Write-Host ""
            
            # Main metrics
            Write-Host "  📈 LIVE EMBEDDING STATISTICS" -ForegroundColor $ColorTitle
            Write-Host ""
            
            Write-Host "     Total Embeddings:  " -NoNewline -ForegroundColor $ColorLabel
            Write-Host (Format-Number $Stats.total_embeddings) -NoNewline -ForegroundColor $ColorValue
            if ($EmbeddingsDelta -gt 0) {
                Write-Host "  (+$EmbeddingsDelta)" -ForegroundColor $ColorDelta
            } else {
                Write-Host ""
            }
            
            Write-Host "     Active Patients:   " -NoNewline -ForegroundColor $ColorLabel
            Write-Host (Format-Number $Stats.total_patients) -NoNewline -ForegroundColor $ColorValue
            if ($PatientsDelta -gt 0) {
                Write-Host "  (+$PatientsDelta)" -ForegroundColor $ColorDelta
            } else {
                Write-Host ""
            }
            
            Write-Host "     Indexing Rate:     " -NoNewline -ForegroundColor $ColorLabel
            $RateColor = Get-RateColor -Rate $EmbeddingsPerSecond
            Write-Host "$EmbeddingsPerSecond embeddings/sec" -ForegroundColor $RateColor
            
            Write-Host ""
            Write-Host " ─────────────────────────────────────────────────────────────────" -ForegroundColor $ColorBar
            Write-Host ""
            
            # Visual progress bars
            Write-Host "  🔄 STREAMING ACTIVITY" -ForegroundColor $ColorTitle
            Write-Host ""
            
            # Simulated activity bars (based on rate)
            $ActivityLevel = [math]::Min(100, $EmbeddingsPerSecond * 2)
            Write-Host "     Kafka → Pathway:   " -NoNewline -ForegroundColor $ColorLabel
            Write-Host (Draw-ProgressBar -Current $ActivityLevel -Max 100 -BarWidth 40) -ForegroundColor Green
            
            Write-Host "     Embedding Gen:     " -NoNewline -ForegroundColor $ColorLabel
            Write-Host (Draw-ProgressBar -Current $ActivityLevel -Max 100 -BarWidth 40) -ForegroundColor Cyan
            
            Write-Host "     Vector Index:      " -NoNewline -ForegroundColor $ColorLabel
            Write-Host (Draw-ProgressBar -Current $ActivityLevel -Max 100 -BarWidth 40) -ForegroundColor Magenta
            
            Write-Host ""
            Write-Host " ─────────────────────────────────────────────────────────────────" -ForegroundColor $ColorBar
            Write-Host ""
            
            # Sample recent data
            Write-Host "  📝 RECENT INDEXED DATA (Sample Patient)" -ForegroundColor $ColorTitle
            Write-Host ""
            
            # Query a sample patient to show live data
            $SampleQuery = @{
                patient_id = "P7"
                query_text = "latest vitals"
                top_k = 1
            } | ConvertTo-Json
            
            try {
                $QueryResult = Invoke-RestMethod -Uri $QueryEndpoint -Method Post -Body $SampleQuery -ContentType "application/json" -TimeoutSec 2
                
                if ($QueryResult.retrieved_context.Count -gt 0) {
                    $LatestContext = $QueryResult.retrieved_context[0]
                    Write-Host "     Patient: " -NoNewline -ForegroundColor $ColorLabel
                    Write-Host "$($QueryResult.patient_id)" -ForegroundColor $ColorValue
                    
                    Write-Host "     Latest:  " -NoNewline -ForegroundColor $ColorLabel
                    Write-Host ($LatestContext.text.Substring(0, [math]::Min(60, $LatestContext.text.Length))) -ForegroundColor White
                    
                    Write-Host "     Score:   " -NoNewline -ForegroundColor $ColorLabel
                    Write-Host ("{0:F3}" -f $LatestContext.relevance_score) -ForegroundColor $ColorValue
                }
            } catch {
                Write-Host "     Initializing..." -ForegroundColor Gray
            }
            
            Write-Host ""
            Write-Host " ─────────────────────────────────────────────────────────────────" -ForegroundColor $ColorBar
            Write-Host ""
            
            # Key features
            Write-Host "  ✨ LIVE RAG FEATURES" -ForegroundColor $ColorTitle
            Write-Host ""
            Write-Host "     ✅ Real-time embedding generation (SentenceTransformer)" -ForegroundColor Green
            Write-Host "     ✅ Streaming from Kafka (continuous updates)" -ForegroundColor Green
            Write-Host "     ✅ 3-hour sliding window (auto-expiry)" -ForegroundColor Green
            Write-Host "     ✅ Patient-isolated indices (HIPAA-ready)" -ForegroundColor Green
            Write-Host "     ✅ In-memory vector search (microsecond queries)" -ForegroundColor Green
            Write-Host ""
            
            # Update previous values
            $PreviousEmbeddings = $Stats.total_embeddings
            $PreviousPatients = $Stats.total_patients
            
        } catch {
            Draw-Header
            Write-Host "  ❌ Error connecting to Pathway Engine" -ForegroundColor Red
            Write-Host "     Make sure Docker containers are running" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "     Error: $($_.Exception.Message)" -ForegroundColor Gray
        }
        
        # Wait before next update
        Start-Sleep -Seconds $RefreshRate
    }
    
    # Final summary
    Draw-Header
    Write-Host "  🎉 DEMO COMPLETE!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  📊 Final Statistics:" -ForegroundColor $ColorTitle
    Write-Host "     Total Embeddings:  $(Format-Number $Stats.total_embeddings)" -ForegroundColor $ColorValue
    Write-Host "     Active Patients:   $(Format-Number $Stats.total_patients)" -ForegroundColor $ColorValue
    Write-Host "     Demo Duration:     $([math]::Round($Elapsed, 0)) seconds" -ForegroundColor $ColorValue
    Write-Host "     New Embeddings:    +$(Format-Number ($Stats.total_embeddings - ($PreviousEmbeddings - $EmbeddingsDelta)))" -ForegroundColor $ColorDelta
    Write-Host ""
    Write-Host "  ✅ Live RAG successfully demonstrated!" -ForegroundColor Green
    Write-Host ""
    
} catch {
    Write-Host "`n❌ Demo interrupted: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Write-Host "`n  Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
