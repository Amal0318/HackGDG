# =================================================================
# LIVE RAG INDEXING DEMO FOR PRESENTATIONS
# =================================================================
# Shows real-time embeddings being created from streaming data
# Perfect for demonstrating the Live RAG architecture
# =================================================================

param(
    [int]$Iterations = 15,
    [int]$RefreshSeconds = 2
)

Write-Host "`n" -NoNewline
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                           ║" -ForegroundColor Cyan
Write-Host "║          LIVE RAG INDEXING - PRESENTATION DEMO            ║" -ForegroundColor Cyan
Write-Host "║          Real-time Embeddings from Kafka Stream           ║" -ForegroundColor Cyan
Write-Host "║                                                           ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$HealthURL = "http://localhost:8080/health"
$QueryURL = "http://localhost:8080/query"
$PreviousCount = 0
$StartTime = Get-Date

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Updates: $Iterations" -ForegroundColor White
Write-Host "  Refresh Rate: ${RefreshSeconds}s" -ForegroundColor White
Write-Host "  Total Duration: $($Iterations * $RefreshSeconds)s`n" -ForegroundColor White

Write-Host "Starting in 3 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 3

for ($i = 1; $i -le $Iterations; $i++) {
    try {
        # Get current stats
        $Stats = Invoke-RestMethod -Uri $HealthURL -Method Get -TimeoutSec 3
        
        # Calculate metrics
        $CurrentCount = $Stats.total_embeddings
        $Delta = $CurrentCount - $PreviousCount
        $Rate = if ($i -gt 1) { [math]::Round($Delta / $RefreshSeconds, 1) } else { 0 }
        $Elapsed = [math]::Round(((Get-Date) - $StartTime).TotalSeconds, 0)
        
        # Clear and display
        Clear-Host
        
        Write-Host ""
        Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
        Write-Host "║          LIVE RAG INDEXING - UPDATE $i/$Iterations" + (" " * (28 - "$i/$Iterations".Length)) + "║" -ForegroundColor Cyan
        Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "  Runtime: ${Elapsed}s" -ForegroundColor Gray
        Write-Host "  Status:  " -NoNewline -ForegroundColor Gray
        Write-Host $Stats.status.ToUpper() -ForegroundColor Green
        Write-Host ""
        
        Write-Host "─────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
        Write-Host ""
        
        # Main stats
        Write-Host "  EMBEDDING STATISTICS" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "    Total Embeddings:   " -NoNewline
        Write-Host ("{0,8:N0}" -f $CurrentCount) -NoNewline -ForegroundColor Green
        if ($Delta -gt 0) {
            Write-Host "   (+$Delta)" -ForegroundColor Magenta
        } else {
            Write-Host ""
        }
        
        Write-Host "    Active Patients:    " -NoNewline
        Write-Host ("{0,8}" -f $Stats.total_patients) -ForegroundColor Cyan
        
        Write-Host "    Indexing Rate:      " -NoNewline
        $RateColor = if ($Rate -gt 30) { "Green" } elseif ($Rate -gt 10) { "Yellow" } else { "White" }
        Write-Host ("{0,8:F1} emb/s" -f $Rate) -ForegroundColor $RateColor
        
        Write-Host ""
        Write-Host "─────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
        Write-Host ""
        
        # Activity visualization
        Write-Host "  STREAMING PIPELINE ACTIVITY" -ForegroundColor Yellow
        Write-Host ""
        
        $BarLength = [math]::Min(40, [math]::Max(1, [int]($Rate * 1.5)))
        $Bar = "█" * $BarLength
        
        Write-Host "    Kafka Stream:       " -NoNewline
        Write-Host $Bar -ForegroundColor Green
        
        Write-Host "    Embedding Gen:      " -NoNewline
        Write-Host $Bar -ForegroundColor Cyan
        
        Write-Host "    Vector Index:       " -NoNewline
        Write-Host $Bar -ForegroundColor Magenta
        
        Write-Host ""
        Write-Host "─────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
        Write-Host ""
        
        # Sample query
        Write-Host "  LIVE DATA SAMPLE (Patient P7)" -ForegroundColor Yellow
        Write-Host ""
        
        $QueryBody = @{
            patient_id = "P7"
            query_text = "latest vitals"
            top_k = 1
        } | ConvertTo-Json
        
        try {
            $QueryResult = Invoke-RestMethod -Uri $QueryURL -Method Post -Body $QueryBody -ContentType "application/json" -TimeoutSec 2
            
            if ($QueryResult.retrieved_context.Count -gt 0) {
                $Context = $QueryResult.retrieved_context[0]
                $TextPreview = $Context.text.Substring(0, [math]::Min(55, $Context.text.Length))
                
                Write-Host "    Latest Entry:       " -NoNewline
                Write-Host "$TextPreview..." -ForegroundColor White
                
                Write-Host "    Relevance Score:    " -NoNewline
                Write-Host ("{0:F3}" -f $Context.relevance_score) -ForegroundColor Green
            }
        } catch {
            Write-Host "    Initializing..." -ForegroundColor Gray
        }
        
        Write-Host ""
        Write-Host "─────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
        Write-Host ""
        
        # Key features
        Write-Host "  LIVE RAG CAPABILITIES" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "    ✓ Real-time embedding (SentenceTransformer)" -ForegroundColor Green
        Write-Host "    ✓ Streaming from Kafka (continuous updates)" -ForegroundColor Green
        Write-Host "    ✓ 3-hour sliding window (auto-expiry)" -ForegroundColor Green
        Write-Host "    ✓ Patient-isolated indices (HIPAA-ready)" -ForegroundColor Green
        Write-Host "    ✓ In-memory vector search (microsecond queries)" -ForegroundColor Green
        Write-Host ""
        
        $PreviousCount = $CurrentCount
        
    } catch {
        Clear-Host
        Write-Host ""
        Write-Host "  ❌ ERROR: Cannot connect to Pathway Engine" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Make sure Docker containers are running:" -ForegroundColor Yellow
        Write-Host "    cd icu-system" -ForegroundColor Gray
        Write-Host "    docker compose ps" -ForegroundColor Gray
        Write-Host ""
        break
    }
    
    if ($i -lt $Iterations) {
        Start-Sleep -Seconds $RefreshSeconds
    }
}

# Final summary
Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    DEMO COMPLETE!                         ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

if ($Stats) {
    Write-Host "  Final Statistics:" -ForegroundColor Yellow
    Write-Host "    Total Embeddings:   " -NoNewline
    Write-Host ("{0:N0}" -f $Stats.total_embeddings) -ForegroundColor Green
    Write-Host "    Active Patients:    " -NoNewline
    Write-Host $Stats.total_patients -ForegroundColor Cyan
    Write-Host "    Demo Duration:      " -NoNewline
    Write-Host "${Elapsed}s" -ForegroundColor White
    Write-Host ""
    Write-Host "  ✓ Live RAG architecture successfully demonstrated!" -ForegroundColor Green
    Write-Host ""
}
