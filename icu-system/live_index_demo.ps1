# Live RAG Indexing Demo - Real-time Embedding Visualization
# Perfect for presentations

param(
    [int]$Duration = 60,
    [int]$RefreshRate = 2
)

$HealthEndpoint = "http://localhost:8080/health"
$QueryEndpoint = "http://localhost:8080/query"
$StartTime = Get-Date
$PreviousEmbeddings = 0
$Iteration = 0

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  LIVE RAG INDEXING DEMO" -ForegroundColor Cyan
Write-Host "  Real-time Streaming Embeddings" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Duration: $Duration seconds | Refresh: ${RefreshRate}s" -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Yellow

Start-Sleep -Seconds 1

try {
    while ($true) {
        $Elapsed = ((Get-Date) - $StartTime).TotalSeconds
        if ($Elapsed -gt $Duration) { break }
        
        $Iteration++
        
        try {
            $Stats = Invoke-RestMethod -Uri $HealthEndpoint -Method Get -TimeoutSec 3
            $Delta = $Stats.total_embeddings - $PreviousEmbeddings
            $Rate = [math]::Round($Delta / $RefreshRate, 1)
            
            Clear-Host
            
            Write-Host "`n========================================" -ForegroundColor Cyan
            Write-Host "  LIVE RAG INDEXING - UPDATE #$Iteration" -ForegroundColor Cyan
            Write-Host "========================================`n" -ForegroundColor Cyan
            
            Write-Host "Runtime:           " -NoNewline -ForegroundColor Yellow
            Write-Host "$([math]::Round($Elapsed, 0))s / ${Duration}s" -ForegroundColor White
            
            Write-Host "Status:            " -NoNewline -ForegroundColor Yellow
            Write-Host "$($Stats.status.ToUpper())" -ForegroundColor Green
            
            Write-Host "`n----------------------------------------" -ForegroundColor Gray
            Write-Host "  EMBEDDING STATISTICS" -ForegroundColor Cyan
            Write-Host "----------------------------------------`n" -ForegroundColor Gray
            
            Write-Host "Total Embeddings:  " -NoNewline -ForegroundColor Yellow
            Write-Host ("{0:N0}" -f $Stats.total_embeddings) -NoNewline -ForegroundColor Green
            if ($Delta -gt 0) {
                Write-Host "  (+$Delta)" -ForegroundColor Magenta
            } else {
                Write-Host ""
            }
            
            Write-Host "Active Patients:   " -NoNewline -ForegroundColor Yellow
            Write-Host $Stats.total_patients -ForegroundColor Green
            
            Write-Host "Indexing Rate:     " -NoNewline -ForegroundColor Yellow
            $RateColor = if ($Rate -gt 30) { "Green" } elseif ($Rate -gt 10) { "Yellow" } else { "Red" }
            Write-Host "$Rate emb/sec" -ForegroundColor $RateColor
            
            Write-Host "`n----------------------------------------" -ForegroundColor Gray
            Write-Host "  STREAMING PIPELINE" -ForegroundColor Cyan
            Write-Host "----------------------------------------`n" -ForegroundColor Gray
            
            $Activity = [math]::Min(50, [math]::Max(5, $Rate))
            $Bar1 = "█" * $Activity
            $Bar2 = "█" * $Activity
            $Bar3 = "█" * $Activity
            
            Write-Host "Kafka Stream:      " -NoNewline -ForegroundColor Yellow
            Write-Host $Bar1 -ForegroundColor Green
            
            Write-Host "Embedding Gen:     " -NoNewline -ForegroundColor Yellow
            Write-Host $Bar2 -ForegroundColor Cyan
            
            Write-Host "Vector Index:      " -NoNewline -ForegroundColor Yellow
            Write-Host $Bar3 -ForegroundColor Magenta
            
            Write-Host "`n----------------------------------------" -ForegroundColor Gray
            Write-Host "  LIVE DATA SAMPLE" -ForegroundColor Cyan
            Write-Host "----------------------------------------`n" -ForegroundColor Gray
            
            # Query sample patient
            $Body = @{
                patient_id = "P7"
                query_text = "latest vitals"
                top_k = 1
            } | ConvertTo-Json
            
            try {
                $Result = Invoke-RestMethod -Uri $QueryEndpoint -Method Post -Body $Body -ContentType "application/json" -TimeoutSec 2
                
                if ($Result.retrieved_context.Count -gt 0) {
                    $Context = $Result.retrieved_context[0]
                    Write-Host "Patient:           " -NoNewline -ForegroundColor Yellow
                    Write-Host $Result.patient_id -ForegroundColor White
                    
                    Write-Host "Latest Data:       " -NoNewline -ForegroundColor Yellow
                    $Text = $Context.text.Substring(0, [math]::Min(50, $Context.text.Length))
                    Write-Host "$Text..." -ForegroundColor White
                    
                    Write-Host "Relevance:         " -NoNewline -ForegroundColor Yellow
                    Write-Host ("{0:F3}" -f $Context.relevance_score) -ForegroundColor Green
                }
            } catch {
                Write-Host "Initializing sample data..." -ForegroundColor Gray
            }
            
            Write-Host "`n----------------------------------------" -ForegroundColor Gray
            Write-Host "  KEY FEATURES" -ForegroundColor Cyan
            Write-Host "----------------------------------------`n" -ForegroundColor Gray
            
            Write-Host "✓ Real-time embedding (SentenceTransformer)" -ForegroundColor Green
            Write-Host "✓ Streaming from Kafka" -ForegroundColor Green
            Write-Host "✓ 3-hour sliding window" -ForegroundColor Green
            Write-Host "✓ Patient-isolated indices" -ForegroundColor Green
            Write-Host "✓ In-memory vector search" -ForegroundColor Green
            
            $PreviousEmbeddings = $Stats.total_embeddings
            
        } catch {
            Clear-Host
            Write-Host "`n❌ Error: Cannot connect to Pathway Engine" -ForegroundColor Red
            Write-Host "Make sure Docker containers are running`n" -ForegroundColor Yellow
        }
        
        Start-Sleep -Seconds $RefreshRate
    }
    
    # Final summary
    Clear-Host
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  DEMO COMPLETE!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
    
    Write-Host "Total Embeddings:  " -NoNewline -ForegroundColor Yellow
    Write-Host ("{0:N0}" -f $Stats.total_embeddings) -ForegroundColor Green
    
    Write-Host "Active Patients:   " -NoNewline -ForegroundColor Yellow
    Write-Host $Stats.total_patients -ForegroundColor Green
    
    Write-Host "Demo Duration:     " -NoNewline -ForegroundColor Yellow
    Write-Host "$([math]::Round($Elapsed, 0)) seconds" -ForegroundColor Green
    
    Write-Host "`n✓ Live RAG successfully demonstrated!`n" -ForegroundColor Green
    
} catch {
    Write-Host "`n❌ Demo interrupted`n" -ForegroundColor Red
}
