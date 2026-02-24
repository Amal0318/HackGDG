# LIVE RAG INDEXING DEMO - Presentation Ready
# Shows real-time embeddings being created

$iterations = 15
$refresh = 2
$health_url = "http://localhost:8080/health"
$query_url = "http://localhost:8080/query"
$prev = 0
$start = Get-Date

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "   LIVE RAG INDEXING DEMO" -ForegroundColor Cyan
Write-Host "   Real-time Streaming Embeddings" -ForegroundColor Cyan
Write-Host "===============================================`n" -ForegroundColor Cyan

Write-Host "Starting demo with $iterations updates...`n" -ForegroundColor Yellow
Start-Sleep -Seconds 2

for ($i = 1; $i -le $iterations; $i++) {
    try {
        $stats = Invoke-RestMethod -Uri $health_url -Method Get -TimeoutSec 3
        $current = $stats.total_embeddings
        $delta = $current - $prev
        $rate = if ($i -gt 1) { [math]::Round($delta / $refresh, 1) } else { 0 }
        $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 0)
        
        Clear-Host
        
        Write-Host "`n===============================================" -ForegroundColor Cyan
        Write-Host "   UPDATE $i of $iterations - ${elapsed}s elapsed" -ForegroundColor Cyan
        Write-Host "===============================================`n" -ForegroundColor Cyan
        
        Write-Host "STATUS: " -NoNewline -ForegroundColor Yellow
        Write-Host "$($stats.status.ToUpper())`n" -ForegroundColor Green
        
        Write-Host "------- EMBEDDING STATISTICS -------`n" -ForegroundColor Gray
        
        Write-Host "  Total Embeddings:  " -NoNewline
        Write-Host "$current" -NoNewline -ForegroundColor Green
        if ($delta -gt 0) {
            Write-Host "  (+$delta NEW)" -ForegroundColor Magenta
        } else {
            Write-Host ""
        }
        
        Write-Host "  Active Patients:   " -NoNewline
        Write-Host "$($stats.total_patients)" -ForegroundColor Cyan
        
        Write-Host "  Indexing Rate:     " -NoNewline
        if ($rate -gt 30) {
            Write-Host "$rate emb/sec" -ForegroundColor Green
        } elseif ($rate -gt 10) {
            Write-Host "$rate emb/sec" -ForegroundColor Yellow
        } else {
            Write-Host "$rate emb/sec" -ForegroundColor White
        }
        
        Write-Host "`n------- STREAMING PIPELINE -------`n" -ForegroundColor Gray
        
        $bar_size = [math]::Min(40, [math]::Max(1, [int]($rate * 1.5)))
        $activity = "█" * $bar_size
        
        Write-Host "  Kafka Stream:      " -NoNewline
        Write-Host $activity -ForegroundColor Green
        
        Write-Host "  Embedding Gen:     " -NoNewline
        Write-Host $activity -ForegroundColor Cyan
        
        Write-Host "  Vector Index:      " -NoNewline
        Write-Host $activity -ForegroundColor Magenta
        
        Write-Host "`n------- LIVE DATA SAMPLE -------`n" -ForegroundColor Gray
        
        $query_body = @{
            patient_id = "P7"
            query_text = "latest vitals"
            top_k = 1
        } | ConvertTo-Json
        
        try {
            $result = Invoke-RestMethod -Uri $query_url -Method Post -Body $query_body -ContentType "application/json" -TimeoutSec 2
            
            if ($result.retrieved_context.Count -gt 0) {
                $context = $result.retrieved_context[0]
                $preview = $context.text.Substring(0, [math]::Min(50, $context.text.Length))
                
                Write-Host "  Patient:           $($result.patient_id)" -ForegroundColor White
                Write-Host "  Latest Entry:      $preview..." -ForegroundColor White
                Write-Host "  Relevance Score:   $($context.relevance_score)" -ForegroundColor Green
            }
        } catch {
            Write-Host "  Initializing sample data..." -ForegroundColor Gray
        }
        
        Write-Host "`n------- KEY FEATURES -------`n" -ForegroundColor Gray
        
        Write-Host "  ✓ Real-time embedding generation" -ForegroundColor Green
        Write-Host "  ✓ Streaming from Kafka" -ForegroundColor Green
        Write-Host "  ✓ 3-hour sliding window" -ForegroundColor Green
        Write-Host "  ✓ Patient-isolated indices" -ForegroundColor Green
        Write-Host "  ✓ In-memory vector search" -ForegroundColor Green
        Write-Host ""
        
        $prev = $current
        
    } catch {
        Clear-Host
        Write-Host "`n❌ ERROR: Cannot connect to Pathway Engine`n" -ForegroundColor Red
        Write-Host "Make sure Docker containers are running:`n" -ForegroundColor Yellow
        Write-Host "  cd icu-system" -ForegroundColor Gray
        Write-Host "  docker compose ps`n" -ForegroundColor Gray
        break
    }
    
    if ($i -lt $iterations) {
        Start-Sleep -Seconds $refresh
    }
}

Write-Host "`n===============================================" -ForegroundColor Green
Write-Host "   DEMO COMPLETE!" -ForegroundColor Green
Write-Host "===============================================`n" -ForegroundColor Green

if ($stats) {
    Write-Host "Final Results:" -ForegroundColor Yellow
    Write-Host "  Total Embeddings:  $($stats.total_embeddings)" -ForegroundColor Green
    Write-Host "  Active Patients:   $($stats.total_patients)" -ForegroundColor Cyan
    Write-Host "  Demo Duration:     ${elapsed}s" -ForegroundColor White
    Write-Host "`n✓ Live RAG successfully demonstrated!`n" -ForegroundColor Green
}
