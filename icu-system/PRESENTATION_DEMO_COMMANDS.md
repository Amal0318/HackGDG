# 🎯 LIVE RAG INDEXING DEMO FOR PRESENTATIONS

Perfect for showing real-time embedding generation during your presentation!

---

## 🚀 Quick Demo (Copy & Paste)

### **Option 1: Continuous Monitor (Recommended)**

Copy and paste this into PowerShell:

```powershell
cd icu-system
$prev = 0
$start = Get-Date
1..20 | ForEach-Object {
    $stats = Invoke-RestMethod -Uri 'http://localhost:8080/health' -Method Get
    $delta = $stats.total_embeddings - $prev
    $rate = if ($_ -gt 1) { [math]::Round($delta / 2, 1) } else { 0 }
    $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 0)
    
    Clear-Host
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  LIVE RAG INDEXING - Update #$_" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    Write-Host "Runtime:           ${elapsed}s" -ForegroundColor Gray
    Write-Host "Status:            $($stats.status.ToUpper())`n" -ForegroundColor Green
    
    Write-Host "Total Embeddings:  " -NoNewline -ForegroundColor Yellow
    Write-Host "$($stats.total_embeddings)" -NoNewline -ForegroundColor Green
    if ($delta -gt 0) { Write-Host "  (+$delta NEW)" -ForegroundColor Magenta } else { Write-Host "" }
    
    Write-Host "Active Patients:   " -NoNewline -ForegroundColor Yellow
    Write-Host "$($stats.total_patients)" -ForegroundColor Cyan
    
    Write-Host "Indexing Rate:     " -NoNewline -ForegroundColor Yellow
    Write-Host "$rate emb/sec`n" -ForegroundColor $(if ($rate -gt 30) { "Green" } elseif ($rate -gt 10) { "Yellow" } else { "White" })
    
    $bar = "█" * [math]::Min(40, [math]::Max(1, [int]($rate * 1.5)))
    Write-Host "Kafka Stream:      " -NoNewline -ForegroundColor Yellow
    Write-Host $bar -ForegroundColor Green
    Write-Host "Embedding Gen:     " -NoNewline -ForegroundColor Yellow
    Write-Host $bar -ForegroundColor Cyan
    Write-Host "Vector Index:      " -NoNewline -ForegroundColor Yellow
    Write-Host $bar -ForegroundColor Magenta
    
    Write-Host "`n✓ Real-time streaming from Kafka" -ForegroundColor White
    Write-Host "✓ Live embedding generation (SentenceTransformer)" -ForegroundColor White
    Write-Host "✓ 3-hour sliding window auto-expiry" -ForegroundColor White
    Write-Host "✓ Patient-isolated vector indices`n" -ForegroundColor White
    
    $prev = $stats.total_embeddings
    Start-Sleep -Seconds 2
}
Write-Host "========================================" -ForegroundColor Green
Write-Host "  DEMO COMPLETE!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green
```

**Duration:** 40 seconds (20 updates × 2 sec each)

---

### **Option 2: Simple Stats Loop**

For a quick check:

```powershell
while ($true) {
    $s = Invoke-RestMethod 'http://localhost:8080/health'
    Clear-Host
    Write-Host "`n🔴 LIVE RAG INDEXING`n" -ForegroundColor Cyan
    Write-Host "Embeddings: $($s.total_embeddings)" -ForegroundColor Green
    Write-Host "Patients:   $($s.total_patients)" -ForegroundColor Yellow
    Write-Host "Status:     $($s.status)`n" -ForegroundColor White
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
    Start-Sleep -Seconds 2
}
```

---

### **Option 3: Compare Before/After**

Show the growth:

```powershell
# Get starting stats
$before = Invoke-RestMethod 'http://localhost:8080/health'
Write-Host "`nBEFORE: $($before.total_embeddings) embeddings" -ForegroundColor Yellow

# Wait 10 seconds
Write-Host "Waiting 10 seconds...`n" -ForegroundColor Gray
Start-Sleep -Seconds 10

# Get ending stats
$after = Invoke-RestMethod 'http://localhost:8080/health'
Write-Host "AFTER:  $($after.total_embeddings) embeddings" -ForegroundColor Green

# Show difference
$new = $after.total_embeddings - $before.total_embeddings
Write-Host "`n✓ Created $new NEW embeddings in 10 seconds!" -ForegroundColor Cyan
Write-Host "  Rate: $([math]::Round($new/10, 1)) emb/sec`n" -ForegroundColor White
```

---

## 🎭 Presentation Talking Points

While the demo runs, explain:

### **What's Happening Behind the Scenes:**

1. **Kafka Stream** → Vitals data flowing every second
2. **Pathway Engine** → Enriches data with computed features
3. **Sentence Transformer** → Converts text to 384-dim vectors
4. **Vector Index** → Stores embeddings in patient-isolated indices
5. **Auto-Expiry** → Removes entries older than 3 hours

### **Key Differentiators:**

| Traditional RAG | Your Live RAG |
|----------------|---------------|
| Static documents | Real-time stream |
| Manual re-indexing | Automatic updates |
| Hours old | Seconds old |
| One-time batch | Continuous flow |

### **Technical Highlights:**

- ✅ **In-memory** for microsecond queries
- ✅ **Patient isolation** for HIPAA compliance
- ✅ **Sliding window** for relevance (3 hours)
- ✅ **Semantic search** with relevance scoring
- ✅ **No database** overhead

---

## 📊 API Endpoints You're Calling

### Health Check:
```
GET http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "rag_index_initialized": true,
  "total_patients": 8,
  "total_embeddings": 16789
}
```

### Query Live RAG:
```
POST http://localhost:8080/query
Content-Type: application/json

{
  "patient_id": "P7",
  "query_text": "high shock index",
  "top_k": 5
}
```

Response: 
```json
{
  "patient_id": "P7",
  "query_text": "high shock index",
  "retrieved_context": [
    {
      "text": "Time 07:10 | Patient P7 | HR 121 | SBP 91...",
      "relevance_score": 0.523,
      "timestamp": "2026-02-24T07:10:46Z"
    }
  ]
}
```

---

## 🎬 Presentation Flow

### **1. Setup (Pre-presentation)**
```powershell
cd icu-system
docker compose ps  # Verify all running
```

### **2. Show Current State**
```powershell
Invoke-RestMethod 'http://localhost:8080/health' | Format-List
```

### **3. Run Live Demo**
Run Option 1 (continuous monitor) for 30-40 seconds

### **4. Query Sample Data**
```powershell
$query = @{
    patient_id = "P7"
    query_text = "why is shock index high"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri 'http://localhost:8080/query' -Method Post -Body $query -ContentType 'application/json' | ConvertTo-Json -Depth 5
```

### **5. Show Chat Integration**
```powershell
$chat = @{
    question = "Why is Patient P7 showing a high shock index?"
    patient_id = "P7"
} | ConvertTo-Json

Invoke-RestMethod -Uri 'http://localhost:8000/chat' -Method Post -Body $chat -ContentType 'application/json'
```

---

## 💡 Pro Tips

1. **Run demo on second monitor** during presentation
2. **Increase font size** in terminal for visibility
3. **Explain each metric** as it updates
4. **Highlight delta** (new embeddings) in color
5. **Show rate spikes** when new patients arrive

---

## ⚡ Quick Test Now

Test it works:

```powershell
Invoke-RestMethod 'http://localhost:8080/health'
```

Expected output:
```
status              : healthy
rag_index_initialized : True
total_patients      : 8
total_embeddings    : 16800+
```

---

## 🎯 Summary for Audience

> "What you're seeing is **Live RAG in action**. Unlike traditional RAG systems that index documents once, our system is **continuously embedding streaming vital signs** from 8 ICU patients. Every 2 seconds, you see new embeddings being created. This means when a doctor queries the system, they get context from **data that was streamed just seconds ago**, not hours or days old static documents."

✅ **Mic drop moment for your presentation!** 🚀
