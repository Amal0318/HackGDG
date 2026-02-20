# RAG Service Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Get FREE Groq API Key (30 seconds)

1. Visit: https://console.groq.com/
2. Sign up with Google (instant)
3. Click "API Keys" → "Create API Key"
4. Copy your key (starts with `gsk_`)

### Step 2: Configure API Key (30 seconds)

Create `.env` file in project root:

```bash
# On Windows (PowerShell)
echo "GROQ_API_KEY=gsk_your_actual_key_here" > .env

# On Linux/Mac
echo "GROQ_API_KEY=gsk_your_actual_key_here" > .env
```

### Step 3: Start the System (2 minutes)

```bash
docker-compose up -d --build rag-service
```

Wait ~1 minute for data indexing to begin.

### Step 4: Test It! (1 minute)

**Check health:**
```bash
curl http://localhost:8002/health
```

**Ask a question:**
```bash
curl -X POST http://localhost:8002/api/handoff/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is patient P1-001 heart rate?",
    "patient_id": "P1-001",
    "time_window_hours": 1
  }'
```

**Get patient summary:**
```bash
curl -X POST http://localhost:8002/api/handoff/summary \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P1-001",
    "hours": 2
  }'
```

## 📝 Example Questions

Try these queries (replace P1-001 with any patient ID):

```bash
# Vital sign queries
"What was patient P1-001's heart rate in the last hour?"
"Show me blood pressure trends for P2-005"
"Has patient P3-002's SpO2 been declining?"

# General queries
"What happened with patient P1-001 in the last 2 hours?"
"Any critical events for patient P2-003?"
"Summarize patient P1-001's status"

# Multi-patient queries
"Show me all patients with elevated heart rate"
"Any alerts in the last hour?"
```

## 🔍 View Logs

```bash
# Watch RAG service logs
docker logs -f icu-rag-service

# See indexing stats
curl http://localhost:8002/api/stats
```

## ❓ Troubleshooting

### "GROQ_API_KEY not set"
- Add your API key to `.env` file
- Restart: `docker-compose restart rag-service`

### "No data found"
- Wait 1-2 minutes for data to index
- Check: `curl http://localhost:8002/api/stats`
- Should show `count > 0`

### Kafka connection errors
- Ensure Kafka is running: `docker ps | grep kafka`
- Restart: `docker-compose restart kafka rag-service`

## 🎯 What You Get

- **Real-time indexing** of patient vitals from Kafka
- **Natural language queries** powered by Groq AI
- **24-hour data retention** (configurable)
- **Free tier**: 30 queries/minute
- **~1 second response time**

## 📊 System Status

Check if everything is working:

```bash
# All services status
docker-compose ps

# RAG service stats
curl http://localhost:8002/api/stats

# Expected output:
{
  "indexer": {
    "running": true,
    "topics": ["vitals"],
    "vector_store_count": 150  # Should be > 0 after 1 min
  }
}
```

## 🎓 Next Steps

1. **Try the frontend integration** - Coming soon!
2. **Customize queries** - Modify prompts in `app/rag_chain.py`
3. **Adjust retention** - Change `DATA_RETENTION_HOURS` in docker-compose.yml
4. **Monitor usage** - Check Groq dashboard for API usage

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.
