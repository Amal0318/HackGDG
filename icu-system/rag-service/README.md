# RAG Service - AI Patient Handoff Assistant

## Overview

The RAG (Retrieval-Augmented Generation) service provides an AI-powered assistant for nurse shift handoffs. It consumes live patient data from Kafka streams and allows nurses to ask natural language questions about patient status.

## Features

- **Live Data Indexing**: Automatically indexes patient vitals and events from Kafka in real-time
- **Natural Language Queries**: Ask questions like "What was patient P1-001's heart rate in the last hour?"
- **Patient Summaries**: Generate comprehensive shift handoff summaries
- **Trend Analysis**: Track vital sign trends over time
- **Vector Search**: Uses ChromaDB for efficient semantic search

## Technology Stack

- **FastAPI**: REST API framework
- **Kafka**: Live data streaming
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Text embeddings (all-MiniLM-L6-v2)
- **LangChain**: RAG orchestration
- **Groq**: LLM provider (free tier available)

## Setup

### 1. Get Groq API Key (Free)

1. Visit [https://console.groq.com/](https://console.groq.com/)
2. Sign up for a free account
3. Generate an API key
4. Copy the API key

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### 3. Start the Service

The RAG service will automatically start with docker-compose:

```bash
docker-compose up -d --build
```

## API Endpoints

### Health Check

```bash
GET http://localhost:8002/health
```

### Query Patient Data

```bash
POST http://localhost:8002/api/handoff/query

{
  "question": "What was patient P1-001's heart rate in the last hour?",
  "patient_id": "P1-001",
  "time_window_hours": 1
}
```

### Generate Patient Summary

```bash
POST http://localhost:8002/api/handoff/summary

{
  "patient_id": "P1-001",
  "hours": 4
}
```

### Get Vital Trend

```bash
POST http://localhost:8002/api/handoff/trend

{
  "patient_id": "P1-001",
  "vital_name": "heart rate",
  "hours": 2
}
```

### Get Statistics

```bash
GET http://localhost:8002/api/stats
```

## Example Queries

- "What was the blood pressure for patient P1-001 in the last 2 hours?"
- "Show me all alerts on Floor 2"
- "What are the vital trends for bed 23?"
- "Has patient P2-005's SpO2 been declining?"
- "Any high-risk patients in the last hour?"

## How It Works

1. **Data Collection**: Kafka consumer subscribes to `vitals` topic from vital-simulator
2. **Indexing**: Incoming vital signs data is converted to text and embedded using Sentence Transformers
3. **Storage**: Embeddings stored in ChromaDB with metadata (patient_id, timestamp, etc.)
4. **Query Processing**:
   - User asks a question
   - Question is embedded
   - Similar documents retrieved from ChromaDB
   - Context + question sent to Groq LLM
   - LLM generates human-readable answer

## Configuration

Environment variables:

- `KAFKA_BOOTSTRAP_SERVERS`: Kafka broker address (default: kafka:9092)
- `KAFKA_TOPICS`: Topics to consume (default: vitals)
- `GROQ_API_KEY`: Your Groq API key
- `GROQ_MODEL`: LLM model (default: llama3-70b-8192)
- `DATA_RETENTION_HOURS`: How long to keep data (default: 24)

## Ports

- **8002**: RAG service API

## Volumes

- `rag-data`: Persistent storage for ChromaDB vector database

## Troubleshooting

### No API Key Warning

If you see "GROQ_API_KEY not set", the service will run but return raw context instead of LLM-generated answers.

**Solution**: Add your API key to `.env` file and restart:

```bash
docker-compose restart rag-service
```

### Kafka Connection Issues

Check if Kafka is running:

```bash
docker ps | grep kafka
```

View RAG service logs:

```bash
docker logs icu-rag-service
```

### No Data Found

The system needs time to index data. Wait 1-2 minutes after startup, then check:

```bash
curl http://localhost:8002/api/stats
```

Should show `vector_store.count > 0`

## Performance

- **Indexing Rate**: ~100-500 messages/second
- **Query Latency**: 
  - Vector search: ~50ms
  - With LLM: ~500-2000ms (depending on Groq load)
- **Storage**: ~1KB per indexed document

## Free Tier Limits

**Groq Free Tier:**
- 30 requests/minute
- 14,400 requests/day
- Sufficient for typical ICU handoff queries

## Development

To run locally without Docker:

```bash
cd icu-system/rag-service
pip install -r requirements.txt
export KAFKA_BOOTSTRAP_SERVERS=localhost:29092
export GROQ_API_KEY=your_key
python -m app.main
```

## Future Enhancements

- [ ] Chat history tracking
- [ ] Multi-turn conversations
- [ ] Voice input support
- [ ] Auto-generated shift reports
- [ ] Integration with frontend chat UI
