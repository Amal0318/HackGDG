FROM python:3.10-slim

WORKDIR /app

# Install system deps needed by sentence-transformers / pathway
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Port 8000  — FastAPI (external, queries + health)
# Port 8666  — Pathway VectorStoreServer (/v1/retrieve, /v1/statistics)
EXPOSE 8000 8666

# Launch FastAPI; Pathway VectorStoreServer starts in a daemon thread inside the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
