# Backend API Service Dockerfile
FROM python:3.10-slim

# Install build tools needed for pycairo (required by reportlab)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libcairo2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port for FastAPI
EXPOSE 8000

# Run the new streaming backend API
CMD ["uvicorn", "app.main_new:app", "--host", "0.0.0.0", "--port", "8000"]