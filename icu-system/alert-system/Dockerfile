# Alert System Dockerfile
# LangChain-powered emergency notification service
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Set environment variables (will be overridden by docker-compose)
ENV PYTHONUNBUFFERED=1
ENV LLM_PROVIDER=gemini

# Run the alert system
CMD ["python", "-m", "app.main"]
