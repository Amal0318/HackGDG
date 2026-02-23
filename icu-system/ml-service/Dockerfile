# ML Service Dockerfile - VitalX Sepsis Prediction
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (includes app/models/ with trained LSTM model)
COPY app/ ./app/

# Run VitalX ML Service with trained LSTM model
CMD ["python", "app/main.py"]