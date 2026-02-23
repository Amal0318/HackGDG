# ML Service Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY app/ ./app/
COPY models/ ./models/
COPY app/model.pth ./model.pth
COPY app/scaler.pkl ./scaler.pkl
COPY app/feature_config.json ./feature_config.json

# Expose port for FastAPI
EXPOSE 8000

# Run the new streaming ML service
CMD ["python", "app/main_new.py"]