# Vital Simulator Service Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose ports
EXPOSE 8000
EXPOSE 5001

# Run the simulator with scenario API
CMD ["python", "app/main.py"]