# ML Service Dockerfile - VitalX Sepsis Prediction
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# Install CPU-only PyTorch first (174 MB vs 755 MB CUDA build)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
# Constraint file prevents pip re-resolving to the 755 MB CUDA build
RUN printf 'torch==2.2.0+cpu\ntorchvision==0.17.0+cpu\ntorchaudio==2.2.0+cpu\n' > /constraints.txt
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c /constraints.txt

# Copy application code (includes app/models/ with trained LSTM model)
COPY app/ ./app/

# Run VitalX ML Service with trained LSTM model
CMD ["python", "app/main.py"]