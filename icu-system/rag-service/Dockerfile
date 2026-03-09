FROM python:3.10-slim

WORKDIR /app

# System deps: poppler-utils for pdf2image, build-essential for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Install CPU-only PyTorch FIRST
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Constraint file: prevents pip upgrading torch to the 915 MB CUDA build.
# Also pins pydantic/pydantic-core to stop pip backtracking through 20+ versions.
# pathway==0.29.1 requires pydantic~=2.9; pinning the last 2.9.x release here
# lets the resolver pick it immediately instead of scanning down from 2.12.x.
RUN printf 'torch==2.5.1+cpu\ntorchvision==0.20.1+cpu\ntorchaudio==2.5.1+cpu\npydantic==2.11.7\npydantic-core==2.33.2\npyarrow==18.0.0\n' > /constraints.txt

# Step 2: Install sentence-transformers early, while torch+cpu is already present.
RUN pip install --no-cache-dir --timeout 300 --retries 5 "sentence-transformers>=2.2.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c /constraints.txt

# Step 2b: Pre-install large packages individually so a single timeout doesn't
# abort the entire install.  pyarrow-18 is ~40 MB and timed out previously.
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    "pyarrow==18.0.0" \
    -c /constraints.txt

# Step 3: Install remaining dependencies.
# sentence-transformers + pyarrow already installed → no re-resolution.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c /constraints.txt

# Stub out packages that pathway.xpacks.llm.parsers imports at module level
# but that this service never uses (avoids downloading 50+ MB optional deps)
COPY create_stubs.py .
RUN python create_stubs.py

# Copy application code
COPY app/ ./app/

# Port 8000  — FastAPI (external, queries + health)
# Port 8666  — Pathway VectorStoreServer (/v1/retrieve, /v1/statistics)
EXPOSE 8000 8666

# Launch FastAPI; Pathway VectorStoreServer starts in a daemon thread inside the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
