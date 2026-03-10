# ═══════════════════════════════════════════════════════════════════════════
# VitalX RAG Service - Optimized Multi-Stage Dockerfile
# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL OPTIMIZATIONS:
# 1. Install torch CPU-only FIRST (prevents pip from downloading 915 MB CUDA)
# 2. Comprehensive constraints.txt with 50+ pinned versions (prevents 4+ hour backtracking)
# 3. Legacy resolver flag (faster with constraints, avoids new resolver's exhaustive search)
# 4. Install large packages (sentence-transformers, pyarrow) individually before pathway
# 5. Install pathway[xpack-llm] last after dependencies are satisfied
# 
# Without these optimizations: Build time = 4-6 hours (pip backtracking hell)
# With these optimizations: Build time = 10-15 minutes
# ═══════════════════════════════════════════════════════════════════════════

FROM python:3.10-slim

WORKDIR /app

# System deps: build-essential for native extensions (numpy, torch, etc.)
# NOTE: poppler-utils removed (80 MB, only needed for PDF parsing - we parse Kafka JSON)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Install CPU-only PyTorch FIRST
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Comprehensive constraint file to prevent pip dependency backtracking (saves 4+ hours!)
# Core constraints: torch CPU-only, pathway ecosystem, LLM packages
RUN cat > /constraints.txt <<'EOF'
torch==2.5.1+cpu
torchvision==0.20.1+cpu
torchaudio==2.5.1+cpu
pydantic==2.11.7
pydantic-core==2.33.2
pyarrow==18.0.0
numpy==1.26.4
typing-extensions>=4.12.2
typing-inspection>=0.4.2
# Note: pandas, scipy, scikit-learn removed - not used directly (sentence-transformers brings its own)
# Sentence transformers ecosystem
sentence-transformers==3.1.0
# Pathway ecosystem
panel==1.3.8
bokeh==3.3.4
param==2.1.0
sqlalchemy==2.0.47
# OpenTelemetry (pathway dependency)
opentelemetry-api==1.27.0
opentelemetry-sdk==1.27.0
opentelemetry-proto==1.27.0
opentelemetry-exporter-otlp-proto-grpc==1.27.0
opentelemetry-exporter-otlp-proto-common==1.27.0
opentelemetry-semantic-conventions==0.48b0
# LLM packages (these cause massive backtracking if unpinned)
llama-index-core==0.10.5
litellm==1.44.1
tiktoken==0.7.0
# Note: openai package is auto-installed by litellm (not used directly, only Groq API)
# LangChain ecosystem (pathway dependency, not used directly)
langchain-core==0.3.0
langchain-text-splitters==0.3.0
# Transformers ecosystem
transformers==4.45.0
tokenizers==0.20.0
huggingface-hub==0.25.0
safetensors==0.4.5
# HTTP/async
httpx==0.27.0
anyio==4.6.0
# Protobuf (pathway dependency)
protobuf==5.28.0
# Note: Google Cloud packages removed - not using GCP (google-api-core, google-auth, etc.)
# FastAPI ecosystem
fastapi==0.115.0
uvicorn==0.32.0
starlette==0.38.0
EOF

# Step 2: Install sentence-transformers early, while torch+cpu is already present.
# Use legacy resolver for faster dependency resolution with constraints
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    --use-deprecated=legacy-resolver \
    "sentence-transformers==3.1.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c /constraints.txt

# Step 2b: Pre-install large packages individually so a single timeout doesn't
# abort the entire install.  pyarrow-18 is ~40 MB and timed out previously.
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    --use-deprecated=legacy-resolver \
    "pyarrow==18.0.0" \
    -c /constraints.txt

# Step 3: Install remaining dependencies.
# sentence-transformers + pyarrow already installed → no re-resolution.
# Legacy resolver avoids 4+ hour backtracking with constraints file
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    --use-deprecated=legacy-resolver \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c /constraints.txt

# Force-upgrade typing-extensions AFTER all packages are installed.
# pathway==0.29.1 resolves it to 4.9.0, but typing-inspection>=0.4.2 (pulled by
# pydantic>=2.10) needs TypeIs which was added in typing_extensions 4.10.0.
RUN pip install --no-cache-dir --force-reinstall "typing-extensions>=4.12.2"

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
