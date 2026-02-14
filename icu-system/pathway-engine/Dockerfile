# VitalX Pathway Engine Production Dockerfile
# Multi-stage build for optimized production deployment

FROM python:3.10-slim AS builder

# Install build dependencies for Pathway (requires Rust compiler)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libstdc++6 \
    curl \
    pkg-config \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for Pathway compilation)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create virtual environment for clean dependency management
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage - minimal runtime image
FROM python:3.10-slim AS production

# Install only necessary runtime dependencies
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user for security (non-root execution)
RUN groupadd -r vitalx && useradd -r -g vitalx vitalx

# Create application directories
RUN mkdir -p /app /app/logs /app/data && \
    chown -R vitalx:vitalx /app

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ /app/app/

# Set ownership of application files
RUN chown -R vitalx:vitalx /app

# Switch to non-root user
USER vitalx

# Environment variables for production
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose monitoring port (Pathway HTTP server)
EXPOSE 8080

# Production command to start VitalX Pathway Engine
CMD ["python", "-m", "app.main"]