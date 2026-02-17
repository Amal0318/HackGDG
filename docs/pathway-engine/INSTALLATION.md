# VitalX Pathway Engine - Installation Guide

Complete guide for setting up Pathway for full integration with the ICU Digital Twin system.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Docker Installation (Recommended)](#docker-installation-recommended)
- [Local Development Installation](#local-development-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Python 3.9+** (Python 3.10+ recommended)
- **Docker Desktop** (for containerized deployment)
- **Windows Build Tools** (for local installation on Windows)

### Windows Build Tools Setup
Pathway requires C++ compilation on Windows. Install one of:

1. **Visual Studio Build Tools** (Recommended):
   ```powershell
   # Download from: https://visualstudio.microsoft.com/downloads/
   # Install "Desktop development with C++" workload
   ```

2. **Or MinGW-w64**:
   ```powershell
   # Download from: https://sourceforge.net/projects/mingw-w64/
   ```

---

## Docker Installation (Recommended)

### 1. Start Docker Desktop

Make sure Docker Desktop is running:
```powershell
# Check if Docker is running
docker --version

# If not started, launch Docker Desktop from Start Menu
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Wait for Docker to be ready (30-60 seconds)
docker ps
```

### 2. Build and Run with Docker Compose

Navigate to the ICU system directory and build:

```powershell
cd C:\Python310\health_ai\HackGDG\icu-system

# Build the pathway-engine container
docker compose build pathway-engine

# Verify the build
docker images | Select-String "pathway-engine"
```

### 3. Start the Full Stack

```powershell
# Start all services
docker compose up -d

# Or start just Kafka and Pathway engine
docker compose up -d kafka pathway-engine

# Check logs
docker compose logs pathway-engine -f
```

### 4. Verify Pathway is Running

```powershell
# Check container status
docker compose ps

# Expected output: pathway-engine should be "running"

# Check Pathway logs for successful startup
docker compose logs pathway-engine | Select-String "started successfully"
```

---

## Local Development Installation

For local development without Docker:

### 1. Run the Setup Script

```powershell
cd C:\Python310\health_ai\HackGDG\icu-system\pathway-engine

# Run the automated setup script
.\setup_local.ps1
```

The script will:
- Create a virtual environment
- Install Pathway and all dependencies
- Set up configuration files
- Verify the installation

### 2. Manual Installation (Alternative)

If the script fails, install manually:

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# This may take 5-10 minutes as Pathway compiles native code
```

### 3. Configure Environment

Create a `.env` file in the pathway-engine directory:

```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:29092
KAFKA_INPUT_TOPIC=vitals
KAFKA_OUTPUT_TOPIC=vitals_enriched
KAFKA_CONSUMER_GROUP=vitalx-pathway-engine-local

# Logging
LOG_LEVEL=DEBUG

# Pathway Settings
PATHWAY_MONITORING_ENABLED=true
PATHWAY_WINDOW_DURATION=30
```

### 4. Start Kafka (Required)

Pathway engine needs Kafka. Start it via Docker Compose:

```powershell
cd C:\Python310\health_ai\HackGDG\icu-system

# Start only Kafka and Zookeeper
docker compose up -d zookeeper kafka

# Wait for Kafka to be ready
docker compose logs kafka | Select-String "started"
```

### 5. Run Pathway Engine Locally

```powershell
cd C:\Python310\health_ai\HackGDG\icu-system\pathway-engine

# Activate virtual environment (if not already active)
.\venv\Scripts\Activate.ps1

# Run the engine
python -m app.main
```

---

## Verification

### Test Installation

Run the test script to verify everything is set up correctly:

```powershell
cd C:\Python310\health_ai\HackGDG\icu-system\pathway-engine

python test_pathway_install.py
```

Expected output:
```
✓ pathway              - Pathway stream processing    [0.8.x]
✓ confluent_kafka      - Kafka client                 [2.3.x]
✓ pydantic             - Data validation              [2.x.x]
...
🎉 All tests passed! Pathway is ready for integration.
```

### Test with Live Data

1. **Start the vital simulator**:
   ```powershell
   docker compose up -d vital-simulator
   ```

2. **Monitor Pathway logs**:
   ```powershell
   docker compose logs pathway-engine -f
   ```

3. **Expected logs**:
   ```
   VitalX Pathway Engine started successfully
   📈 Publishing enriched vital signs stream
   🎯 Real-time risk analysis and anomaly detection active
   ```

4. **Check output topic**:
   ```powershell
   docker exec -it icu-kafka kafka-console-consumer `
     --bootstrap-server localhost:29092 `
     --topic vitals_enriched `
     --from-beginning
   ```

---

## Troubleshooting

### Issue: Pathway installation fails with compilation errors

**Solution**: Install Visual Studio Build Tools
```powershell
# Download and install from:
https://visualstudio.microsoft.com/downloads/

# Select "Desktop development with C++" workload
```

### Issue: Docker command not found

**Solution**: Docker Desktop not in PATH or not running
```powershell
# Start Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Wait 30-60 seconds, then test
docker --version

# If still not found, add to PATH:
$env:Path += ";C:\Program Files\Docker\Docker\resources\bin"
```

### Issue: Kafka connection timeout

**Solution**: Ensure Kafka is running and healthy
```powershell
# Check Kafka status
docker compose ps kafka

# Restart if needed
docker compose restart kafka

# Wait for healthy status
docker compose logs kafka -f
```

### Issue: ModuleNotFoundError: No module named 'pathway'

**Solution**: Virtual environment not activated or installation failed
```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Reinstall Pathway
pip install --upgrade pathway
```

### Issue: Pathway engine starts but no data flows

**Solution**: Check topic configuration
```powershell
# List Kafka topics
docker exec -it icu-kafka kafka-topics `
  --bootstrap-server localhost:29092 --list

# Expected topics: vitals, vitals_enriched

# Check if data is in input topic
docker exec -it icu-kafka kafka-console-consumer `
  --bootstrap-server localhost:29092 `
  --topic vitals --max-messages 5
```

### Issue: Permission denied errors in Docker

**Solution**: Run Docker commands with proper privileges or restart Docker Desktop

```powershell
# Restart Docker Desktop
Stop-Service com.docker.service -Force
Start-Service com.docker.service

# Or restart via GUI
```

---

## Architecture Overview

```
┌─────────────────┐
│ Vital Simulator │ ──> Kafka Topic: vitals
└─────────────────┘
         │
         v
┌─────────────────────────────────────┐
│    VitalX Pathway Engine            │
│  ┌──────────────────────────────┐   │
│  │ 1. Kafka Consumer            │   │
│  │ 2. Data Validation           │   │
│  │ 3. Risk Analysis             │   │
│  │ 4. Anomaly Detection         │   │
│  │ 5. Stream Enrichment         │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         v
    Kafka Topic: vitals_enriched
         │
         v
┌─────────────────┐
│   ML Service    │
└─────────────────┘
```

---

## Next Steps

After successful installation:

1. **Start full integration test**:
   ```powershell
   cd C:\Python310\health_ai\HackGDG\icu-system
   docker compose up
   ```

2. **Monitor all services**:
   ```powershell
   docker compose ps
   docker compose logs -f
   ```

3. **Access the system**:
   - ML Service: http://localhost:8001
   - Backend API: http://localhost:8000
   - Pathway Monitoring: http://localhost:8080

4. **View enriched data**:
   ```powershell
   docker exec -it icu-kafka kafka-console-consumer `
     --bootstrap-server localhost:29092 `
     --topic vitals_enriched `
     --from-beginning `
     --max-messages 10
   ```

---

## Support

For issues or questions:
- Check the logs: `docker compose logs pathway-engine`
- Run the test script: `python test_pathway_install.py`
- Review Pathway documentation: https://pathway.com/developers/
- Check Kafka connectivity: `docker compose logs kafka`

---

**Installation completed!** 🎉

Your Pathway engine is ready for full integration with the ICU Digital Twin system.
