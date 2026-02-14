# VitalX Pathway Engine - Quick Start Guide

## 🚀 Quick Installation Steps

### Option 1: Docker (Recommended)

1. **Ensure Docker Desktop is running**:
   ```powershell
   # Check if Docker is accessible
   docker --version
   ```

2. **Build the Pathway Engine**:
   ```powershell
   cd C:\Python310\health_ai\HackGDG\icu-system
   docker compose build pathway-engine
   ```

3. **Start the full system**:
   ```powershell
   docker compose up
   ```

4. **Verify Pathway is running**:
   ```powershell
   docker compose ps
   docker compose logs pathway-engine -f
   ```

### Option 2: Local Development

1. **Run the setup script**:
   ```powershell
   cd C:\Python310\health_ai\HackGDG\icu-system\pathway-engine
   .\setup_local.ps1
   ```

2. **Start Kafka via Docker**:
   ```powershell
   cd C:\Python310\health_ai\HackGDG\icu-system
   docker compose up -d kafka zookeeper
   ```

3. **Run Pathway locally**:
   ```powershell
   cd C:\Python310\health_ai\HackGDG\icu-system\pathway-engine
   .\venv\Scripts\Activate.ps1
   python -m app.main
   ```

## 📋 Verification Checklist

- [ ] Docker Desktop is installed and running
- [ ] Pathway engine builds successfully
- [ ] Kafka is running and healthy
- [ ] Pathway engine connects to Kafka
- [ ] Data flows from vital-simulator → pathway-engine → ml-service

## 🛠 Troubleshooting Commands

```powershell
# Check all container status
docker compose ps

# View Pathway logs
docker compose logs pathway-engine -f

# Restart Pathway engine
docker compose restart pathway-engine

# Check Kafka topics
docker exec -it icu-kafka kafka-topics --bootstrap-server localhost:29092 --list

# View enriched data output
docker exec -it icu-kafka kafka-console-consumer \
  --bootstrap-server localhost:29092 \
  --topic vitals_enriched \
  --from-beginning \
  --max-messages 5
```

## 📊 Full Integration Test

```powershell
# 1. Start all services
docker compose up -d

# 2. Check all services are running
docker compose ps

# 3. Monitor Pathway logs
docker compose logs pathway-engine -f

# Expected output:
# ✓ VitalX Pathway Engine started successfully
# ✓ Publishing enriched vital signs stream
# ✓ Real-time risk analysis and anomaly detection active
```

## 📁 Files Created

- `requirements.txt` - Updated with proper Pathway dependencies
- `Dockerfile` - Updated with Rust compiler for Pathway build
- `setup_local.ps1` - Automated local installation script
- `test_pathway_install.py` - Installation verification script
- `INSTALLATION.md` - Comprehensive installation guide
- `QUICKSTART.md` - This file

## 🎯 Next Steps

After successful installation:

1. Access the ML Service API: http://localhost:8001
2. Access the Backend API: http://localhost:8000
3. Monitor Pathway metrics: http://localhost:8080 (if enabled)

---

For detailed troubleshooting, see [INSTALLATION.md](INSTALLATION.md)
