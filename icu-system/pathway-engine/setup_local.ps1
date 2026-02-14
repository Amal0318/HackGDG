# VitalX Pathway Engine - Local Development Setup Script
# This script sets up Pathway and all dependencies for local development on Windows

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "VitalX Pathway Engine - Local Development Setup" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

if ($pythonVersion -notmatch "Python 3\.(9|10|11|12)") {
    Write-Host "  ERROR: Python 3.9+ is required for Pathway" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".\venv") {
    Write-Host "  Virtual environment already exists, removing..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".\venv"
}

python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host "  Virtual environment created successfully" -ForegroundColor Green

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "  Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to upgrade pip" -ForegroundColor Red
    exit 1
}
Write-Host "  Package managers upgraded successfully" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "Installing Pathway and dependencies..." -ForegroundColor Yellow
Write-Host "  This may take several minutes as Pathway needs to compile..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host "  Note: Pathway requires Visual Studio Build Tools or MinGW on Windows" -ForegroundColor Yellow
    Write-Host "  Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
    exit 1
}
Write-Host "  All dependencies installed successfully" -ForegroundColor Green

# Verify Pathway installation
Write-Host ""
Write-Host "Verifying Pathway installation..." -ForegroundColor Yellow
python -c "import pathway as pw; print(f'Pathway version: {pw.__version__}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Pathway import failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Pathway verified successfully" -ForegroundColor Green

# Create necessary directories
Write-Host ""
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
if (!(Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
    Write-Host "  Created logs directory" -ForegroundColor Green
}

# Create .env file if it doesn't exist
Write-Host ""
Write-Host "Setting up environment configuration..." -ForegroundColor Yellow
if (!(Test-Path ".\.env")) {
    @"
# VitalX Pathway Engine Configuration
# Local Development Environment

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
"@ | Out-File -FilePath ".\.env" -Encoding utf8
    Write-Host "  Created .env configuration file" -ForegroundColor Green
} else {
    Write-Host "  .env file already exists" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Make sure Kafka is running (via Docker Compose)" -ForegroundColor White
Write-Host "     cd ..; docker compose up kafka -d" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Activate the virtual environment (if not already active):" -ForegroundColor White
Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Run the Pathway engine:" -ForegroundColor White
Write-Host "     python -m app.main" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Or use the Docker container:" -ForegroundColor White
Write-Host "     cd ..; docker compose up pathway-engine" -ForegroundColor Gray
Write-Host ""
