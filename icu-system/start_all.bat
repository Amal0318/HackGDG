@echo off
echo ============================================
echo Starting Complete ICU Monitoring System
echo ============================================
echo.

cd d:\Programs\HackGDG_Final\icu-system

echo [1/5] Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [2/5] Stopping all containers...
docker-compose down

echo.
echo [3/5] Building all services...
docker-compose build

echo.
echo [4/5] Starting all services...
docker-compose up -d

echo.
echo [5/5] Waiting for services to be ready...
timeout /t 15 /nobreak >nul

echo.
echo ============================================
echo System Status Check
echo ============================================
docker-compose ps

echo.
echo ============================================
echo ICU Monitoring System is LIVE!
echo ============================================
echo.
echo Services Running:
echo   Frontend:       http://localhost:3000
echo   Backend API:    http://localhost:8000
echo   API Docs:       http://localhost:8000/docs
echo   ML Service:     http://localhost:8001
echo.
echo Login Credentials:
echo   Admin (CMO):    admin / admin123
echo   Doctor:         doctor / doctor123
echo   Nurse:          nurse / nurse123
echo.
echo Monitoring:
echo   - 24 patients across 3 floors (8 per floor)
echo   - Real-time vitals streaming via Kafka
echo   - ML risk predictions
echo   - Email alerts to: sriram.rp08@gmail.com
echo.
echo To stop: docker-compose down
echo To view logs: docker-compose logs -f [service-name]
echo.
pause
