@echo off
echo ========================================
echo  Backend Container Rebuild Script
echo ========================================
echo.
echo Current backend version: 0.1.0 (OLD)
echo Target version: 1.0.0 (NEW - with auth + APIs)
echo.
echo This will:
echo 1. Stop old backend container
echo 2. Rebuild with updated code
echo 3. Restart all services
echo.
pause

cd /d D:\Programs\HackGDG_Final\icu-system

echo.
echo [1/3] Stopping services...
docker-compose down backend-api

echo.
echo [2/3] Rebuilding backend with new code...
docker-compose build backend-api

echo.
echo [3/3] Starting all services...
docker-compose up -d

echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak

echo.
echo ========================================
echo  Testing Backend API
echo ========================================
curl http://localhost:8000/

echo.
echo.
echo ========================================
echo  Checking Available Endpoints
echo ========================================
curl http://localhost:8000/docs

echo.
echo Backend rebuild complete!
echo Open browser:
echo   - Frontend: http://localhost:3000
echo   - Backend API Docs: http://localhost:8000/docs
echo.
pause
