@echo off
echo ========================================
echo  ML Service Rebuild Script
echo ========================================
echo.
echo This will rebuild the ML service with new debugging logs
echo.
pause

cd /d d:\Programs\HackGDG_Final\icu-system

echo.
echo [1/3] Stopping ML service...
docker compose down ml-service

echo.
echo [2/3] Rebuilding ML service...
docker compose build ml-service

echo.
echo [3/3] Starting ML service...
docker compose up -d ml-service

echo.
echo Waiting for service to be ready...
timeout /t 5 /nobreak

echo.
echo ========================================
echo  Viewing ML Service Logs
echo ========================================
echo Press Ctrl+C to stop viewing logs
echo.
docker compose logs -f ml-service
