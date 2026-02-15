@echo off
echo ============================================
echo Building and Starting Frontend Container
echo ============================================
echo.

cd d:\Programs\HackGDG_Final\icu-system

echo [1/4] Stopping existing frontend container...
docker stop icu-frontend 2>nul
docker rm icu-frontend 2>nul

echo.
echo [2/4] Building frontend Docker image...
docker-compose build frontend

echo.
echo [3/4] Starting frontend container...
docker-compose up -d frontend

echo.
echo [4/4] Waiting for frontend to be ready...
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo Frontend is now running!
echo ============================================
echo.
echo Access the dashboard at:
echo   http://localhost:3000
echo.
echo Login credentials:
echo   Admin:  admin / admin123
echo   Doctor: doctor / doctor123
echo   Nurse:  nurse / nurse123
echo.
echo Backend API: http://localhost:8000
echo.
pause
