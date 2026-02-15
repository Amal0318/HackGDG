@echo off
echo ========================================
echo Building Frontend Docker Container
echo ========================================
echo.

cd /d %~dp0

echo Step 1: Stop Vite dev server (if running)
echo Press Ctrl+C in the Vite terminal to stop it first!
echo.
pause

echo Step 2: Building frontend Docker image...
docker-compose build frontend

echo.
echo Step 3: Starting frontend container...
docker-compose up -d frontend

echo.
echo ========================================
echo Frontend Container Started!
echo ========================================
echo.
echo Access the application at:
echo   http://localhost:3000
echo.
echo The frontend is now running in Docker with:
echo   - Nginx web server
echo   - Production-optimized React build
echo   - API proxy to backend (no CORS issues!)
echo.
echo To view logs:
echo   docker logs -f icu-frontend
echo.
echo To stop:
echo   docker-compose stop frontend
echo.
pause
