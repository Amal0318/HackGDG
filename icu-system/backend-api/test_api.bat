@echo off
REM Quick test script for Backend API
REM Tests authentication and basic endpoints

echo ================================================
echo Backend API - Quick Test Script
echo ================================================
echo.

set API_URL=http://localhost:8000

echo [1/4] Testing health endpoint...
curl -s %API_URL%/health
echo.
echo.

echo [2/4] Testing login (doctor credentials)...
curl -s -X POST %API_URL%/api/auth/login ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"doctor\",\"password\":\"doctor123\"}" > token.json
echo.
echo.

echo [3/4] Extracting token...
REM You'll need to manually extract token from token.json
echo See token.json file for authentication token
echo.

echo [4/4] API Documentation available at:
echo   Swagger UI: %API_URL%/docs
echo   ReDoc:      %API_URL%/redoc
echo.

echo ================================================
echo Test complete! Check above for results.
echo ================================================
pause
