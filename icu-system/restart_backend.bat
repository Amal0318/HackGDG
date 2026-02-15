@echo off
echo Restarting backend-api with updated CORS configuration...

cd d:\Programs\HackGDG_Final\icu-system

echo Stopping backend-api...
docker stop icu-backend-api

echo Removing old container...
docker rm icu-backend-api

echo Rebuilding and starting backend-api...
docker-compose up -d --build backend-api

echo.
echo Waiting for backend to be ready...
timeout /t 5 /nobreak > nul

echo.
echo Testing backend API...
curl http://localhost:8000/

echo.
echo Backend API restarted! 
echo CORS should now allow http://localhost:3000
pause
