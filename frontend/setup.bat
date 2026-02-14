@echo off
echo ================================================
echo VitalX Frontend - Setup Script
echo ================================================

echo.
echo Installing dependencies...
call npm install

echo.
echo Creating .env file from example...
if not exist .env (
    copy .env.example .env
    echo .env file created successfully!
) else (
    echo .env file already exists, skipping...
)

echo.
echo ================================================
echo Setup complete!
echo ================================================
echo.
echo To start the development server, run:
echo     npm run dev
echo.
echo Or use the start-frontend.bat script
echo ================================================

pause
