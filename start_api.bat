@echo off
echo ========================================
echo ICU DETERIORATION PREDICTION API
echo ========================================
echo.

echo Checking required files...

if not exist "best_lstm_model.pth" (
    echo ERROR: Model file not found: best_lstm_model.pth
    pause
    exit /b 1
)

if not exist "feature_scaler.pkl" (
    echo ERROR: Scaler file not found: feature_scaler.pkl
    echo Run: python create_scaler.py
    pause
    exit /b 1
)

echo All files present!
echo.
echo Starting API server...
echo Server will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

D:\Programs\HACKGDG\.venv\Scripts\python.exe main.py
