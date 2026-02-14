@echo off
REM Cleanup script for ml-service
REM Removes Python cache, temporary files, and large data files

echo ================================================
echo ML-Service Cleanup Script
echo ================================================

echo.
echo [1/4] Removing Python cache files...
if exist models\__pycache__ rd /s /q models\__pycache__
if exist training\__pycache__ rd /s /q training\__pycache__
if exist utils\__pycache__ rd /s /q utils\__pycache__
if exist app\__pycache__ rd /s /q app\__pycache__
del /s /q *.pyc 2>nul
echo Done.

echo.
echo [2/4] Removing large data files...
if exist synthetic_mimic_style_icU.csv del /q synthetic_mimic_style_icU.csv 2>nul
echo Done.

echo.
echo [3/4] Removing empty saved_models folder at root...
if exist saved_models rd /s /q saved_models 2>nul
echo Done.

echo.
echo [4/4] Removing temporary files...
if exist temp rd /s /q temp 2>nul
if exist test_outputs rd /s /q test_outputs 2>nul
del /q *.tmp 2>nul
echo Done.

echo.
echo ================================================
echo Cleanup completed successfully!
echo ================================================
pause
