@echo off
echo.
echo  =====================================================
echo    Smart Vacuum Cleaner - Reinforcement Learning
echo  =====================================================
echo.

cd /d "%~dp0"

echo  [1/2] Starting backend server...
start "Smart Vacuum Backend" cmd /k "uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"

echo  [2/2] Waiting for server to start...
timeout /t 3 /nobreak >nul

echo  Opening frontend in browser...
start http://localhost:8000

echo.
echo  Done! The app is running at http://localhost:8000
echo  Close the backend window to stop the server.
echo.
pause
cd "C:\Users\Preet\OneDrive\Documents\ARL PROJECT\smart-vacuum-rl"
C:/Users/Preet/AppData/Local/Programs/Python/Python313/python.exe -m uvicorn backend.app:app --reload








cd smart-vacuum-rl
uvicorn backend.app:app --reload