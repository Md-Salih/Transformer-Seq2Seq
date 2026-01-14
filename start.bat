@echo off
echo ========================================
echo Starting Transformer Seq2Seq
echo ========================================
echo.

echo [1/2] Starting Backend Server...
start "Backend Server" cmd /k "cd /d %~dp0 && python app.py"
timeout /t 3 /nobreak >nul

echo [2/2] Starting Frontend Server...
cd frontend
start "Frontend Server" cmd /k "npm run dev"

echo.
echo ========================================
echo SERVERS STARTED!
echo ========================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to open browser...
pause >nul
start http://localhost:3000
