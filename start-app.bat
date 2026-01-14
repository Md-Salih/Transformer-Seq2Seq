@echo off
echo ========================================
echo Starting Transformer Summarization App
echo ========================================
echo.

echo [1/2] Starting Backend Server...
cd /d "%~dp0"
start cmd /k ".venv\Scripts\activate && python app.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo [2/2] Starting Frontend Server...
cd frontend
start cmd /k "npm run dev"

echo.
echo ========================================
echo Application Started!
echo ========================================
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3001 (or 3000)
echo.
echo Press any key to exit this window...
pause >nul
