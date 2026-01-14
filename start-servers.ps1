# Start Transformer Seq2Seq Application
# This script starts both backend and frontend servers

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "   TRANSFORMER SEQ2SEQ - APPLICATION STARTUP" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

# Clean up any existing processes
Write-Host "[1/4] Cleaning up existing processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -match 'app.py' -or $_.Path -match 'Transformer-Seq2Seq' } | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process node -ErrorAction SilentlyContinue | Where-Object { $_.Path -match 'Transformer-Seq2Seq' } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Start Backend Server
Write-Host "[2/4] Starting Backend Server (Flask)..." -ForegroundColor Yellow
$backendPath = "D:\SDE\Projects\Transformer-Seq2Seq"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; .venv\Scripts\activate; python app.py" -WindowStyle Normal
Start-Sleep -Seconds 8  # Wait for model to load

# Start Frontend Server  
Write-Host "[3/4] Starting Frontend Server (Vite)..." -ForegroundColor Yellow
$frontendPath = "D:\SDE\Projects\Transformer-Seq2Seq\frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; npm run dev" -WindowStyle Normal
Start-Sleep -Seconds 5  # Wait for Vite to start

# Verify servers are running
Write-Host "[4/4] Verifying servers..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

$backendRunning = $false
$frontendRunning = $false

try {
    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/health" -Method Get -UseBasicParsing -TimeoutSec 3
    if ($response.status -eq "healthy") {
        $backendRunning = $true
    }
} catch {
    Write-Host "   Backend: Not responding yet..." -ForegroundColor Red
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 3
    if ($response.StatusCode -eq 200) {
        $frontendRunning = $true
    }
} catch {
    Write-Host "   Frontend: Not responding yet..." -ForegroundColor Red
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "   SERVER STATUS" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

if ($backendRunning) {
    Write-Host "   ✓ Backend:  http://localhost:5000 (Running)" -ForegroundColor Green
} else {
    Write-Host "   ✗ Backend:  http://localhost:5000 (Starting...)" -ForegroundColor Yellow
}

if ($frontendRunning) {
    Write-Host "   ✓ Frontend: http://localhost:3000 (Running)" -ForegroundColor Green
} else {
    Write-Host "   ✗ Frontend: http://localhost:3000 (Starting...)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "   Open http://localhost:3000 in your browser to use the app!" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""
