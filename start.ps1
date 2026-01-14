# Start the Transformer Seq2Seq Application

Write-Host "🚀 Starting Transformer Seq2Seq Application..." -ForegroundColor Cyan
Write-Host ""

# Check if backend is already running
$backendRunning = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*app.py*" }

if (-not $backendRunning) {
    Write-Host "📡 Starting Backend Server..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python app.py" -WindowStyle Normal
    Start-Sleep -Seconds 3
} else {
    Write-Host "✓ Backend already running" -ForegroundColor Green
}

# Start frontend
Write-Host "⚛️  Starting React Frontend..." -ForegroundColor Yellow
cd "$PSScriptRoot\frontend"
npm run dev

Write-Host ""
Write-Host "✨ Application started!" -ForegroundColor Green
Write-Host "   Backend:  http://localhost:5000" -ForegroundColor Cyan
Write-Host "   Frontend: http://localhost:3000" -ForegroundColor Cyan
