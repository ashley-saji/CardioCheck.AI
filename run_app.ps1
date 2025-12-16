# Heart Disease Prediction Web App Launcher
# Double-click this file or run in PowerShell

Write-Host "Starting Heart Disease Prediction Web App..." -ForegroundColor Green
Write-Host ""

# Change to project directory
Set-Location "C:\Users\ashle\OneDrive\Desktop\web app final"

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Start Streamlit app
Write-Host "Starting Streamlit server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "The app will open in your default browser." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Cyan
Write-Host ""

streamlit run streamlit_app.py

Read-Host "Press Enter to exit"