param(
    [string]$Message = "Deploy: pin deps + diagnostics",
    [switch]$SkipLocalInstall
)

$ErrorActionPreference = 'Stop'

Write-Host "Preparing deployment..."

if (-not $SkipLocalInstall) {
    Write-Host "Installing requirements (optional step)"
    if (-not (Test-Path .venv)) {
        Write-Host "Creating virtual environment"
        python -m venv .venv
    }
    . .\.venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
}

Write-Host "Skipping local smoke test"

Write-Host "Committing and pushing"
& git add -A
& git commit -m $Message
& git push

Write-Host "Pushed. Streamlit Cloud will rebuild automatically."
Write-Host "After the build, hard refresh the app (Ctrl+F5)."
