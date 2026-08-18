$ErrorActionPreference = "Stop"

$venvPath = Join-Path $env:USERPROFILE ".venvs\stork-rl"
$python = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Run .\setup.ps1 first."
}

& $python -m pip install -e ".[live]"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install live-test dependencies (exit code $LASTEXITCODE)."
}

& $python -m playwright install chromium
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Playwright Chromium (exit code $LASTEXITCODE)."
}

Write-Host "Live browser test setup complete."

