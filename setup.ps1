$ErrorActionPreference = "Stop"

$venvPath = Join-Path $env:USERPROFILE ".venvs\stork-rl"
Write-Host "Creating the virtual environment at $venvPath"

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -m venv $venvPath
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -m venv $venvPath
} else {
    throw "Python 3.10 or newer was not found. Install Python, then run setup.ps1 again."
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create the virtual environment (exit code $LASTEXITCODE)."
}

$python = Join-Path $venvPath "Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update pip (exit code $LASTEXITCODE)."
}

# pygame-ce exposes the same `pygame` module while providing binary wheels for
# Python 3.14. Refuse a source build so a missing wheel fails quickly and clearly.
& $python -m pip install --only-binary pygame-ce -e ".[train,test]"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install project dependencies (exit code $LASTEXITCODE)."
}

& $python -c "import sys, pygame, torch; print(f'Python {sys.version.split()[0]} / pygame-ce {pygame.version.ver} / PyTorch {torch.__version__}')"
if ($LASTEXITCODE -ne 0) {
    throw "Dependency import check failed (exit code $LASTEXITCODE)."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with:"
Write-Host "  & `"$venvPath\Scripts\Activate.ps1`""
