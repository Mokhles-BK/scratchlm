<#
.SYNOPSIS
    One-time setup for ScratchLM: find Python 3.11+, create .venv, install the
    CUDA build of PyTorch and the other pinned requirements.

.DESCRIPTION
    Decision log for the PyTorch CUDA index URL:
      - nvidia-smi reports driver CUDA 13.3, but PyTorch does not ship a
        "cu133" build. The closest CUDA build PyTorch publishes is cu130
        (CUDA 13.0). CUDA driver forward-compatibility means a cu130 runtime
        binary runs on a 13.3 driver, so cu130 is the correct and ONLY
        CUDA-capable choice here.
      - https://download.pytorch.org/whl/cu124 exists but has NO cp312
        win_amd64 wheels (PyTorch dropped cp312 for cu124), so cu124 is not
        an option on Python 3.12.
      - PyPI (pypi.org/simple/torch/) has no cu130 builds at all — CUDA wheels
        exist ONLY on download.pytorch.org. That is why we pass --index-url
        for the torch install and fall back to PyPI for everything else.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "==> ScratchLM setup" -ForegroundColor Cyan
Write-Host "    Repo root: $RepoRoot"

# --- 1. Find a Python 3.11+ interpreter -------------------------------
# $PythonExe is the program to run, $PythonArgs are extra arguments for it.
# They are kept SEPARATE on purpose: a string like "py -3.11" is treated by
# PowerShell as one program name containing a space and fails. With the
# call operator we write:  & $PythonExe @PythonArgs -m venv ...
$PythonExe  = $null
$PythonArgs = @()

# Preferred: the py launcher with an explicit version (3.11 first, then 3.12).
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in @("3.11", "3.12", "3.13")) {
        $out = & py "-$v" --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $out -match "Python 3\.\d+") {
            $PythonExe  = "py"
            $PythonArgs = @("-$v")
            Write-Host "    Python: $($out.Trim()) (via py launcher, -$v)" -ForegroundColor Green
            break
        }
    }
}

# Fallback: whatever `python` resolves to, if it is 3.11 or newer.
if (-not $PythonExe) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $out = (& $cmd.Path --version 2>&1 | Out-String).Trim()
        if ($out -match "Python 3\.(\d+)" -and [int]$matches[1] -ge 11) {
            $PythonExe = $cmd.Path
            Write-Host "    Python: $out (fallback: python on PATH)" -ForegroundColor Green
        }
    }
}

if (-not $PythonExe) {
    Write-Error "No Python 3.11+ interpreter found. Install Python 3.11 or newer and re-run."
    exit 1
}

# --- 2. Create the virtual environment ---------------------------------
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "==> Creating .venv" -ForegroundColor Cyan
    & $PythonExe @PythonArgs -m venv (Join-Path $RepoRoot ".venv")
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed"; exit 1 }
} else {
    Write-Host "==> .venv already exists, reusing" -ForegroundColor Yellow
}

# --- 3. Upgrade pip inside the venv ------------------------------------
Write-Host "==> Upgrading pip" -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed"; exit 1 }

# --- 4. Install PyTorch (CUDA build) from the cu130 index ---------------
# See the decision log at the top of this script for why cu130.
Write-Host "==> Installing torch 2.10.0+cu130 from download.pytorch.org/whl/cu130" -ForegroundColor Cyan
& $VenvPython -m pip install `
    --index-url "https://download.pytorch.org/whl/cu130" `
    "torch==2.10.0+cu130"
if ($LASTEXITCODE -ne 0) { Write-Error "torch install failed"; exit 1 }

# --- 5. Install the rest from PyPI --------------------------------------
# Deliberately NOT using --index-url here: the cu130 index serves some
# packages with local-version tags that conflict with plain PyPI pins.
Write-Host "==> Installing remaining requirements from PyPI" -ForegroundColor Cyan
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Error "requirements install failed"; exit 1 }

Write-Host ""
Write-Host "Setup complete. Activate with:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "Then run:  python scripts\gpu_check.py"