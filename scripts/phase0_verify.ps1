<#
.SYNOPSIS
    Read-only Phase 0 health check for ScratchLM. Changes nothing.

.DESCRIPTION
    Checks, in order:
      1. Git root of the project folder (must be THIS folder, not C:\Users\mokhl)
      2. Junk folder "CUsersmokhlscratchlm" (created once by a backslash-stripping tool)
      3. .venv python version + torch CUDA
      4. Dataset files in data\raw
    Run from anywhere:  .\scripts\phase0_verify.ps1
#>

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
Write-Host "==> Phase 0 verify, repo root: $RepoRoot" -ForegroundColor Cyan

# --- 1. Git root ---------------------------------------------------------
Write-Host "`n[1] Git root" -ForegroundColor Cyan
$top = (git rev-parse --show-toplevel 2>$null)
if (-not $top) {
    Write-Host "    NO git repo found here or above." -ForegroundColor Yellow
    Write-Host "    Fix (after you approve): git init   (inside $RepoRoot)"
} else {
    $topWin = (Resolve-Path $top).Path
    Write-Host "    git top-level: $topWin"
    if ($topWin -ieq $RepoRoot) {
        Write-Host "    OK: project folder is its own repo root." -ForegroundColor Green
    } else {
        Write-Host "    PROBLEM: git root is a PARENT folder ($topWin)." -ForegroundColor Red
        Write-Host "    git commands here would act on your home repo. Fix: git init   (inside $RepoRoot)"
    }
}

# --- 2. Junk folder ------------------------------------------------------
Write-Host "`n[2] Junk folder" -ForegroundColor Cyan
$junk = Join-Path $RepoRoot "CUsersmokhlscratchlm"
if (Test-Path -LiteralPath $junk) {
    Write-Host "    FOUND: $junk" -ForegroundColor Red
    Get-ChildItem -LiteralPath $junk -Force | Select-Object Name, Mode | Format-Table | Out-String | Write-Host
    Write-Host "    Inspect it, then delete yourself:"
    Write-Host "    Remove-Item -LiteralPath '$junk' -Recurse -Force"
} else {
    Write-Host "    OK: no junk folder." -ForegroundColor Green
}

# --- 3. venv + torch -----------------------------------------------------
Write-Host "`n[3] venv and torch" -ForegroundColor Cyan
$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path $py) {
    & $py -c "import sys, torch; print('    python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
} else {
    Write-Host "    .venv missing: run scripts\setup.ps1" -ForegroundColor Yellow
}

# --- 4. Dataset files ----------------------------------------------------
Write-Host "`n[4] Dataset" -ForegroundColor Cyan
foreach ($name in "train.txt", "validation.txt") {
    $p = Join-Path $RepoRoot "data\raw\$name"
    if (Test-Path $p) {
        "    {0,-16} {1,12:N0} bytes" -f $name, (Get-Item $p).Length | Write-Host
    } else {
        Write-Host "    $name missing: run python scripts\download_data.py" -ForegroundColor Yellow
    }
}
