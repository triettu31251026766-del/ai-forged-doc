# setup_env.ps1
# Tao moi truong ao (uu tien 3.12, 3.11, 3.13) + cai torch CUDA 12.8 (RTX 5060 / Blackwell)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Khong tim thay lenh 'py'. Hay cai Python tu python.org (co Python Launcher)."
    exit 1
}

$ver = $null
foreach ($v in @("3.12", "3.11", "3.13")) {
    & py "-$v" --version *> $null
    if ($LASTEXITCODE -eq 0) { $ver = $v; break }
}

if (-not $ver) {
    Write-Host "Khong tim thay Python 3.11/3.12/3.13. Kiem tra bang: py -0p"
    exit 1
}

Write-Host "=== Dung Python $ver ==="
if ($ver -eq "3.13") {
    Write-Host "Luu y: neu gap loi wheel torch, hay cai them Python 3.12 roi chay lai."
}

& py "-$ver" -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tao venv that bai."
    exit 1
}

& ".\venv\Scripts\Activate.ps1"

Write-Host "=== Cap nhat pip ==="
python -m pip install --upgrade pip

Write-Host "=== Cai torch CUDA 12.8 ==="
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "=== Cai requirements ==="
python -m pip install -r requirements.txt

Write-Host "=== Kiem tra moi truong ==="
python scripts/check_env.py
