# setup_env.ps1
# Tao moi truong ao Python 3.11/3.12 + cai torch CUDA 12.8 (ho tro RTX 5060 / Blackwell)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$ver = $null
foreach ($v in @("3.12", "3.11", "3.13")) {
    py -$v --version *> $null
    if ($LASTEXITCODE -eq 0) { $ver = $v; break }
}
if (-not $ver) {
    Write-Host "Chua co Python 3.11/3.12/3.13. Hay cai them (https://www.python.org/downloads/) roi chay lai."
    exit 1
}
if ($ver -eq "3.13") {
    Write-Host "Dung Python 3.13 (khuyen nghi cai them 3.12 neu gap loi wheel)."
}
Write-Host "Dung Python $ver"

py -$ver -m venv venv
& ".\venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

python scripts/check_env.py
