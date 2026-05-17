# One-shot: create .venv, install CUDA PyTorch, then project requirements.
# Run from repository root in PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\scripts\setup_venv.ps1
# Optional: .\scripts\setup_venv.ps1 -Cuda cu118

param(
    [ValidateSet("cu118", "cu126", "cu128")]
    [string]$Cuda = "cu126"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv in $root ..."
    python -m venv .venv
}

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "Expected $py after venv creation."
}

Write-Host "Upgrading pip ..."
& $py -m pip install -U pip setuptools wheel

$indexUrl = "https://download.pytorch.org/whl/$Cuda"
Write-Host "Installing PyTorch (GPU, $Cuda) ..."
& $py -m pip install torch torchvision torchaudio --index-url $indexUrl

Write-Host "Installing requirements.txt ..."
& $py -m pip install -r requirements.txt

Write-Host "Smoke test ..."
& $py -c @"
import torch
import cv2
import yaml
import scipy.optimize
import ultralytics
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('imports ok')
"@

Write-Host "Done. Activate:  .\.venv\Scripts\Activate.ps1"
