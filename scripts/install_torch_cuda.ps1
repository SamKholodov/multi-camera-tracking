# Install GPU PyTorch into the active venv (Windows).
# CUDA wheel channel must match your NVIDIA driver (newer driver = more options).
# See https://pytorch.org/get-started/locally/  (Windows + Pip + нужная CUDA).
#
# Usage (from repo root, after venv is created and activated):
#   .\.venv\Scripts\Activate.ps1
#   .\scripts\install_torch_cuda.ps1
# Or pick an older runtime:
#   .\scripts\install_torch_cuda.ps1 -Cuda cu118

param(
    [ValidateSet("cu118", "cu126", "cu128")]
    [string]$Cuda = "cu126"
)

$ErrorActionPreference = "Stop"
$indexUrl = "https://download.pytorch.org/whl/$Cuda"
Write-Host "Installing torch torchvision torchaudio from $indexUrl ..."
python -m pip install -U pip
python -m pip install torch torchvision torchaudio --index-url $indexUrl
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
