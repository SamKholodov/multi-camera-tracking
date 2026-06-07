# Run all S02 detector ablation configs (DeepOcSort, same tracker params).
# Usage (from repo root, venv active):
#   .\scripts\run_detector_ablation.ps1
# Optional: .\scripts\run_detector_ablation.ps1 -Models yolov8s,yolov8m

param(
    [string]$Models = "yolov8s,yolov8m,yolov8l,yolov8x,yolo26m,yolo26l,yolo26x,rtdetr_l"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$names = $Models -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }

foreach ($name in $names) {
    $cfg = "config/s02_baseline/deepocsort/detectors/$name.yml"
    if (-not (Test-Path $cfg)) {
        Write-Warning "Skip missing config: $cfg"
        continue
    }
    Write-Host "`n========== $name ==========" -ForegroundColor Cyan
    & $py run.py --config $cfg
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $cfg (exit $LASTEXITCODE)"
    }
}

Write-Host "`nAll runs finished. Results under outputs/s02_baseline/deepocsort/detectors/" -ForegroundColor Green
