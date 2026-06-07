# Run S02 ReID ablation (fixed YOLOv8m + DeepOcSort): MSMT17 OSNet vs custom checkpoints.
# Usage (from repo root, venv active):
#   .\scripts\run_reid_ablation.ps1
# Optional: .\scripts\run_reid_ablation.ps1 -Models osnet_ibn_msmt17,vehicle_osnet_veri_vric

param(
    [string]$Models = "osnet_ibn_msmt17,vehicle_osnet_veri_vric,vehicle_osnet_view_finetune"
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
    $cfg = "config/reid_ablation/$name.yaml"
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

Write-Host "`nAll runs finished. Results under outputs/reid_ablation/" -ForegroundColor Green
