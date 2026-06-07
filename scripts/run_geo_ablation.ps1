# Geometry gate ablation: vehicle view ReID + YOLOv8m, geometry_max_distance 3 / 5 / 9 m.
# Requires haversine geometry in pipeline (meters, not degrees).
#
# Usage (repo root, venv active):
#   .\scripts\run_geo_ablation.ps1
#   .\scripts\run_geo_ablation.ps1 -Distances 3,5

param(
    [string]$Distances = "3,5,9"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$names = $Distances -split "," | ForEach-Object { "vehicle_view_geo_$($_.Trim())" }

foreach ($name in $names) {
    $cfg = "config/geo_ablation/$name.yaml"
    if (-not (Test-Path $cfg)) {
        Write-Warning "Skip missing config: $cfg"
        continue
    }
    Write-Host "`n========== $name (geometry_max_distance) ==========" -ForegroundColor Cyan
    & $py run.py --config $cfg
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $cfg (exit $LASTEXITCODE)"
    }
}

Write-Host "`nAll geo ablation runs finished. Results under outputs/geo_ablation/" -ForegroundColor Green
