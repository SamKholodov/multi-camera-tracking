# Run S02 tracker ablation (fixed YOLOv8m): sort, ocsort (no ReID), deepocsort (ReID), botsort.
# Usage (from repo root, venv active):
#   .\scripts\run_tracker_ablation.ps1
# Optional: .\scripts\run_tracker_ablation.ps1 -Trackers sort,ocsort,deepocsort

param(
    [string]$Trackers = "sort,ocsort,deepocsort,botsort"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$names = $Trackers -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }

foreach ($name in $names) {
    $cfg = "config/baseline_trackers/$name.yaml"
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

Write-Host "`nAll runs finished. Results under outputs/baseline_trackers/" -ForegroundColor Green
