# Run detector ablation, then tracker ablation (one GPU job at a time).
# Usage: .\scripts\run_ablation_sequential.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$log = Join-Path $root "outputs\ablation_sequential.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

Log "=== Detector ablation (deepocsort + ReID) ==="
& (Join-Path $root "scripts\run_detector_ablation.ps1") 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) {
    throw "Detector ablation failed (exit $LASTEXITCODE)"
}

Log "=== Tracker ablation (sort, ocsort, deepocsort, botsort) ==="
& (Join-Path $root "scripts\run_tracker_ablation.ps1") 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) {
    throw "Tracker ablation failed (exit $LASTEXITCODE)"
}

Log "All ablations finished."
