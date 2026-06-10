# Run S02 latency ablation configs.
# Usage (from repo root):
#   powershell -File scripts/run_latency_ablation.ps1

param(
    [string]$Variants = "seq_960,batch_960,batch_640,batch_640_reid"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$names = $Variants -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }

foreach ($name in $names) {
    $cfg = "config/latency_ablation/$name.yaml"
    if (-not (Test-Path $cfg)) {
        Write-Warning "Skip missing config: $cfg"
        continue
    }
    Write-Host "`n========== latency: $name ==========" -ForegroundColor Cyan
    & $py run.py --config $cfg
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $cfg (exit $LASTEXITCODE)"
    }
}

Write-Host "`nAggregating latency summary..." -ForegroundColor Cyan
& $py scripts/aggregate_latency_ablation.py
if ($LASTEXITCODE -ne 0) {
    throw "Aggregation failed (exit $LASTEXITCODE)"
}

Write-Host "`nAll latency runs finished. Results under outputs/latency_ablation/" -ForegroundColor Green
