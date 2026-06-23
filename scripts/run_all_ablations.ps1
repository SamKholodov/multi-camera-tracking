# Run all S02 ablation experiments sequentially (one GPU job at a time).
# Usage (from repo root):
#   .\scripts\run_all_ablations.ps1
# Optional:
#   .\scripts\run_all_ablations.ps1 -SkipBotsort
#   .\scripts\run_all_ablations.ps1 -SkipValidation

param(
    [switch]$SkipBotsort,
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$log = Join-Path $root "outputs\all_ablations.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

function Run-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )
    Log "=== $Label ==="
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $Label (exit $LASTEXITCODE)"
    }
}

if (-not $SkipValidation) {
    Log "Preflight: validate configs and asset paths"
    & $py (Join-Path $root "scripts\validate_ablation_configs.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Preflight failed: fix missing models/datasets (see above), then rerun."
    }
}

if (-not $SkipBotsort) {
    Log "Checking boxmot (required for botsort ablation)"
    & $py -c "import boxmot" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "boxmot not installed. Run: pip install boxmot OR use -SkipBotsort"
    }
}

Log "Starting full ablation suite (log: $log)"

Run-Step -Label "ReID ablation" -Action {
    & (Join-Path $root "scripts\run_reid_ablation.ps1")
}

if ($SkipBotsort) {
    Run-Step -Label "Tracker ablation without botsort" -Action {
        & (Join-Path $root "scripts\run_tracker_ablation.ps1") -Trackers "sort,ocsort,deepocsort"
    }
} else {
    Run-Step -Label "Tracker ablation" -Action {
        & (Join-Path $root "scripts\run_tracker_ablation.ps1")
    }
}

Run-Step -Label "Detector ablation deepocsort" -Action {
    & (Join-Path $root "scripts\run_detector_ablation.ps1")
}

Run-Step -Label "Detector ablation ocsort no ReID" -Action {
    foreach ($m in @("yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolo26m", "yolo26l", "yolo26x", "rtdetr_l")) {
        $cfg = "config/s02_baseline/ocsort/detectors/$m.yml"
        Log "ocsort/$m"
        & $py run.py --config $cfg
        if ($LASTEXITCODE -ne 0) {
            throw "Failed: ocsort/$m (exit $LASTEXITCODE)"
        }
    }
}

Run-Step -Label "Geometry ablation" -Action {
    & (Join-Path $root "scripts\run_geo_ablation.ps1")
}

Run-Step -Label "EMA vs AAF" -Action {
    & $py run.py --config config/ema_vs_aaf/ema.yaml
    if ($LASTEXITCODE -ne 0) { throw "Failed: ema.yaml" }
    & $py run.py --config config/ema_vs_aaf/aaf.yaml
    if ($LASTEXITCODE -ne 0) { throw "Failed: aaf.yaml" }
}

Run-Step -Label "S02 baseline" -Action {
    & $py run.py --config config/s02_baseline.yaml
}

Log "All ablations finished successfully."
