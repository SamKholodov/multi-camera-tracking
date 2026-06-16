# Run GTA MCMT ablation suite (YOLO26l): baseline trackers, ReID, association.
# Run conf ablation first to pick detector.conf_thres:
#   .\scripts\run_conf_ablation.ps1
#   python scripts/apply_gta_conf_thres.py --conf <best>
# Usage (from repo root, venv active):
#   .\scripts\run_gta_ablations.ps1
# Optional:
#   .\scripts\run_gta_ablations.ps1 -SkipBotsort
#   .\scripts\run_gta_ablations.ps1 -SkipEval
#   .\scripts\run_gta_ablations.ps1 -Suites baseline,reid,assoc
#   .\scripts\run_gta_ablations.ps1 -MaxIouDist 0.7

param(
    [switch]$SkipBotsort,
    [switch]$SkipEval,
    [string]$Suites = "baseline,reid,assoc",
    [double]$MaxIouDist = 0.7
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$log = Join-Path $root "outputs\configs_gta\gta_ablations.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

function Assert-PathExists($path, $label) {
    if (-not (Test-Path $path)) {
        throw "Preflight failed: missing $label at $path"
    }
}

function Run-Config {
    param(
        [string]$Label,
        [string]$ConfigPath,
        [string]$PredDir
    )
    Log "=== RUN $Label ==="
    & $py run.py --config $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $ConfigPath (exit $LASTEXITCODE)"
    }
    if (-not $SkipEval) {
        Log "=== EVAL $Label ==="
        & $py (Join-Path $root "scripts\eval_gta_mcmt.py") `
            --gt-root (Join-Path $root "datasets\gta_mcmt") `
            --pred-dir (Join-Path $root $PredDir) `
            --apply-roi `
            --max-iou-dist $MaxIouDist
        if ($LASTEXITCODE -ne 0) {
            throw "Eval failed: $PredDir (exit $LASTEXITCODE)"
        }
    }
}

Log "Preflight: ensure detector weights"
& $py (Join-Path $root "scripts\ensure_yolo_weights.py") "models/yolo26l_fine_tune_gta.pt"
if ($LASTEXITCODE -ne 0) {
    throw "ensure_yolo_weights.py failed (exit $LASTEXITCODE)"
}
Assert-PathExists (Join-Path $root "configs_gta\gta_zone_tracklet.yaml") "zone tracklet graph"
Assert-PathExists (Join-Path $root "models\osnet_ibn_x1_0_msmt17.pt") "MSMT17 ReID"
Assert-PathExists (Join-Path $root "runs\vehicle_reid\osnet_x1_0_veri_vric_wild\epoch_120.pth") "wild epoch_120 ReID"
foreach ($cam in 0..3) {
    Assert-PathExists (Join-Path $root "datasets\gta_mcmt\cam-$cam\calibration.txt") "cam-$cam calibration"
}

if (-not $SkipBotsort) {
    Log "Checking boxmot (required for botsort)"
    & $py -c "import boxmot" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "boxmot not installed. Run: pip install boxmot OR use -SkipBotsort"
    }
}

$selected = $Suites -split "," | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ }
Log "Starting GTA ablations (suites: $($selected -join ', '), max_iou_dist=$MaxIouDist, log: $log)"

if ($selected -contains "baseline") {
    $trackers = @("sort", "ocsort", "deepocsort")
    if (-not $SkipBotsort) {
        $trackers += "botsort"
    }
    foreach ($name in $trackers) {
        $cfg = "configs_gta/baseline_trackers/$name.yaml"
        $pred = "outputs/configs_gta/baseline_trackers/$name/per_cam"
        Run-Config -Label "baseline_trackers/$name" -ConfigPath $cfg -PredDir $pred
    }
}

if ($selected -contains "reid") {
    $models = @(
        "osnet_ibn_msmt17",
        "vehicle_osnet_veri_vric",
        "vehicle_osnet_view_finetune",
        "vehicle_osnet_veri_vric_wild_epoch120"
    )
    foreach ($name in $models) {
        $cfg = "configs_gta/reid_ablation/$name.yaml"
        $pred = "outputs/configs_gta/reid_ablation/$name/per_cam"
        Run-Config -Label "reid_ablation/$name" -ConfigPath $cfg -PredDir $pred
    }
}

if ($selected -contains "assoc") {
    $assocRuns = @(
        @{ Name = "reid_only"; Pred = "outputs/configs_gta/assoc_ablation/reid_only/per_cam" },
        @{ Name = "+zone_tracklet"; Pred = "outputs/configs_gta/assoc_ablation/+zone_tracklet/per_cam" },
        @{ Name = "+geometry_overlap"; Pred = "outputs/configs_gta/assoc_ablation/+geometry_overlap/per_cam" }
    )
    foreach ($run in $assocRuns) {
        $cfg = "configs_gta/assoc_ablation/$($run.Name).yaml"
        Run-Config -Label "assoc_ablation/$($run.Name)" -ConfigPath $cfg -PredDir $run.Pred
    }
}

if (-not $SkipEval) {
    Log "Aggregating metrics"
    & $py (Join-Path $root "scripts\aggregate_gta_ablations.py") --max-iou-dist $MaxIouDist --apply-roi
    if ($LASTEXITCODE -ne 0) {
        throw "aggregate_gta_ablations.py failed (exit $LASTEXITCODE)"
    }
}

Log "All GTA ablations finished."
