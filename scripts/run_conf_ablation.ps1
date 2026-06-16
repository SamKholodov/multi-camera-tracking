# Run GTA detector conf_thres ablation (YOLO26l + DeepOcSort + pure ReID).
# Usage (from repo root, venv active):
#   .\scripts\run_conf_ablation.ps1
# Optional:
#   .\scripts\run_conf_ablation.ps1 -Values 0.1,0.2,0.3,0.4,0.5,0.6
#   .\scripts\run_conf_ablation.ps1 -SkipEval
#   .\scripts\run_conf_ablation.ps1 -MaxIouDist 0.7
#
# After reviewing outputs/configs_gta/conf_ablation/summary.csv:
#   python scripts/apply_gta_conf_thres.py --conf 0.3
#   .\scripts\run_gta_ablations.ps1

param(
    [string]$Values = "0.1,0.2,0.3,0.4,0.5,0.6",
    [switch]$SkipEval,
    [double]$MaxIouDist = 0.7,
    [int]$MaxFrame = 2000
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$log = Join-Path $root "outputs\configs_gta\conf_ablation.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

$invariant = [System.Globalization.CultureInfo]::InvariantCulture

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

function ConfTag([double]$conf) {
    return $conf.ToString("0.00", $invariant).Replace(".", "_")
}

Log "Generating conf ablation configs (values: $Values)"
& $py (Join-Path $root "scripts\generate_conf_ablation_configs.py") --values $Values
if ($LASTEXITCODE -ne 0) {
    throw "generate_conf_ablation_configs.py failed (exit $LASTEXITCODE)"
}

Log "Preflight: ensure YOLO26l weights"
& $py (Join-Path $root "scripts\ensure_yolo_weights.py") "models/yolo26l.pt"
if ($LASTEXITCODE -ne 0) {
    throw "ensure_yolo_weights.py failed (exit $LASTEXITCODE)"
}

$confs = $Values -split "," | ForEach-Object {
    [double]::Parse($_.Trim(), $invariant)
} | Where-Object { $_ -ge 0 }
Log "Starting conf ablation ($($confs.Count) runs, max_iou_dist=$MaxIouDist)"

foreach ($conf in $confs) {
    $tag = ConfTag $conf
    $cfg = "configs_gta/conf_ablation/conf_$tag.yaml"
    $pred = "outputs/configs_gta/conf_ablation/conf_$tag/per_cam"
    if (-not (Test-Path $cfg)) {
        throw "Missing config: $cfg"
    }

    Log "=== RUN conf=$conf ==="
    & $py run.py --config $cfg
    if ($LASTEXITCODE -ne 0) {
        throw "Failed: $cfg (exit $LASTEXITCODE)"
    }

    if (-not $SkipEval) {
        Log "=== EVAL conf=$conf ==="
        & $py (Join-Path $root "scripts\eval_gta_mcmt.py") `
            --gt-root (Join-Path $root "datasets\gta_mcmt") `
            --pred-dir (Join-Path $root $pred) `
            --apply-roi `
            --max-iou-dist $MaxIouDist `
            --max-frame $MaxFrame
        if ($LASTEXITCODE -ne 0) {
            throw "Eval failed: $pred (exit $LASTEXITCODE)"
        }
    }
}

if (-not $SkipEval) {
    Log "Aggregating conf ablation metrics"
    & $py (Join-Path $root "scripts\aggregate_conf_ablation.py") --max-iou-dist $MaxIouDist --apply-roi --max-frame $MaxFrame
    if ($LASTEXITCODE -ne 0) {
        throw "aggregate_conf_ablation.py failed (exit $LASTEXITCODE)"
    }
}

Log "Conf ablation finished. See outputs/configs_gta/conf_ablation/summary.csv"
