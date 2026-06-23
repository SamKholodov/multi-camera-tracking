# CityFlow S02 full ablation suite (mirror of configs_gta).
# Detector: models/yolo26l.pt. Eval: CityFlow protocol.
#
# Usage (repo root):
#   .\scripts\run_cityflow_ablations.ps1
# Optional:
#   .\scripts\run_cityflow_ablations.ps1 -AblationGroups baseline,reid,geo
#   .\scripts\run_cityflow_ablations.ps1 -Force -Resync
#   .\scripts\run_cityflow_ablations.ps1 -SkipEval

param(
    [Alias("Groups")]
    [string[]]$AblationGroups = @(
        "baseline", "assoc", "reid", "temporal", "kinematic", "trajectory", "geo",
        "baseline_trackers", "byte", "conf", "ema_vs_aaf", "latency", "zone_tracklet", "sort"
    ),
    [int]$MinFrame = 0,
    [switch]$Force,
    [switch]$Resync,
    [switch]$SkipEval,
    [double]$MaxIouDist = 0.5
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$gtRoot = Join-Path $root "datasets\AICity22_Track1_MTMC_Tracking\validation\S02"
$manifest = Join-Path $gtRoot "sync_manifest.json"
$cameras = @(6, 7, 8, 9)
$log = Join-Path $root "outputs\configs_cityflow\ablations.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

function Get-GtMaxFrame {
    $code = @"
import sys
from pathlib import Path
sys.path.insert(0, r'$root')
from scripts.cityflow_sync_eval import s02_complete_frame_threshold
print(s02_complete_frame_threshold(Path(r'$gtRoot')))
"@
    return [int](& $py -c $code)
}

function Get-MaxPredFrame([string]$PredDir) {
    if ([string]::IsNullOrWhiteSpace($PredDir)) { return 0 }
    $mx = 0
    foreach ($cam in $cameras) {
        $f = Join-Path $PredDir ("c{0:D3}.txt" -f $cam)
        if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { continue }
        try {
            foreach ($line in [System.IO.File]::ReadLines($f)) {
                if ($line -match '^\s*$') { continue }
                $frame = [int]($line -split ',')[0]
                if ($frame -gt $mx) { $mx = $frame }
            }
        }
        catch {
            Log "WARN Get-MaxPredFrame: cannot read $f ($($_.Exception.Message))"
        }
    }
    return $mx
}

function Clear-RunOutput([string]$PredRel) {
    $runDir = Join-Path $root (Split-Path $PredRel -Parent)
    if (Test-Path $runDir) {
        Log "REMOVE output dir: $runDir"
        Remove-Item -Recurse -Force $runDir
    }
}

function Remove-StaleOutputDirs([string]$ConfigDir, [string]$OutGroup) {
    $cfgPath = Join-Path $root $ConfigDir
    $keep = @{}
    if (Test-Path $cfgPath) {
        Get-ChildItem $cfgPath -Filter "*.yaml" -ErrorAction SilentlyContinue | ForEach-Object {
            $keep[$_.BaseName] = $true
        }
    }
    $outRoot = Join-Path $root "outputs\configs_cityflow\$OutGroup"
    if (-not (Test-Path $outRoot)) { return }
    Get-ChildItem $outRoot -Directory | ForEach-Object {
        if (-not $keep.ContainsKey($_.Name)) {
            Log "REMOVE stale output dir (no config): $($_.FullName)"
            Remove-Item -Recurse -Force $_.FullName
        }
    }
}

function Test-AllRunsComplete([string[]]$PredRels) {
    $bad = @()
    foreach ($rel in @($PredRels)) {
        $rel = [string]$rel
        if ([string]::IsNullOrWhiteSpace($rel)) { continue }
        if ($rel -notmatch '/per_cam$') { continue }
        $pred = Join-Path $root $rel
        $maxF = Get-MaxPredFrame $pred
        if ($maxF -lt $script:CompleteMinFrame) {
            $parts = $rel -replace '\\', '/' -split '/'
            $name = $parts[-2]
            $group = $parts[-3]
            $bad += "$group/$name (max_frame=$maxF)"
        }
    }
    if ($bad.Count -gt 0) {
        throw "Incomplete runs remain (need max_frame >= $($script:CompleteMinFrame)): $($bad -join ', ')"
    }
}

function Invoke-CityFlowEval([string]$PredDir) {
    if (-not (Test-Path -LiteralPath $PredDir -PathType Container)) {
        throw "Eval pred dir missing: $PredDir"
    }
    $evalArgs = @(
        (Join-Path $root "scripts\eval_s02.py"),
        "--gt-root", $gtRoot,
        "--pred-dir", $PredDir,
        "--cameras"
    ) + ($cameras | ForEach-Object { [string]$_ }) + @(
        "--cityflow-protocol",
        "--max-iou-dist", [string]$MaxIouDist
    )
    & $py @evalArgs 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Eval failed: $PredDir (exit $LASTEXITCODE)"
    }
}

function Run-Config {
    param([string]$ConfigPath, [string]$PredRel)
    $name = [IO.Path]::GetFileNameWithoutExtension($ConfigPath)
    $parent = Split-Path $ConfigPath -Parent
    $group = Split-Path $parent -Leaf
    if ($group -eq "configs_cityflow") { $group = $name }

    $pred = Join-Path $root $PredRel
    $maxF = Get-MaxPredFrame $pred
    $isComplete = ($maxF -ge $script:CompleteMinFrame)

    if ($isComplete -and -not $Force) {
        Log "SKIP RUN $group/$name (complete max_frame=$maxF)"
        if (-not $SkipEval) {
            Log "=== EVAL $group/$name ==="
            Invoke-CityFlowEval -PredDir $pred
        }
        return
    }

    if ($Force -and (Test-Path (Split-Path $pred -Parent))) {
        Clear-RunOutput $PredRel
        $maxF = 0
    }
    elseif ($maxF -gt 0 -and $maxF -lt $script:CompleteMinFrame) {
        Log "INCOMPLETE $group/$name (max_frame=$maxF < $($script:CompleteMinFrame)); clearing stale outputs"
        Clear-RunOutput $PredRel
        $maxF = 0
    }

    Log "=== RUN $group/$name (prev max_frame=$maxF -> target $($script:CompleteMinFrame)) ==="
    & $py run.py --config $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $ConfigPath (exit $LASTEXITCODE)"
    }

    $maxFAfter = Get-MaxPredFrame $pred
    if ($maxFAfter -lt $script:CompleteMinFrame) {
        Log "WARN $group/$name still incomplete: max_frame=$maxFAfter < $($script:CompleteMinFrame)"
    }
    else {
        Log "OK $group/$name complete: max_frame=$maxFAfter"
    }

    if (-not $SkipEval) {
        Log "=== EVAL $group/$name ==="
        Invoke-CityFlowEval -PredDir $pred
    }
}

function Run-ConfigDir {
    param([string]$ConfigDir, [string]$OutGroup)
    Remove-StaleOutputDirs $ConfigDir $OutGroup
    $dir = Join-Path $root $ConfigDir
    if (-not (Test-Path $dir)) {
        Log "WARN missing config dir: $ConfigDir"
        return @()
    }
    $rels = @()
    Get-ChildItem $dir -Filter "*.yaml" | Sort-Object Name | ForEach-Object {
        $rel = "outputs/configs_cityflow/$OutGroup/$($_.BaseName)/per_cam"
        $rels += $rel
        $null = Run-Config -ConfigPath $_.FullName -PredRel $rel
    }
    return ,$rels
}

if ($Resync -or -not (Test-Path $manifest)) {
    Log "Sync AICity S02 videos"
    $syncArgs = @()
    if ($Force -or $Resync) { $syncArgs += "--force" }
    & $py (Join-Path $root "scripts\sync_aicity_s02_videos.py") @syncArgs
    if ($LASTEXITCODE -ne 0) { throw "sync_aicity_s02_videos failed" }
}
else {
    Log "SKIP sync (manifest exists: $manifest). Use -Resync to rebuild."
}

$ExpectedFrames = Get-GtMaxFrame
$script:CompleteMinFrame = if ($MinFrame -gt 0) { $MinFrame } else { $ExpectedFrames }
Log "Complete threshold=$($script:CompleteMinFrame) frames (sync manifest or GT max); Force=$($Force.IsPresent)"

Log "Regenerate CityFlow configs"
& $py (Join-Path $root "scripts\generate_configs_cityflow.py")
if ($LASTEXITCODE -ne 0) { throw "generate_configs_cityflow failed" }

$selected = @(
    $AblationGroups |
        ForEach-Object { $_ -split "[,\s]+" } |
        ForEach-Object { $_.Trim().ToLower() } |
        Where-Object { $_ }
) | Select-Object -Unique
if ($selected.Count -eq 0) {
    throw "No ablation groups selected."
}
Log "Selected groups: $($selected -join ', ')"
$predRels = @()

if ($selected -contains "baseline") {
    $cfg = Join-Path $root "configs_cityflow\baseline.yaml"
    $rel = "outputs/configs_cityflow/baseline/per_cam"
    $predRels += $rel
    Run-Config -ConfigPath $cfg -PredRel $rel
}

if ($selected -contains "assoc") {
    $predRels += @(Run-ConfigDir "configs_cityflow\assoc_ablation" "assoc_ablation")
}

if ($selected -contains "reid") {
    $predRels += @(Run-ConfigDir "configs_cityflow\reid_ablation" "reid_ablation")
}

if ($selected -contains "temporal") {
    $predRels += @(Run-ConfigDir "configs_cityflow\temporal_ablation" "temporal_ablation")
}

if ($selected -contains "kinematic") {
    $predRels += @(Run-ConfigDir "configs_cityflow\kinematic_ablation" "kinematic_ablation")
}

if ($selected -contains "trajectory") {
    $predRels += @(Run-ConfigDir "configs_cityflow\trajectory_ablation" "trajectory_ablation")
}

if ($selected -contains "geo") {
    $predRels += @(Run-ConfigDir "configs_cityflow\geo_ablation" "geo_ablation")
}

if ($selected -contains "baseline_trackers") {
    $predRels += @(Run-ConfigDir "configs_cityflow\baseline_trackers" "baseline_trackers")
}

if ($selected -contains "byte") {
    $predRels += @(Run-ConfigDir "configs_cityflow\byte_ablation" "byte_ablation")
}

if ($selected -contains "conf") {
    $predRels += @(Run-ConfigDir "configs_cityflow\conf_ablation" "conf_ablation")
}

if ($selected -contains "ema_vs_aaf") {
    $predRels += @(Run-ConfigDir "configs_cityflow\ema_vs_aaf" "ema_vs_aaf")
}

if ($selected -contains "latency") {
    $predRels += @(Run-ConfigDir "configs_cityflow\latency_ablation" "latency_ablation")
}

if ($selected -contains "zone_tracklet") {
    $cfg = Join-Path $root "configs_cityflow\zone_tracklet.yaml"
    $rel = "outputs/configs_cityflow/zone_tracklet/per_cam"
    $predRels += $rel
    Run-Config -ConfigPath $cfg -PredRel $rel
}

if ($selected -contains "sort") {
    $predRels += @(Run-ConfigDir "configs_cityflow\sort" "sort")
}

$predRels = @($predRels)
if ($predRels.Count -gt 0) {
    Test-AllRunsComplete $predRels
}

if (-not $SkipEval) {
    Log "=== AGGREGATE all ablation groups ==="
    & $py (Join-Path $root "scripts\aggregate_cityflow_ablation.py") --cityflow-protocol --max-iou-dist $MaxIouDist
    if ($LASTEXITCODE -ne 0) {
        throw "aggregate_cityflow_ablation failed (exit $LASTEXITCODE)"
    }
}

Log "CityFlow ablation suite finished"
