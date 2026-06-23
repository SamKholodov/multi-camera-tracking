# Overnight full GTA ablations: temporal + kinematic + trajectory + geo (~10k frames).
# Usage (repo root):
#   .\scripts\run_overnight_temporal_geo.ps1
# Optional:
#   .\scripts\run_overnight_temporal_geo.ps1 -AblationGroups temporal,kinematic
#   .\scripts\run_overnight_temporal_geo.ps1 -Groups temporal,kinematic,trajectory,geo
#   .\scripts\run_overnight_temporal_geo.ps1 -Force          # re-run even if complete
#   .\scripts\run_overnight_temporal_geo.ps1 -MinFrame 9990  # override complete threshold

param(
    [Alias("Groups")]
    [string[]]$AblationGroups = @("temporal", "kinematic", "trajectory", "geo"),
    [int]$MinFrame = 0,
    [switch]$Force,
    [switch]$SkipEval,
    [double]$MaxIouDist = 0.7
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$gtRoot = Join-Path $root "datasets\gta_mcmt"
$log = Join-Path $root "outputs\configs_gta\overnight_temporal_geo.log"
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
from scripts.eval_gta_mcmt import gta_gt_max_frame
print(gta_gt_max_frame(Path(r'$gtRoot')))
"@
    return [int](& $py -c $code)
}

function Get-MaxPredFrame($predDir) {
    $f = Join-Path $predDir "c000.txt"
    if (-not (Test-Path $f)) { return 0 }
    $mx = 0
    Get-Content $f | ForEach-Object {
        if ($_ -match '^\s*$') { return }
        $frame = [int]($_ -split ',')[0]
        if ($frame -gt $mx) { $mx = $frame }
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
    $keep = @{}
    Get-ChildItem (Join-Path $root $ConfigDir) -Filter "*.yaml" | ForEach-Object {
        $keep[$_.BaseName] = $true
    }
    $outRoot = Join-Path $root "outputs\configs_gta\$OutGroup"
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
    foreach ($rel in $PredRels) {
        $pred = Join-Path $root $rel
        $name = Split-Path (Split-Path $rel -Parent) -Leaf
        $group = Split-Path (Split-Path (Split-Path $rel -Parent) -Parent) -Leaf
        $maxF = Get-MaxPredFrame $pred
        if ($maxF -lt $script:CompleteMinFrame) {
            $bad += "$group/$name (max_frame=$maxF)"
        }
    }
    if ($bad.Count -gt 0) {
        throw "Incomplete runs remain (need max_frame >= $($script:CompleteMinFrame)): $($bad -join ', ')"
    }
}

function Invoke-GtaEval([string]$PredDir) {
    & $py (Join-Path $root "scripts\eval_gta_mcmt.py") `
        --gt-root $gtRoot `
        --pred-dir $PredDir `
        --apply-roi `
        --align-pred-frames `
        --max-iou-dist $MaxIouDist
    if ($LASTEXITCODE -ne 0) {
        throw "Eval failed: $PredDir (exit $LASTEXITCODE)"
    }
}

function Run-Config {
    param([string]$ConfigPath, [string]$PredRel)
    $name = [IO.Path]::GetFileNameWithoutExtension($ConfigPath)
    $group = Split-Path (Split-Path $ConfigPath -Parent) -Leaf
    $pred = Join-Path $root $PredRel
    $maxF = Get-MaxPredFrame $pred
    $isComplete = ($maxF -ge $script:CompleteMinFrame)

    if ($isComplete -and -not $Force) {
        Log "SKIP RUN $group/$name (complete max_frame=$maxF)"
        if (-not $SkipEval) {
            Log "=== EVAL $group/$name ==="
            Invoke-GtaEval -PredDir $pred
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
        Invoke-GtaEval -PredDir $pred
    }
}

$ExpectedFrames = Get-GtMaxFrame
$script:CompleteMinFrame = if ($MinFrame -gt 0) { $MinFrame } else { $ExpectedFrames }
Log "GT max_frame=$ExpectedFrames; complete threshold=$($script:CompleteMinFrame); Force=$($Force.IsPresent)"

Log "Regenerate full-frame ablation configs"
& $py scripts\generate_temporal_ablation_configs.py
if ($LASTEXITCODE -ne 0) { throw "generate_temporal_ablation_configs failed" }
& $py scripts\generate_geo_ablation_configs.py
if ($LASTEXITCODE -ne 0) { throw "generate_geo_ablation_configs failed" }

$selected = @(
    $AblationGroups |
        ForEach-Object { $_ -split "[,\s]+" } |
        ForEach-Object { $_.Trim().ToLower() } |
        Where-Object { $_ }
) | Select-Object -Unique
if ($selected.Count -eq 0) {
    throw "No ablation groups selected. Pass -AblationGroups temporal,kinematic,trajectory,geo"
}
Log "Selected groups: $($selected -join ', ')"
$predRels = @()

if ($selected -contains "temporal") {
    Remove-StaleOutputDirs "configs_gta\temporal_ablation" "temporal_ablation"
    Get-ChildItem (Join-Path $root "configs_gta\temporal_ablation") -Filter "*.yaml" |
        Sort-Object Name | ForEach-Object {
            $rel = "outputs/configs_gta/temporal_ablation/$($_.BaseName)/per_cam"
            $predRels += $rel
            Run-Config -ConfigPath $_.FullName -PredRel $rel
        }
}

if ($selected -contains "kinematic") {
    Remove-StaleOutputDirs "configs_gta\kinematic_ablation" "kinematic_ablation"
    Get-ChildItem (Join-Path $root "configs_gta\kinematic_ablation") -Filter "*.yaml" |
        Sort-Object Name | ForEach-Object {
            $rel = "outputs/configs_gta/kinematic_ablation/$($_.BaseName)/per_cam"
            $predRels += $rel
            Run-Config -ConfigPath $_.FullName -PredRel $rel
        }
}

if ($selected -contains "trajectory") {
    Remove-StaleOutputDirs "configs_gta\trajectory_ablation" "trajectory_ablation"
    Get-ChildItem (Join-Path $root "configs_gta\trajectory_ablation") -Filter "*.yaml" |
        Sort-Object Name | ForEach-Object {
            $rel = "outputs/configs_gta/trajectory_ablation/$($_.BaseName)/per_cam"
            $predRels += $rel
            Run-Config -ConfigPath $_.FullName -PredRel $rel
        }
}

if ($selected -contains "geo") {
    Remove-StaleOutputDirs "configs_gta\geo_ablation" "geo_ablation"
    Get-ChildItem (Join-Path $root "configs_gta\geo_ablation") -Filter "*.yaml" |
        Sort-Object Name | ForEach-Object {
            $rel = "outputs/configs_gta/geo_ablation/$($_.BaseName)/per_cam"
            $predRels += $rel
            Run-Config -ConfigPath $_.FullName -PredRel $rel
        }
}

Test-AllRunsComplete $predRels

if (-not $SkipEval) {
    Log "=== AGGREGATE all ablation groups ==="
    & $py scripts\aggregate_temporal_ablation.py --apply-roi --max-iou-dist $MaxIouDist
    if ($LASTEXITCODE -ne 0) {
        throw "aggregate_temporal_ablation failed (exit $LASTEXITCODE)"
    }
}

Log "Overnight suite finished"
