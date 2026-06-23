# Run temporal / kinematic / trajectory ablations on full GTA (~10k frames).
# Thin wrapper around the overnight suite (no geo group).
# Usage:
#   .\scripts\run_temporal_ablation.ps1
# Optional:
#   .\scripts\run_temporal_ablation.ps1 -Groups temporal,kinematic -SkipEval
#   .\scripts\run_temporal_ablation.ps1 -Force

param(
    [Alias("Groups")]
    [string[]]$AblationGroups = @("temporal", "kinematic", "trajectory"),
    [switch]$SkipEval,
    [switch]$Force,
    [double]$MaxIouDist = 0.7
)

$ErrorActionPreference = "Stop"
$overnight = Join-Path $PSScriptRoot "run_overnight_temporal_geo.ps1"
if (-not (Test-Path $overnight)) {
    throw "Missing $overnight"
}

$args = @(
    "-AblationGroups", ($AblationGroups -join ","),
    "-MaxIouDist", $MaxIouDist
)
if ($SkipEval) { $args += "-SkipEval" }
if ($Force) { $args += "-Force" }

& $overnight @args
if ($LASTEXITCODE -ne 0) {
    throw "run_overnight_temporal_geo failed (exit $LASTEXITCODE)"
}
