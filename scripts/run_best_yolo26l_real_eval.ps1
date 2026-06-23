# Run best_configs (GTA + CityFlow) with yolo26l real fine-tune, then eval.
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}
Set-Location $PSScriptRoot\..

$py = ".\.venv\Scripts\python.exe"
$log = "outputs\best_configs\yolo26l_real_run.log"
New-Item -ItemType Directory -Force -Path "outputs\best_configs" | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Run-Py($label, [string[]]$Args) {
    Log $label
    & $py @Args 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Run-Py "=== RUN GTA best_configs + yolo26l real ===" @("-u", "run.py", "--config", "best_configs/gta_mcmt_best_yolo26l_real.yaml")

Run-Py "=== EVAL GTA SCT ===" @(
    "-u", "scripts/eval_gta_mcmt.py",
    "--gt-root", "datasets/gta_mcmt",
    "--pred-dir", "outputs/best_configs/gta_mcmt_best_yolo26l_real/per_cam_local",
    "--cameras", "0", "1", "2", "3",
    "--max-iou-dist", "0.7", "--apply-roi", "--align-pred-frames"
)

Run-Py "=== EVAL GTA MCMT ===" @(
    "-u", "scripts/eval_gta_mcmt.py",
    "--gt-root", "datasets/gta_mcmt",
    "--pred-dir", "outputs/best_configs/gta_mcmt_best_yolo26l_real/per_cam",
    "--cameras", "0", "1", "2", "3",
    "--max-iou-dist", "0.7", "--apply-roi", "--align-pred-frames"
)

Run-Py "=== RUN CityFlow best_configs + yolo26l real ===" @("-u", "run.py", "--config", "best_configs/cityflow_mcmt_best_yolo26l_real.yaml")

Run-Py "=== EVAL CityFlow SCT ===" @(
    "-u", "scripts/eval_s02.py",
    "--gt-root", "datasets/AICity22_Track1_MTMC_Tracking/validation/S02",
    "--pred-dir", "outputs/best_configs/cityflow_mcmt_best_yolo26l_real/per_cam_local",
    "--cityflow-protocol"
)

Run-Py "=== EVAL CityFlow MCMT ===" @(
    "-u", "scripts/eval_s02.py",
    "--gt-root", "datasets/AICity22_Track1_MTMC_Tracking/validation/S02",
    "--pred-dir", "outputs/best_configs/cityflow_mcmt_best_yolo26l_real/per_cam",
    "--cityflow-protocol"
)

Log "=== DONE ==="
