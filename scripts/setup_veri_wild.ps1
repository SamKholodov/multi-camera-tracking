# Assemble VeRI-Wild from split downloads under wild/ into reid_datasets/VeRI-Wild.
# Output layout matches reid_datasets/VeRI-Wild/README.md.
# Usage: .\scripts\setup_veri_wild.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WildRoot = Join-Path $RepoRoot "wild"
$DestRoot = Join-Path $RepoRoot "reid_datasets\VeRI-Wild"
$ImagesDir = Join-Path $DestRoot "images"
$SplitDir = Join-Path $DestRoot "train_test_split"
$PartsDir = Join-Path $WildRoot "_rar_parts"
$SevenZip = "C:\Program Files\7-Zip\7z.exe"

if (-not (Test-Path $WildRoot)) {
    throw "wild/ folder not found at $WildRoot"
}
if (-not (Test-Path $SevenZip)) {
    throw "7-Zip not found at $SevenZip"
}

New-Item -ItemType Directory -Force -Path $ImagesDir, $SplitDir, $PartsDir | Out-Null

Write-Host "Collecting train_test_split annotations (README filenames)..."
$splitMappings = @(
    @{ Source = "train_list_start0.txt"; Dest = "train_list.txt" }
    @{ Source = "train_list.txt"; Dest = "train_list.txt" }
    @{ Source = "test_3000_id.txt"; Dest = "test_3000.txt" }
    @{ Source = "test_3000.txt"; Dest = "test_3000.txt" }
    @{ Source = "test_3000_id_query.txt"; Dest = "test_3000_query.txt" }
    @{ Source = "test_3000_query.txt"; Dest = "test_3000_query.txt" }
    @{ Source = "test_5000_id.txt"; Dest = "test_5000.txt" }
    @{ Source = "test_5000.txt"; Dest = "test_5000.txt" }
    @{ Source = "test_5000_id_query.txt"; Dest = "test_5000_query.txt" }
    @{ Source = "test_5000_query.txt"; Dest = "test_5000_query.txt" }
    @{ Source = "test_10000_id.txt"; Dest = "test_10000.txt" }
    @{ Source = "test_10000.txt"; Dest = "test_10000.txt" }
    @{ Source = "test_10000_id_query.txt"; Dest = "test_10000_query.txt" }
    @{ Source = "test_10000_query.txt"; Dest = "test_10000_query.txt" }
)

$seenDest = @{}
foreach ($map in $splitMappings) {
    if ($seenDest.ContainsKey($map.Dest)) {
        continue
    }
    $sourceFile = Get-ChildItem -Path $WildRoot -Recurse -File -Filter $map.Source |
        Select-Object -First 1
    if ($null -eq $sourceFile) {
        continue
    }
    Copy-Item -Path $sourceFile.FullName -Destination (Join-Path $SplitDir $map.Dest) -Force
    $seenDest[$map.Dest] = $true
}

$requiredSplits = @(
    "train_list.txt",
    "test_3000.txt",
    "test_3000_query.txt",
    "test_5000.txt",
    "test_5000_query.txt",
    "test_10000.txt",
    "test_10000_query.txt"
)
foreach ($name in $requiredSplits) {
    if (-not (Test-Path (Join-Path $SplitDir $name))) {
        throw "Missing required split file: $name"
    }
}

$readme = Get-ChildItem -Path $WildRoot -Recurse -File -Filter "README.md" |
    Where-Object { $_.DirectoryName -match "VeRI-Wild" } |
    Select-Object -First 1
if ($readme) {
    Copy-Item -Path $readme.FullName -Destination (Join-Path $DestRoot "README.md") -Force
}

Write-Host "Collecting RAR parts into wild/_rar_parts..."
$rarFiles = Get-ChildItem -Path $WildRoot -Recurse -File -Filter "images.part*.rar" |
    Where-Object { $_.FullName -notmatch "\\_rar_parts\\" } |
    Group-Object Name |
    ForEach-Object { $_.Group | Select-Object -First 1 }

if ($rarFiles.Count -eq 0) {
    throw "No images.part*.rar archives found under wild/"
}

foreach ($rar in $rarFiles) {
    Copy-Item -Path $rar.FullName -Destination (Join-Path $PartsDir $rar.Name) -Force
}

$part01 = Join-Path $PartsDir "images.part01.rar"
if (-not (Test-Path $part01)) {
    throw "Missing images.part01.rar in collected parts"
}

$existingImages = @(Get-ChildItem -Path $ImagesDir -Directory -ErrorAction SilentlyContinue).Count
if ($existingImages -gt 1000) {
    Write-Host "images/ already contains $existingImages vehicle folders; skipping extraction."
} else {
    Write-Host "Extracting images to $ImagesDir (this may take a while)..."
    & $SevenZip x $part01 ("-o{0}" -f $ImagesDir) -y
    if ($LASTEXITCODE -ne 0) {
        throw "7-Zip extraction failed with exit code $LASTEXITCODE"
    }

    $nestedImages = Join-Path $ImagesDir "images"
    if (Test-Path $nestedImages) {
        Write-Host "Flattening nested images/images directory..."
        Get-ChildItem -Path $nestedImages -Force | ForEach-Object {
            Move-Item -Path $_.FullName -Destination $ImagesDir -Force
        }
        Remove-Item -Path $nestedImages -Recurse -Force
    }
}

$imageFolders = @(Get-ChildItem -Path $ImagesDir -Directory -ErrorAction SilentlyContinue).Count
$trainList = Join-Path $SplitDir "train_list.txt"
$trainLines = if (Test-Path $trainList) {
    (Get-Content $trainList | Where-Object { $_.Trim() -ne "" }).Count
} else { 0 }

Write-Host ""
Write-Host "VeRI-Wild setup complete (README layout):"
Write-Host "  Destination: $DestRoot"
Write-Host "  Image folders: $imageFolders"
Write-Host "  Train list lines: $trainLines"
Write-Host ""
Write-Host "Add to training config: datasets: veri, vric, veri_wild"
