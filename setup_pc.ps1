# Windows setup for Fruit Fly Plays Terraria (see docs/PC_SETUP.md).
# Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup_pc.ps1 -Bundle C:\path\to\fruitfly_pc_bundle.zip
# Safe to re-run: every step skips what is already done. -Force overwrites a
# world or character that already exists in tModLoader's save folder.
param(
    [string]$Bundle = "",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot
Set-Location $Repo

function Step($msg) { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }

Step "checking tools"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is missing: install Git for Windows (https://git-scm.com)" }
$py = $null
foreach ($cand in @(@{exe = "py"; args = @("-3.11")}, @{exe = "python"; args = @()})) {
    if (-not (Get-Command $cand.exe -ErrorAction SilentlyContinue)) { continue }
    try { $ver = & $cand.exe @($cand.args) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null } catch { $ver = "" }
    if ($ver -eq "3.11") { $py = $cand; break }
}
if (-not $py) { throw "Python 3.11 not found: install it from python.org (tick 'Add python.exe to PATH')" }
Write-Host "python: $($py.exe) $($py.args)"

Step "python environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $py.exe @($py.args) -m venv .venv
}
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r requirements.txt

Step "fly brain model (about 370 MB: the connectome tables)"
if (-not (Test-Path "brain\Drosophila_brain_model\Connectivity_783.parquet")) {
    git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model brain\Drosophila_brain_model
} else { Write-Host "already there" }

$Docs = [Environment]::GetFolderPath("MyDocuments")
$Tml = Join-Path $Docs "My Games\Terraria\tModLoader"

if ($Bundle) {
    Step "unpacking $Bundle"
    $tmp = Join-Path $env:TEMP ("fruitfly_bundle_" + [guid]::NewGuid())
    Expand-Archive -Path $Bundle -DestinationPath $tmp
    foreach ($kind in @("Worlds", "Players")) {
        $dest = Join-Path $Tml $kind
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Get-ChildItem (Join-Path $tmp $kind) -File -ErrorAction SilentlyContinue | ForEach-Object {
            $target = Join-Path $dest $_.Name
            if ((Test-Path $target) -and -not $Force) { Write-Host "  keeping existing $kind\$($_.Name) (use -Force to overwrite)" }
            else { Copy-Item $_.FullName $target -Force; Write-Host "  $kind\$($_.Name)" }
        }
    }
    Copy-Item (Join-Path $tmp "repo\*") $Repo -Recurse -Force
    Remove-Item $tmp -Recurse -Force
}

Step "linking the mod into tModLoader's ModSources"
$ModSources = Join-Path $Tml "ModSources"
New-Item -ItemType Directory -Force -Path $ModSources | Out-Null
$Link = Join-Path $ModSources "FruitFlyBridge"
if (Test-Path $Link) { Write-Host "already linked: $Link" }
else {
    New-Item -ItemType Junction -Path $Link -Target (Join-Path $Repo "bridge\mod\FruitFlyBridge") | Out-Null
    Write-Host "linked $Link"
}

Step "checks"
$ok = $true
foreach ($f in @("brain\data\flywire_annotations.tsv.gz", "brain\Drosophila_brain_model\Completeness_783.csv")) {
    if (Test-Path $f) { Write-Host "  ok      $f" } else { Write-Host "  MISSING $f" -ForegroundColor Yellow; $ok = $false }
}
& .venv\Scripts\python.exe training\moonlord\test_harness.py | Select-Object -Last 1
& .venv\Scripts\python.exe training\moonlord\test_training.py | Select-Object -Last 1
Write-Host ""
if ($ok) { Write-Host "Done. Next: in tModLoader, Workshop > Develop Mods > FruitFlyBridge > Build + Reload (docs/PC_SETUP.md, step 4)." -ForegroundColor Green }
else { Write-Host "Setup finished with missing files (see above); docs/PC_SETUP.md says where they come from." -ForegroundColor Yellow }
