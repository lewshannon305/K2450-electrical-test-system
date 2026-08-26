param(
    [string]$Python = "python",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$buildRoot = Join-Path $repoRoot ".build"
$distRoot = Join-Path $buildRoot "dist"
$workRoot = Join-Path $buildRoot "work"
$specRoot = Join-Path $buildRoot "spec"
$appName = -join ([char[]](75, 50, 52, 53, 48, 30005, 23398, 27979, 35797, 31995, 32479))
$builtApp = Join-Path $distRoot $appName
$builtExe = Join-Path $builtApp "$appName.exe"
$builtRuntime = Join-Path $builtApp "runtime"
$targetExe = Join-Path $repoRoot "$appName.exe"
$targetRuntime = Join-Path $repoRoot "runtime"

function Assert-DirectChildOfRepo([string]$Path) {
    $resolvedParent = [System.IO.Path]::GetFullPath((Split-Path $Path -Parent))
    if ($resolvedParent -ne $repoRoot) {
        throw "Refusing to modify a path outside the repository root: $Path"
    }
}

Push-Location $repoRoot
try {
    & $Python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller is not installed. Run: $Python -m pip install pyinstaller"
    }

    if (-not $SkipTests) {
        Write-Host "[1/3] Running automated tests..."
        & $Python -m unittest discover -s source/tests -t source -v
        if ($LASTEXITCODE -ne 0) { throw "Tests failed; packaging stopped." }
    }

    Write-Host "[2/3] Building the Windows application..."
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --windowed `
        --name $appName `
        --contents-directory runtime `
        --distpath $distRoot `
        --workpath $workRoot `
        --specpath $specRoot `
        source/main.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
    if (-not (Test-Path -LiteralPath $builtExe -PathType Leaf)) {
        throw "The packaged executable was not found."
    }
    if (-not (Test-Path -LiteralPath $builtRuntime -PathType Container)) {
        throw "The packaged runtime directory was not found."
    }

    Write-Host "[3/3] Updating the runnable files in the repository root..."
    Assert-DirectChildOfRepo $targetExe
    Assert-DirectChildOfRepo $targetRuntime
    if (Test-Path -LiteralPath $targetExe) {
        Remove-Item -LiteralPath $targetExe -Force
    }
    if (Test-Path -LiteralPath $targetRuntime) {
        Remove-Item -LiteralPath $targetRuntime -Recurse -Force
    }
    Copy-Item -LiteralPath $builtExe -Destination $targetExe
    Copy-Item -LiteralPath $builtRuntime -Destination $targetRuntime -Recurse

    $exeSize = [math]::Round((Get-Item -LiteralPath $targetExe).Length / 1MB, 1)
    $runtimeSize = [math]::Round(
        ((Get-ChildItem -LiteralPath $targetRuntime -File -Recurse | Measure-Object Length -Sum).Sum / 1MB),
        1
    )
    Write-Host "Done: $appName.exe ($exeSize MB) + runtime ($runtimeSize MB)"
}
finally {
    Pop-Location
}
