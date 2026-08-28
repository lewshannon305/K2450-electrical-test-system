param(
    [string]$Python = "python",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildRoot = Join-Path $repoRoot ".build"
$distRoot = Join-Path $buildRoot "dist"
$workRoot = Join-Path $buildRoot "work"
$specRoot = Join-Path $buildRoot "spec"
$releaseRoot = Join-Path $buildRoot "release"
$appName = -join ([char[]](75, 50, 52, 53, 48, 30005, 23398, 27979, 35797, 31995, 32479))
$archiveName = "$appName-Windows-x64"
$builtApp = Join-Path $distRoot $appName
$builtExe = Join-Path $builtApp "$appName.exe"
$builtRuntime = Join-Path $builtApp "runtime"
$releaseStage = Join-Path $releaseRoot $archiveName
$releaseZip = Join-Path $releaseRoot "$archiveName.zip"
$originalPath = $env:PATH

function Assert-InBuildRoot([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    $prefix = $buildRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the build directory: $Path"
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
        & $Python -m unittest discover -s tests -t . -v
        if ($LASTEXITCODE -ne 0) { throw "Tests failed; packaging stopped." }
    }

    Write-Host "[2/3] Building the Windows application..."
    # Developer tools may prepend incompatible native runtimes to PATH.
    $env:PATH = (($originalPath -split ';') | Where-Object {
        $_ -and $_ -notmatch '[\\/]\.cache[\\/]codex-runtimes[\\/]'
    }) -join ';'
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
        main.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
    if (-not (Test-Path -LiteralPath $builtExe -PathType Leaf)) {
        throw "The packaged executable was not found."
    }
    if (-not (Test-Path -LiteralPath $builtRuntime -PathType Container)) {
        throw "The packaged runtime directory was not found."
    }
    $bundledIcu = Get-ChildItem -LiteralPath $builtRuntime -File -Filter "icu*.dll"
    if ($bundledIcu) {
        $names = ($bundledIcu.Name -join ", ")
        throw "Unexpected ICU DLLs were bundled ($names). Check the build PATH before publishing."
    }

    Write-Host "[3/3] Creating the GitHub Release archive..."
    Assert-InBuildRoot $releaseStage
    Assert-InBuildRoot $releaseZip
    if (Test-Path -LiteralPath $releaseStage) {
        Remove-Item -LiteralPath $releaseStage -Recurse -Force
    }
    if (Test-Path -LiteralPath $releaseZip) {
        Remove-Item -LiteralPath $releaseZip -Force
    }
    New-Item -ItemType Directory -Path $releaseStage -Force | Out-Null
    Copy-Item -LiteralPath $builtExe -Destination $releaseStage
    Copy-Item -LiteralPath $builtRuntime -Destination $releaseStage -Recurse
    Copy-Item -LiteralPath (Join-Path $repoRoot "configs") -Destination $releaseStage -Recurse
    Copy-Item -LiteralPath (Join-Path $repoRoot "Readme.md") -Destination $releaseStage
    Compress-Archive -Path (Join-Path $releaseStage "*") -DestinationPath $releaseZip -CompressionLevel Optimal

    $zipSize = [math]::Round((Get-Item -LiteralPath $releaseZip).Length / 1MB, 1)
    Write-Host "Done: $releaseZip ($zipSize MB)"
}
finally {
    $env:PATH = $originalPath
    Pop-Location
}
