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
$archiveName = "K2450-Electrical-Test-System-Windows-x64"
$builtApp = Join-Path $distRoot $appName
$builtExe = Join-Path $builtApp "$appName.exe"
$builtRuntime = Join-Path $builtApp "runtime"
$releaseStage = Join-Path $releaseRoot $archiveName
$releaseZip = Join-Path $releaseRoot "$archiveName.zip"
$plotSmokeRoot = Join-Path $buildRoot "packaged-plot-smoke"
$iconPath = Join-Path $repoRoot "assets\app_icon.ico"
$originalPath = $env:PATH

function Assert-InBuildRoot([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    $prefix = $buildRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the build directory: $Path"
    }
}

function Remove-BuildPath([string]$Path) {
    Assert-InBuildRoot $Path
    if (Test-Path -LiteralPath $Path -PathType Container) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    elseif (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Force
    }
}

function Compress-ReleaseArchive([string]$Source, [string]$Destination) {
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Remove-BuildPath $Destination
            Compress-Archive -Path (Join-Path $Source "*") `
                -DestinationPath $Destination -CompressionLevel Optimal
            return
        }
        catch {
            if ($attempt -eq 5) { throw }
            Write-Warning (
                "Release files are temporarily locked; retrying archive " +
                "creation ($attempt/5)..."
            )
            Start-Sleep -Seconds 2
        }
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
    if (-not (Test-Path -LiteralPath $iconPath -PathType Leaf)) {
        throw "The application icon was not found: $iconPath"
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
        --icon $iconPath `
        --hidden-import matplotlib.backends.backend_svg `
        --hidden-import matplotlib.backends.backend_pdf `
        --hidden-import matplotlib.backends.backend_agg `
        --add-data "$iconPath;assets" `
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
    $archiveListing = & $Python -m PyInstaller.utils.cliutils.archive_viewer `
        -r -b $builtExe 2>&1
    foreach ($backend in @(
        "matplotlib.backends.backend_svg",
        "matplotlib.backends.backend_pdf",
        "matplotlib.backends.backend_agg"
    )) {
        if (-not ($archiveListing -match [regex]::Escape($backend))) {
            throw "The packaged executable is missing $backend."
        }
    }
    Remove-BuildPath $plotSmokeRoot
    New-Item -ItemType Directory -Path $plotSmokeRoot -Force | Out-Null
    $smokeProcess = Start-Process -FilePath $builtExe `
        -ArgumentList @(
            "--plot-smoke-test",
            ('"{0}"' -f $plotSmokeRoot)
        ) `
        -PassThru -Wait -WindowStyle Hidden
    if ($smokeProcess.ExitCode -ne 0) {
        throw "The packaged SVG/PDF/PNG smoke test failed."
    }
    $smokeOutputs = Get-ChildItem -LiteralPath $plotSmokeRoot -Recurse -File
    foreach ($extension in @(".svg", ".pdf", ".png")) {
        if (-not ($smokeOutputs | Where-Object { $_.Extension -eq $extension })) {
            throw "The packaged plot smoke test did not create $extension."
        }
    }
    Remove-BuildPath $plotSmokeRoot
    $bundledIcu = Get-ChildItem -LiteralPath $builtRuntime -File -Filter "icu*.dll"
    if ($bundledIcu) {
        $names = ($bundledIcu.Name -join ", ")
        throw "Unexpected ICU DLLs were bundled ($names). Check the build PATH before publishing."
    }

    Write-Host "[3/3] Creating the local release archive..."
    Remove-BuildPath $releaseStage
    Remove-BuildPath $releaseZip
    New-Item -ItemType Directory -Path $releaseStage -Force | Out-Null
    Copy-Item -LiteralPath $builtExe -Destination $releaseStage
    Copy-Item -LiteralPath $builtRuntime -Destination $releaseStage -Recurse
    Copy-Item -LiteralPath (Join-Path $repoRoot "configs") -Destination $releaseStage -Recurse
    Copy-Item -LiteralPath (Join-Path $repoRoot "Readme.md") -Destination $releaseStage
    Compress-ReleaseArchive $releaseStage $releaseZip

    $zipSize = [math]::Round((Get-Item -LiteralPath $releaseZip).Length / 1MB, 1)
    foreach ($temporaryPath in @(
        $releaseStage,
        $distRoot,
        $workRoot,
        $specRoot
    )) {
        Remove-BuildPath $temporaryPath
    }
    Write-Host "Done: $releaseZip ($zipSize MB)"
}
finally {
    $env:PATH = $originalPath
    Pop-Location
}
