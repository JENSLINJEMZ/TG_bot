# Installer bootstrap for Windows (PowerShell).
# Ensures `uv` exists, then hands off to install.py.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 --with-models
#   powershell -ExecutionPolicy Bypass -File install.ps1 --test
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot
Write-Host "[install] detected OS: Windows ($env:PROCESSOR_ARCHITECTURE)"

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        (Join-Path $HOME ".local\bin\uv.exe"),
        (Join-Path $HOME ".cargo\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\uv.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

$uv = Find-Uv
if (-not $uv) {
    Write-Host "[install] uv not found - installing it..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $uv = Find-Uv
}
if (-not $uv) {
    Write-Error "Could not install uv automatically. Get it from https://docs.astral.sh/uv/ and re-run."
    exit 1
}

Write-Host "[install] using uv: $uv"
& $uv run --no-project python install.py @args
exit $LASTEXITCODE