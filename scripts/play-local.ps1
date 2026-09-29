$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$configuration = Join-Path $projectRoot 'config\local\play.yaml'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project Python is missing. Install the rl and play dependencies first.'
}
if (-not (Test-Path -LiteralPath $configuration)) {
    throw 'The local play configuration is missing: config/local/play.yaml.'
}
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:PYTHONUTF8 = '1'
Push-Location $projectRoot
try {
    & $python tools/netplay.py --config-name play_local +local=play
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
