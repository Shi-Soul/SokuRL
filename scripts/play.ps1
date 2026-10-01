param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Overrides)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project Python is missing: .venv/Scripts/python.exe.'
}
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:PYTHONUTF8 = '1'
if ($Overrides.Count -eq 0) { $Overrides = @('operation=menu') }
Push-Location $projectRoot
try {
    & $python tools/play.py @Overrides
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
