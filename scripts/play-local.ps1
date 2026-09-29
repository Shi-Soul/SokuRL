param([int]$PlayerSeat, [string]$Opponent)
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
if ($PlayerSeat -eq 0) {
    $PlayerSeat = [int](Read-Host 'Your seat: 1 or 2 (AI uses Marisa at 1P, Reimu at 2P)')
}
if ($PlayerSeat -notin @(1, 2)) { throw 'PlayerSeat must be 1 or 2.' }
if ([string]::IsNullOrWhiteSpace($Opponent)) {
    Write-Host 'Opponents: ppo, rush, zoning, counter, community_combo, community_guard, pressure, footsies, anti_air, air_rush, bullet_wall, graze_hunter, hit_and_run, corner_trap, spirit_siege, skill_cycle'
    $Opponent = Read-Host 'Opponent name'
}
$overrides = @("human.seat=$PlayerSeat")
if ($Opponent -ne 'ppo') {
    if ($Opponent -notmatch '^[a-z_]+$') { throw 'Enter one of the listed opponent names.' }
    $overrides += @('+play_opponent=rule', "play_rule=$Opponent")
}
Push-Location $projectRoot
try {
    & $python tools/netplay.py --config-name play_local +local=play @overrides
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
