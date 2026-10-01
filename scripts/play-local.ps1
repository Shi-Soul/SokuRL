param([int]$PlayerSeat, [string]$Opponent)
$ErrorActionPreference = 'Stop'
$overrides = @('play.connection=local')
if ($PlayerSeat -ne 0) { $overrides += "play.human.seat=$PlayerSeat" }
if ([string]::IsNullOrWhiteSpace($Opponent)) {
    $overrides += 'operation=menu'
} else {
    $overrides += "opponent=$Opponent"
    if (-not $Opponent.StartsWith('god:')) {
        $overrides += @('track=human', 'play.ai.character=1')
    }
}
& (Join-Path $PSScriptRoot 'play.ps1') @overrides
exit $LASTEXITCODE
