@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0play-local.ps1"
set play_result=%errorlevel%
if not %play_result%==0 pause
exit /b %play_result%
