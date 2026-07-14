# scripts\launch.ps1 — 在新 PowerShell 窗口中运行流水线
#
# 用法：
#   .\scripts\launch.ps1 "My Fortune Angel Made Me a Queen"
#   .\scripts\launch.ps1 "My Fortune Angel Made Me a Queen" -Yes
#   .\scripts\launch.ps1 "My Fortune Angel Made Me a Queen" -TranslateOnly
#   .\scripts\launch.ps1 "My Fortune Angel Made Me a Queen" -Extra "--skip-preflight"

param(
    [Parameter(Mandatory)][string]$Drama,
    [switch]$Yes,
    [switch]$TranslateOnly,
    [string]$DeepSeekKey = "",
    [string]$OpenRouterKey = "",
    [string]$Extra = ""
)

$root      = Split-Path $PSScriptRoot -Parent
$flags     = @()
if ($Yes)           { $flags += "--yes" }
if ($TranslateOnly) { $flags += "--translate-only" }
if ($Extra)         { $flags += $Extra }
$flagStr   = if ($flags) { " " + ($flags -join " ") } else { "" }

# Escape single quotes for use inside single-quoted strings in the generated script
$DramaEsc  = $Drama -replace "'", "''"

$keyLines  = ""
if ($DeepSeekKey)   { $keyLines += "`n`$env:DEEPSEEK_API_KEY   = '$DeepSeekKey'" }
if ($OpenRouterKey) { $keyLines += "`n`$env:OPENROUTER_API_KEY = '$OpenRouterKey'" }

$script = @"
Set-Location '$root'
`$env:PATH += ';D:\Program Files\Git\cmd'$keyLines

Write-Host '>>> $DramaEsc' -ForegroundColor Cyan
python pipeline.py "$Drama"$flagStr

Write-Host ''
Write-Host 'Pipeline finished. Press Enter to close.' -ForegroundColor Green
Read-Host
"@

$tmp = "$env:TEMP\launch_drama.ps1"
[System.IO.File]::WriteAllText($tmp, $script, [System.Text.Encoding]::UTF8)
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $tmp -WorkingDirectory $root
Write-Host "Launched: $Drama$flagStr"
