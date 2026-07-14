# scripts\launch.ps1 — 每部剧单独开窗口，严格串行（前一部 python 进程退出后再开下一个）
#
# 用法（单部）：
#   .\scripts\launch.ps1 "My Fortune Angel Made Me a Queen" -Yes
#   .\scripts\launch.ps1 "My Fortune Angel Made Me a Queen" -TranslateOnly
#
# 用法（多部，串行无人值守）：
#   .\scripts\launch.ps1 "Drama A","Drama B","Drama C" -Yes
#   每部剧在独立窗口中运行；python 进程退出后自动关闭该窗口并启动下一部

param(
    [Parameter(Mandatory)][string[]]$Dramas,
    [switch]$Yes,
    [switch]$TranslateOnly,
    [string]$DeepSeekKey = "",
    [string]$OpenRouterKey = "",
    [string]$Extra = ""
)

$root = Split-Path $PSScriptRoot -Parent

$flags = @()
if ($Yes)           { $flags += "--yes" }
if ($TranslateOnly) { $flags += "--translate-only" }
if ($Extra)         { $flags += $Extra }
$flagStr = if ($flags) { " " + ($flags -join " ") } else { "" }

$keyLines = ""
if ($DeepSeekKey)   { $keyLines += "`n`$env:DEEPSEEK_API_KEY   = '$DeepSeekKey'" }
if ($OpenRouterKey) { $keyLines += "`n`$env:OPENROUTER_API_KEY = '$OpenRouterKey'" }

$total = $Dramas.Count
$i     = 0

foreach ($Drama in $Dramas) {
    $i++
    $DramaEsc = $Drama -replace "'", "''"

    $script = @"
Set-Location '$root'
`$env:PATH += ';D:\Program Files\Git\cmd'$keyLines

Write-Host '[$i/$total] $DramaEsc' -ForegroundColor Cyan
python pipeline.py "$Drama"$flagStr
exit `$LASTEXITCODE
"@

    $tmp = "$env:TEMP\launch_drama_$i.ps1"
    [System.IO.File]::WriteAllText($tmp, $script, [System.Text.Encoding]::UTF8)

    Write-Host "[$i/$total] Launching: $Drama" -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-File", $tmp `
        -WorkingDirectory $root -Wait
    Write-Host "[$i/$total] Done: $Drama" -ForegroundColor Green
}

Write-Host ""
Write-Host "All $total drama(s) complete." -ForegroundColor Yellow
