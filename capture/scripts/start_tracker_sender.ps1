[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$LinuxHost,
    [int]$Port = 8766,
    [string]$Python = "python",
    [string]$Token = $env:EGO_TRACKER_TOKEN,
    [double]$Reconnect = 2.0,
    [double]$SendFps = 90.0,
    [Nullable[double]]$Countdown = $null,
    [switch]$NoKeyboard
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sender = Join-Path $projectRoot "tracker_sender_windows.py"

Write-Host "启动 Windows Tracker 发送端 -> ${LinuxHost}:$Port" -ForegroundColor Cyan
$senderArgs = @(
    $sender,
    "--host", $LinuxHost,
    "--port", [string]$Port,
    "--reconnect", $Reconnect.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--send-fps", $SendFps.ToString([Globalization.CultureInfo]::InvariantCulture)
)
if ($Token) {
    $senderArgs += @("--token", $Token)
}
if ($null -ne $Countdown) {
    $senderArgs += @(
        "--countdown",
        $Countdown.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
}
if ($NoKeyboard) {
    $senderArgs += "--no-keyboard"
}
& $Python @senderArgs

if ($LASTEXITCODE -ne 0) {
    throw "Tracker 发送端退出，exit=$LASTEXITCODE"
}
