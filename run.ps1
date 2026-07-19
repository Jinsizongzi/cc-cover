[CmdletBinding()]
param(
    [string[]]$Roots,
    [string]$Config = "",
    [string]$Ffmpeg = "",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [switch]$IncludeWhitespaceOnly,
    [switch]$IncludeMissing,
    [switch]$NoHashVideos
)

$ErrorActionPreference = "Stop"
$Utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8
[Console]::OutputEncoding = $Utf8
$OutputEncoding = $Utf8
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "虚拟环境不存在，请先运行 .\setup.ps1。"
}

if (-not $Roots -or $Roots.Count -eq 0) {
    $InputPath = (Read-Host "请输入需要扫描的视频文件夹完整路径").Trim().Trim('"')
    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        throw "扫描路径不能为空。"
    }
    $Roots = @($InputPath)
}

$ResolvedRoots = @()
foreach ($Root in $Roots) {
    $Resolved = Resolve-Path -LiteralPath $Root -ErrorAction Stop
    if (-not (Test-Path -LiteralPath $Resolved.Path -PathType Container)) {
        throw "扫描路径不是文件夹：$Root"
    }
    $ResolvedRoots += $Resolved.Path
}

$Arguments = @("-m", "cc_cover", "transcribe", "--device", $Device)
if ($Config) {
    $Arguments += @("--config", $Config)
}
if ($Ffmpeg) {
    $Arguments += @("--ffmpeg", $Ffmpeg)
}
if ($IncludeWhitespaceOnly) {
    $Arguments += "--include-whitespace-only"
}
if ($IncludeMissing) {
    $Arguments += "--include-missing"
}
if ($NoHashVideos) {
    $Arguments += "--no-hash-videos"
}
$Arguments += $ResolvedRoots

& $Python @Arguments
exit $LASTEXITCODE
