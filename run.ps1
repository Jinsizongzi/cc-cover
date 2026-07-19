[CmdletBinding()]
param(
    [string[]]$Roots = @("F:\LLM\深度学习"),
    [string]$Config = "",
    [string]$Ffmpeg = "",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [switch]$Apply,
    [switch]$IncludeWhitespaceOnly,
    [switch]$IncludeMissing,
    [switch]$NoHashVideos
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "虚拟环境不存在，请先运行 .\setup.ps1。"
}

$Arguments = @("-m", "cc_cover", "transcribe", "--device", $Device)
if ($Config) {
    $Arguments += @("--config", $Config)
}
if ($Ffmpeg) {
    $Arguments += @("--ffmpeg", $Ffmpeg)
}
if ($Apply) {
    $Arguments += "--apply"
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
if (-not $Config) {
    $Arguments += $Roots
}

& $Python @Arguments
exit $LASTEXITCODE
