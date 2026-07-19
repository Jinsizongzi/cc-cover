[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("3.10", "3.11", "3.12")]
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"
$Utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8
[Console]::OutputEncoding = $Utf8
$OutputEncoding = $Utf8
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VenvRoot "Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "未找到 Windows Python Launcher（py.exe）。请先安装 Python $PythonVersion。"
}

if (-not (Test-Path -LiteralPath $Python)) {
    & py "-$PythonVersion" -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "创建 Python 虚拟环境失败。"
    }
}

& $Python -m pip install --upgrade pip setuptools wheel
if ($Device -eq "cuda") {
    & $Python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
} else {
    & $Python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) {
    throw "安装 PyTorch 失败。"
}

& $Python -m pip install -e "${ProjectRoot}[asr]"
if ($LASTEXITCODE -ne 0) {
    throw "安装 cc-cover 依赖失败。"
}

& $Python -c "import ctranslate2, funasr, faster_whisper, imageio_ffmpeg, torch; print('cc-cover environment OK'); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('ffmpeg:', imageio_ffmpeg.get_ffmpeg_exe())"
if ($LASTEXITCODE -ne 0) {
    throw "环境自检失败。"
}

Write-Host "安装完成。双击 start.cmd，输入需要扫描的文件夹路径即可开始。"
