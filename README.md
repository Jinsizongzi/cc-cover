# cc-cover

`cc-cover` 是一个本地字幕补全工具。它递归扫描视频文件，只处理同名且字节大小为 `0` 的 `.txt`，使用 **FunASR** 生成中文正文和时间戳，并用 **faster-whisper** 进行第二模型对照与冲突审计，最后按同目录既有字幕格式写回。

## 安全原则

- 默认只处理严格的零字节 `.txt`，不会覆盖已有字幕内容。
- 非空 `.txt` 在处理前记录 SHA-256，写回前后都会复核，发现变化立即停止。
- 视频默认记录 SHA-256，运行期间视频或目标文件变化时拒绝写回。
- 输出先写入独立运行目录，只有指定 `--apply` 才会原子替换目标文件。
- 写回前保留原始目标备份；批量写回失败时自动回滚本次已写文件。
- 先处理少量试运行样本并执行质量门禁，再继续其余视频。
- FunASR 是写回正文来源；faster-whisper 仅作为第二候选和审计依据，不会直接拼接两套结果。

## 环境要求

- Windows 10/11
- Python `3.10`、`3.11` 或 `3.12`
- NVIDIA GPU 推荐；也支持 CPU，但大型模型会明显更慢
- 首次使用需要联网下载 Python 依赖和模型

## 快速开始

在 PowerShell 中进入项目目录：

```powershell
Set-Location E:\cc-cover
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -Device cuda
```

如果没有兼容的 NVIDIA GPU：

```powershell
.\setup.ps1 -Device cpu
```

先扫描，不运行模型也不修改文件：

```powershell
.\.venv\Scripts\cc-cover.exe scan "F:\LLM\深度学习"
```

确认候选列表后，生成、校验并写回全部零字节字幕：

```powershell
.\run.ps1 -Apply
```

`run.ps1` 默认扫描 `F:\LLM\深度学习`。也可传入其他一个或多个目录：

```powershell
.\run.ps1 -Roots "D:\课程一", "D:\课程二" -Apply
```

不加 `-Apply` 时，只生成到 `runs` 目录，不修改原始字幕。检查产物后再写回：

```powershell
.\.venv\Scripts\cc-cover.exe resume ".\runs\20260719_120000_1234" --apply
```

## 使用已有环境与模型

可以直接指定本机 FFmpeg 和模型缓存：

```powershell
.\.venv\Scripts\cc-cover.exe transcribe "F:\LLM\深度学习" `
  --ffmpeg "F:\tools\ffmpeg\bin\ffmpeg.exe" `
  --model-cache "F:\models\cc-cover" `
  --device cuda `
  --apply
```

也可设置环境变量 `CC_COVER_FFMPEG`。未指定时，程序优先使用 `imageio-ffmpeg` 自带的 FFmpeg，再尝试系统 `PATH`。

## 配置文件

复制 `config.example.json` 后修改路径：

```powershell
Copy-Item .\config.example.json .\config.json
.\.venv\Scripts\cc-cover.exe scan --config .\config.json
.\.venv\Scripts\cc-cover.exe transcribe --config .\config.json --apply
```

命令行参数优先于配置文件。配置中的相对路径以配置文件所在目录为基准。

默认不会处理“只有空格或换行”的 `.txt`，也不会创建缺失的 `.txt`。如确有需要，可显式启用：

```powershell
.\run.ps1 -IncludeWhitespaceOnly -IncludeMissing -Apply
```

## 格式匹配

程序从扫描目录内已有的非空同名字幕中检测：

- 文本编码及 BOM
- `CRLF` 或 `LF` 换行
- `MM:SS`、`HH:MM:SS` 或纯文本样式
- 段落间空行和末尾换行
- 是否保留句末标点

优先采用目标视频同目录的主流格式；同目录没有样本时采用扫描根目录的主流格式；完全没有样本时回退为 UTF-8、无 BOM、CRLF、`MM:SS + 文本 + 空行`。

## 运行产物

每次运行会在 `runs/<run_id>` 中保存：

- `manifest.json`：候选快照、配置、阶段状态
- `engines/funasr/*.json`：FunASR 原始结果
- `engines/faster_whisper/*.json`：faster-whisper 对照结果
- `audits/*.json`：逐段匹配与冲突审计
- `prepared/*.txt`：待写回的最终字幕
- `backups/*`：写回前目标文件备份
- `commit_report.json` 与 `verification.json`：写回及最终复核结果

运行中断后，使用 `resume` 会复用已完成的模型结果，不会从头重复处理。

## 常用命令

```powershell
cc-cover scan ROOT [ROOT ...]
cc-cover transcribe ROOT [ROOT ...] [--apply]
cc-cover resume RUN_DIR [--apply]
cc-cover verify RUN_DIR
```

查看完整参数：

```powershell
cc-cover --help
cc-cover transcribe --help
```

## 开发验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
```
