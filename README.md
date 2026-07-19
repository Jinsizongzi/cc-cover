# cc-cover

`cc-cover` 是一个本地字幕补全工具。启动后由用户输入需要扫描的文件夹，程序递归查找视频文件，只处理同名且字节大小为 `0` 的 `.txt`，使用 **FunASR** 生成中文正文和时间戳，并用 **faster-whisper** 进行第二模型对照与冲突审计。全部校验通过后，程序会直接替换这些空字幕文件。

项目不预设任何扫描目录，也不会自动选择某个课程文件夹。

## 安全原则

- 默认只处理严格的零字节 `.txt`，不会覆盖已有字幕内容。
- 非空 `.txt` 在处理前记录 SHA-256，写回前后都会复核，发现变化立即停止。
- 视频默认记录 SHA-256，运行期间视频或目标文件变化时拒绝写回。
- 输出先生成到独立运行目录，通过格式与质量校验后自动原子替换目标文件。
- 写回前保留目标文件备份；批量写回失败时自动回滚本次已写文件。
- 先处理少量试运行样本并执行质量门禁，再继续其余视频。
- FunASR 是写回正文来源；faster-whisper 仅作为第二候选和审计依据，不会直接拼接两套结果。

## 环境要求

- Windows 10/11
- Python `3.10`、`3.11` 或 `3.12`
- NVIDIA GPU 推荐；也支持 CPU，但大型模型会明显更慢
- 首次使用需要联网下载 Python 依赖和模型

## 首次安装

在项目目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -Device cuda
```

如果没有兼容的 NVIDIA GPU：

```powershell
.\setup.ps1 -Device cpu
```

## 启动程序

安装完成后，双击项目中的 `start.cmd`。

程序会依次执行：

1. 提示输入需要扫描的视频文件夹完整路径。
2. 在用户提供的路径中递归扫描视频及同名字幕文件。
3. 显示待补全的零字节 `.txt` 列表。
4. 使用 FunASR 和 faster-whisper 生成及审计字幕。
5. 按已有字幕格式自动替换零字节 `.txt`。
6. 复核写回结果并显示运行目录。

也可以从 PowerShell 启动，程序同样会提示输入路径：

```powershell
.\run.ps1
```

需要从命令行直接提供一个或多个目录时：

```powershell
.\run.ps1 -Roots "<待扫描目录>"
```

路径中包含空格或中文时，请保留双引号。

## 配置文件

`config.example.json` 只保存模型、运行产物和扫描策略，不保存扫描路径。扫描路径始终由启动时输入或 `-Roots` 参数提供。

```powershell
Copy-Item .\config.example.json .\config.json
.\run.ps1 -Config .\config.json
```

命令行参数优先于配置文件。配置中的相对路径以配置文件所在目录为基准。

默认不会处理“只有空格或换行”的 `.txt`，也不会创建缺失的 `.txt`。如确有需要，可显式启用：

```powershell
.\run.ps1 -IncludeWhitespaceOnly -IncludeMissing
```

## 使用已有环境与模型

可以指定 FFmpeg、模型缓存和计算设备，同时仍由用户提供扫描目录：

```powershell
.\run.ps1 `
  -Ffmpeg "<ffmpeg.exe 路径>" `
  -Device cuda
```

也可设置环境变量 `CC_COVER_FFMPEG`。未指定时，程序优先使用 `imageio-ffmpeg` 自带的 FFmpeg，再尝试系统 `PATH`。

## 格式匹配

程序从用户提供的扫描目录内已有非空同名字幕中检测：

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
- `prepared/*.txt`：通过校验后用于写回的最终字幕
- `backups/*`：写回前目标文件备份
- `commit_report.json` 与 `verification.json`：写回及最终复核结果

运行中断后，使用 `resume` 会复用已完成的模型结果，并在校验通过后直接完成写回：

```powershell
.\.venv\Scripts\cc-cover.exe resume "<运行目录>"
```

## 命令行

```text
cc-cover scan ROOT [ROOT ...]
cc-cover transcribe ROOT [ROOT ...]
cc-cover resume RUN_DIR
cc-cover verify RUN_DIR
```

`transcribe` 和 `resume` 都会在校验通过后直接写回，不需要额外确认参数。

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
