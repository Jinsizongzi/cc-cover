# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 变更

- faster-whisper 默认模型由 `large-v3-turbo` 换成完整版 `large-v3`：当前硬件下转写耗时仍远快于实时，换完整版让双模型交叉校验（质量门禁）更可信。
- 热词列表上限由 120 条提升到 200 条；`hotwords`/`initial_prompt` 两个参数不再互斥回退，当前安装的 faster-whisper 版本支持 `hotwords` 时，`initial_prompt` 也一并传入，不再被跳过。

## [0.7.0] - 2026-08-19

### 变更

- 转写流程由"先跑完一轮 FunASR 再跑一轮 faster-whisper"改为逐候选交替处理两个引擎（一个候选紧接着跑完两个引擎，再处理下一个候选），从根源修复进度条"第 N 个"计数在单引擎整轮跑完前不动的问题，同时减少模型重复加载开销（#109）。
- 候选处理失败（视频指纹校验、音频提取、字幕段校验）不再中止整批：记录后跳过，继续处理剩余候选；写回只覆盖真正处理完并通过质量门禁的那部分候选，CLI 退出码在有候选失败时仍为非零。失败候选清单在完成弹窗与运行摘要（summary.txt）里单独列出，跟质量门禁失败区分开（#104）。

## [0.6.0] - 2026-08-18

### 新增

- 运行环境版本一致性检测：启动时若已装依赖版本落后于当前代码，状态栏提示"有更新可用"（不弹窗、不打断）；点击"安装 / 修复运行环境"只重装真正落后的包，不再无差别全量重装（首次全新安装行为不变）。版本全部匹配但仍怀疑环境本身损坏时，会弹出确认框提供强制完整重装的手动入口（#100、#101、#103）。
- "运行环境"面板新增可选的 HF Token 配置项，用于从 Hugging Face Hub 下载模型时避免未认证请求限流、提高下载速度（#96）。

### 修复

- "运行环境尚未安装"状态提示追加一句提示，告诉用户已有装好的环境可以通过"更改…"按钮直接指过去，不必重新安装（#102）。
- 进度条"约剩余"时间不再随批次内单引擎（funasr 或 faster_whisper）单独跑的那一整轮持续攀升到失真；百分比与可视化进度条改用更细的步数计量，「第 N 个」候选完成计数的严格语义不变（#97）。
- 运行日志剥离 tqdm 进度条的 ANSI 转义残留（此前会显示成类似「[A」「34m」的乱码）（#98）。

## [0.5.0] - 2026-08-18

### 工程

- CLI 到 GUI 的事件契约类型化：pipeline.py/cli.py 逐行输出结构化 JSON 事件（`Event`/`Phase`），替换 GUI 侧原来靠正则从人读文本反推状态的解析方式（#65-#70）。
- 拆分 gui_support.py（1516 行、8 类互不相关职责）为 8 个职责单一的兄弟模块（settings/data_root/commands/storage/progress/candidates/win_native/human_readable），gui_support.py 整体删除（#71-#79，ADR-0001）。
- 拆掉 CCCoverApp 里 5 个 worker（setup_environment/scan_directory/start_transcription/_resume_run_dir/check_environment）各自手写的取消/错误处理样板，统一收编进 `run_in_background()`/`WorkerOutcome`；`_poll_events` 删除旧的字符串 + tuple 分支（#80-#87）。

本次改动均为内部重构，不改变任何用户可见行为。

## [0.4.0] - 2026-08-03

### 新增

- 产品定位调整为「为视频生成同名 TXT」：所有视频默认都是候选，无论同名 TXT 是否存在、是否为空或已有内容，都会用 FunASR 与 faster-whisper 生成和审计字幕，校验通过后覆盖或创建同名 TXT；不再要求目标字幕匹配旧格式，不再提供「包含纯空白 TXT」「创建缺失 TXT」选项（#17、#21）。
- 扩展名白名单扫描与同 stem 冲突检测：多视频指向同一目标 TXT 时标记为冲突，默认不处理、不写回（#21）。
- 输出格式重构：时间戳固定为 `MM:SS` / `H:MM:SS`（小时数不限），UTF-8 无 BOM、CRLF 换行、段间空行、末尾换行（#22）。
- 质量门禁：逐段匹配与冲突审计，按密度、置信度与 `high_risk` 标记告警（#23）。
- 失败与取消体验：失败原因附上下文与运行目录定位，支持停止当前任务与中断后恢复（#24）。
- 数据根可配置与便携模式：默认数据根为程序所在目录，固定包含 `venv\`、`model-cache\`、`runs\`、`temp\` 与 `settings.json`，可一键切换（#33）。
- 单实例锁：同一数据目录只允许一个实例运行，重复启动时提示并尝试切换到已打开窗口（#27）。
- 运行设备合并与自动检测：单一「运行设备」选择（NVIDIA GPU / CPU），启动时经 `nvidia-smi` 自动检测默认值，切换后自动复检（#28）。
- 热词重构：删除内置术语表，改为从用户热词文件与文件名 token 过滤生成（#29）。
- 运行目录清理入口：按时间/状态/大小列出运行记录、勾选删除（不自动删除），总占用超过 5GB 提示建议清理（#30）。
- 每次运行生成 `summary.txt` 人读摘要；完成/失败提示可直达本次运行目录（#30）。
- 开始前确认框、进度条（第 N / 共 M 个 + 百分比 + 已用时长 + 粗估剩余）、完成弹窗与 Windows 提示音（#31）。
- 磁盘预检：安装 / 修复前检查目标盘剩余空间（CPU 约 5.8GB、CUDA 约 8.5GB），不足时明确提示（#32）。
- 安装进度显示：当前组件（如 3/6）、已下载大小与按下载速度粗估的剩余时间（#32）。
- 缓存与数据管理：显示模型缓存占用，提供「打开缓存位置」「清理模型缓存」「清理全部本地数据」入口（#32）。
- 界面截图随项目文档维护，README 展示真实界面（#35）。

### 变更

- 性能：视频全量读取次数由每项约 9 次降到 2 次，以大小 + 修改时间快速校验辅助（#20）。
- 界面排版：功能说明 / 操作指南文字按窗口宽度手动断行，消除默认窗口下自动换行导致的尾部留白（#35 跟进）。

### 修复

- 修复 faster-whisper 尾部零长度幻觉段的处理，并补充格式校验错误上下文（#19）。
- 修复运行环境修复时未强制重装 CUDA torch 的问题。

### 工程

- CI/测试增强：`tests.yml` 增加 Windows runner 与 ubuntu 并行；补齐写回/回滚安全、新格式校验、目标冲突、新扫描语义、哈希策略、faster-whisper 段换算与 GUI 冒烟测试；Windows runner 以 `PYTHONUTF8`/`PYTHONIOENCODING` 保证中文输出（#34）。
- 发布形态：PyInstaller 由 onefile 改为 onedir（`--noupx`、写入版本信息与图标，`assets/app.ico`、`packaging/version_info.txt`），并排除 ASR 运行时栈（torch/funasr 等由数据根 venv 首跑安装，冻结包不含，避免数 GB 膨胀）；新增 Inno Setup 安装器（`packaging/CC-Cover.iss`，默认当前用户安装到 `%LOCALAPPDATA%\Programs\CC-Cover`、可自选目录、卸载时清理数据根）与绿色便携压缩包（解压即用，数据根=exe 目录）；GitHub Actions 在 tag 时构建并发布安装器与便携包两种发布物，并对两者做运行时冒烟（静默安装/卸载、便携 exe 启动存活）后才发布（#36）。

## [0.3.1] - 2026-07-22

### 修复

- 运行选项（设备、哈希等）不再误传给扫描流程，保持扫描契约纯净。

## [0.3.0] - 2026-07-19

### 新增

- 首个交互式 Windows 图形界面版本：选择视频文件夹、扫描候选、双模型识别、格式校验并替换同名 TXT。
- 安全双模型字幕恢复：FunASR 负责中文正文与句级时间戳，faster-whisper 负责对照与冲突审计。
- 通过质量与格式校验后原子写回目标 TXT，写回前备份，失败时回滚。

[Unreleased]: https://github.com/Jinsizongzi/cc-cover/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/Jinsizongzi/cc-cover/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Jinsizongzi/cc-cover/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Jinsizongzi/cc-cover/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Jinsizongzi/cc-cover/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/Jinsizongzi/cc-cover/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Jinsizongzi/cc-cover/releases/tag/v0.3.0
