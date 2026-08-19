# cc-cover

Windows 桌面应用（Tkinter GUI + CLI），用 FunASR 与 faster-whisper 给视频文件补全缺失的字幕 TXT。这份术语表统一项目里反复出现、容易叫岔的概念。

## Language

**Candidate**：
一个待补全字幕的视频——同名 TXT 缺失或为空，discovery 阶段识别出来、pipeline 阶段处理它。
_Avoid_: Sample（`pipeline.py` 目前大量用这个词指同一个东西，是历史遗留，未来应统一成 Candidate）

**Run**：
一次流水线执行，处理一批 Candidate，产物落在一个带 `manifest.json` 的运行目录下。
_Avoid_: Task

**Options**：
一次 Run 的配置——设备、模型、语言、pilot 数量等。当前代码里有三种独立类型表达同一个概念（`PipelineOptions`/`GuiOptions`/`GuiSettings`），未来应该收敛成一个。
_Avoid_: Config、Settings 单独指代同一件事（这两个词目前分别绑定在配置文件和 `GuiSettings` 上，容易和 Options 混）

**Protected Text**：
不是任何视频写回目标、但内容非空的 TXT 文件——处理过程中必须保持不变，写回前后要校验它没被动过。
_Avoid_: protected_nonempty_txt、protected_texts（代码里两种字段名并存，界面上还有第三种中文说法，指同一件事）

**Stage**：
质量门禁通过后、写回课程目录之前的中间状态——字幕先写到 `prepared/` 暂存目录等待 commit。是一个动词（`stage()` 方法），也是 manifest 的生命周期状态前缀（`staged_all`/`staged_partial`）。
_Avoid_: 用 Stage 表达"流水线处理到哪一步失败"——那是 Phase，两者刻意分开，见下。

**Phase**：
标注一次失败发生在流水线处理的哪个环节，挂在 `PipelineError`/`EngineError` 上的封闭枚举：setup（run 创建、设备/FFmpeg 解析、热词文件、resume 读取 manifest 等预检）、audio_extract（音频提取）、fingerprint（候选转写前后的完整性校验——视频在处理期间被改动/替换）、funasr（FunASR 转写）、faster_whisper（faster-whisper 转写）、quality_gate（质量门禁）、writeback（写回）、verify（写回后的最终复核）。
不含 scan——扫描阶段的错误是 `DiscoveryError`，不是 `PipelineError`/`EngineError`，不挂 Phase。
_Avoid_: Stage（跟上面的既有含义会撞——那个词已经被"暂存"这个动作占用了，Phase 是刻意选的另一个词，避免 `error.stage` 被误读成 manifest 的 `staged_all` 状态）

**候选级失败（Candidate-specific failure）** / **系统性失败（Systemic failure）**：
`SubtitlePipeline.run_candidates()` 处理某个候选时失败，按代码位置（不是按异常类型或 Phase）分两类。候选级失败——指纹校验（转写前/后）、音频提取、字幕段校验——记录进 `manifest.json` 的 `candidate_failures` 字段后跳过该候选，继续处理批次里的下一个；resume 时不会重试已记录的候选级失败。系统性失败——只有 `engine.transcribe()` 自身抛出的异常，以及引擎未加载这个编程错误——照常向上传播，中止整批排在后面的候选。`stage()`/`commit()`/`verify()` 排除候选级失败的候选后处理剩下的，也就是说一次运行可能"部分成功"：写回的是通过质量门禁的那部分候选，不要求全批候选都成功。
_Avoid_: 把候选级失败跟质量门禁失败（Phase.QUALITY_GATE）混为一谈——质量门禁失败是另一件事，维持"全批候选必须都通过质量门禁才写回"的既有全有全无语义不变，不参与候选级失败的"部分成功"逻辑。

**Event**：
CLI 子进程（`pipeline.py`/`cli.py`）向 GUI 报告运行状态用的结构化消息，逐行 JSON，跟人读文字打印在同一条 stdout 上，靠是否为合法 JSON 区分。已知 kind：`engine_start`、`progress`、`run_dir`、`error`、`done`、`candidate_failed`。
_Avoid_: Log line（指人读文字那一行，跟 Event 是两回事，别混用）；WorkerOutcome（那是 GUI 进程内部线程间的消息，不跨进程、不序列化，外形容易跟 Event 混，实际是两个不同边界上的概念，见下）

**WorkerOutcome**：
GUI 内部"后台 worker 线程 → UI 主线程"这条边界上的消息，走 `self.events`（进程内 `queue.Queue`），从不跨进程、从不序列化成 JSON。覆盖一次后台任务的终态：`idle`、`done`、`cancelled`、`error`。`self.events` 队列里还流过另一批进度类通知（下载进度、日志行、设备探测结果等），那批不属于 WorkerOutcome，维持现状用字符串标签 + tuple 表达。
_Avoid_: Event（那是"CLI 子进程 → GUI 进程"跨进程边界的消息，逐行 JSON；WorkerOutcome 是同进程内跨线程边界，两者刻意分开，别把其中一个的 kind 取值套到另一个上）
