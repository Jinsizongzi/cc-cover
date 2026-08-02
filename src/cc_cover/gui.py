from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

from cc_cover import __version__
from cc_cover.gui_support import (
    FailureInfo,
    GuiOptions,
    RuntimePaths,
    command_environment,
    environment_check_command,
    environment_status_label,
    error_text,
    failure_info_from_command,
    first_failed_sample,
    python_candidates,
    resume_command,
    run_is_resumable,
    runtime_paths,
    scan_command,
    setup_commands,
    stopped_message,
    terminate_process_tree,
    transcribe_command,
)


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BACKGROUND = "#f5f7fb"
PANEL = "#ffffff"
INK = "#172033"
MUTED = "#667085"
PRIMARY = "#3157d5"
PRIMARY_DARK = "#2445b3"
SUCCESS = "#17803d"
WARNING = "#b54708"
ERROR = "#b42318"


class TaskCancelled(RuntimeError):
    """用户主动停止任务时抛出，不当作任务失败处理。"""


FEATURE_TEXT = """核心功能

1. 全量扫描
所有视频默认都是候选：无论同名 TXT 是否存在、是否为空或已有内容，都会重新生成并在校验通过后覆盖。与视频不同名的 TXT 不会被触碰。

2. 双模型字幕生成
FunASR 负责中文正文和句级时间戳；faster-whisper 作为第二模型，对识别结果进行匹配、差异分析和风险审计。

3. 固定输出格式
字幕固定输出为 MM:SS / H:MM:SS 时间戳加字幕文字，段间空一行；UTF-8 无 BOM、CRLF 换行、末尾换行。

4. 自动替换字幕
用户点击“开始补全并替换”后，软件完成扫描、识别、质量校验和格式校验，并直接原子替换目标 TXT，无需额外写回参数。

5. 冲突检测
发现阶段检测同 stem 多视频指向同一目标 TXT 的冲突，标记“冲突”，默认不处理、不写回，报告会列出所有相关视频。

6. 写回保护
处理前记录视频和字幕快照；写回前复核文件状态；写回时先保存备份；批量写回失败会自动回滚。

7. 可恢复运行
模型结果、审计报告、待写字幕、备份和复核报告都会保存在运行目录。中断后可通过“继续中断任务”选择运行目录继续。

8. 本地处理
视频、音频和字幕均在本机处理。只有首次安装依赖或首次下载模型时需要联网。"""


GUIDE_TEXT = """操作指南

首次使用

1. 打开 CC-Cover.exe。
2. 在“运行环境”区域选择 NVIDIA GPU 或 CPU。
3. 点击“安装 / 修复运行环境”。软件会自动创建隔离环境并安装所需组件，不需要打开命令行窗口。
4. 等待状态显示“运行环境已就绪”。首次安装耗时取决于网络速度。
5. 若从 CPU 改为 GPU（或反过来），必须再次点击“安装 / 修复运行环境”，否则会继续使用旧的 PyTorch 包。

补全字幕

1. 点击“选择文件夹”，选择需要处理的视频目录。软件不会预设任何扫描路径。
2. 选择目录后会自动进行快速扫描，也可以点击“重新扫描”。
3. 在候选列表中确认待补全字幕；冲突项会标记“冲突”，默认不会处理。
4. 根据需要调整推理设备、视频哈希保护与 FFmpeg 路径。
5. 点击“开始补全并替换”。软件将自动完成双模型识别、审计、格式化、备份、替换和最终复核。
6. 在“运行日志”页查看实时进度。完成后可点击“打开运行目录”查看详细产物。

中断恢复

1. 运行期间可随时点击“停止当前任务”（扫描、环境检查、转写、写回均支持）。
2. 主动停止不会弹出错误框，而是提示“任务已停止，产物已暂存，可点击‘继续中断任务’恢复”。
3. 失败或停止后可直接点击“继续中断任务”恢复，软件会复用已完成的模型结果，并在校验通过后继续写回。
4. 也可以点击“打开运行目录”查看运行产物与日志。

选项说明

• 自动设备：优先使用可用 GPU，否则使用 CPU。
• 视频哈希保护：处理前计算视频 SHA-256，安全性更高，但首次扫描大型目录会更慢。
• FFmpeg：通常无需指定；只有自动检测失败时才选择 ffmpeg.exe。

注意事项

• 点击开始后，校验通过的同名 TXT 会被直接替换或创建。
• 不要在运行期间移动、改名或编辑候选视频和 TXT。
• 首次运行模型会下载较大的模型文件，请保持网络稳定和足够磁盘空间。
• 软件关闭前会提示是否停止正在运行的任务。"""


class CCCoverApp(ttk.Frame):
    def __init__(self, master: tk.Tk, paths: RuntimePaths):
        super().__init__(master, padding=0)
        self.master = master
        self.paths = paths
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.busy = False
        self.cancel_requested = False
        self.stop_triggered = False
        self.environment_ready = False
        self.last_report: dict[str, Any] | None = None

        self.scan_path = tk.StringVar()
        self.device = tk.StringVar(value="auto")
        self.accelerator = tk.StringVar(value="cuda")
        self.ffmpeg = tk.StringVar()
        self.hash_videos = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="正在检查运行环境…")
        self.environment_status = tk.StringVar(value="检查中")
        self.summary = tk.StringVar(value="尚未选择扫描目录")

        self._configure_window()
        self._configure_styles()
        self._build_interface()
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)
        self.after(350, self.check_environment)

    def _configure_window(self) -> None:
        self.master.title(f"CC-Cover {__version__} · 双模型字幕补全")
        self.master.geometry("1080x760")
        self.master.minsize(920, 650)
        self.master.configure(background=BACKGROUND)
        self.pack(fill="both", expand=True)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.master)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("App.TFrame", background=BACKGROUND)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure(
            "Title.TLabel",
            background=BACKGROUND,
            foreground=INK,
            font=("Microsoft YaHei UI", 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=BACKGROUND,
            foreground=MUTED,
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=PANEL,
            foreground=INK,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=PANEL,
            foreground=MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Primary.TButton",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(18, 9),
        )
        style.configure("Action.TButton", padding=(12, 7))
        style.configure("Treeview", rowheight=28, font=("Microsoft YaHei UI", 9))
        style.configure(
            "Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold")
        )

    def _build_interface(self) -> None:
        header = ttk.Frame(self, style="App.TFrame", padding=(28, 22, 28, 12))
        header.pack(fill="x")
        title_row = ttk.Frame(header, style="App.TFrame")
        title_row.pack(fill="x")
        ttk.Label(title_row, text="CC-Cover", style="Title.TLabel").pack(side="left")
        ttk.Label(
            title_row,
            text=f"  v{__version__}",
            style="Subtitle.TLabel",
        ).pack(side="left", pady=(8, 0))
        ttk.Label(
            header,
            text="选择目录后，自动扫描、双模型识别、格式校验并替换空字幕 TXT",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=24, pady=(0, 18))

        self.work_tab = ttk.Frame(self.notebook, style="App.TFrame", padding=4)
        self.feature_tab = ttk.Frame(self.notebook, style="App.TFrame", padding=12)
        self.guide_tab = ttk.Frame(self.notebook, style="App.TFrame", padding=12)
        self.log_tab = ttk.Frame(self.notebook, style="App.TFrame", padding=12)
        self.notebook.add(self.work_tab, text="  字幕补全  ")
        self.notebook.add(self.feature_tab, text="  功能说明  ")
        self.notebook.add(self.guide_tab, text="  操作指南  ")
        self.notebook.add(self.log_tab, text="  运行日志  ")

        self._build_work_tab()
        self._build_text_tab(self.feature_tab, FEATURE_TEXT)
        self._build_text_tab(self.guide_tab, GUIDE_TEXT)
        self._build_log_tab()

    def _panel(self, parent: ttk.Frame, padding: tuple[int, int] = (18, 14)) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=padding)
        panel.pack(fill="x", pady=(0, 10))
        return panel

    def _build_work_tab(self) -> None:
        environment_panel = self._panel(self.work_tab)
        environment_panel.columnconfigure(1, weight=1)
        ttk.Label(
            environment_panel, text="运行环境", style="Section.TLabel"
        ).grid(row=0, column=0, sticky="w")
        self.environment_label = ttk.Label(
            environment_panel,
            textvariable=self.environment_status,
            style="Body.TLabel",
        )
        self.environment_label.grid(row=0, column=1, sticky="w", padx=(14, 0))
        accelerator_box = ttk.Frame(environment_panel, style="Panel.TFrame")
        accelerator_box.grid(row=0, column=2, padx=(12, 8))
        ttk.Radiobutton(
            accelerator_box,
            text="NVIDIA GPU",
            variable=self.accelerator,
            value="cuda",
        ).pack(side="left")
        ttk.Radiobutton(
            accelerator_box,
            text="CPU",
            variable=self.accelerator,
            value="cpu",
        ).pack(side="left", padx=(8, 0))
        self.setup_button = ttk.Button(
            environment_panel,
            text="安装 / 修复运行环境",
            style="Action.TButton",
            command=self.setup_environment,
        )
        self.setup_button.grid(row=0, column=3, sticky="e")

        path_panel = self._panel(self.work_tab)
        path_panel.columnconfigure(0, weight=1)
        ttk.Label(path_panel, text="扫描目录", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", columnspan=3
        )
        self.path_entry = ttk.Entry(path_panel, textvariable=self.scan_path)
        self.path_entry.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.choose_button = ttk.Button(
            path_panel,
            text="选择文件夹",
            style="Action.TButton",
            command=self.choose_directory,
        )
        self.choose_button.grid(row=1, column=1, padx=(10, 0), pady=(10, 0))
        self.scan_button = ttk.Button(
            path_panel,
            text="重新扫描",
            style="Action.TButton",
            command=self.scan_directory,
        )
        self.scan_button.grid(row=1, column=2, padx=(8, 0), pady=(10, 0))

        options_panel = self._panel(self.work_tab)
        ttk.Label(options_panel, text="处理选项", style="Section.TLabel").pack(
            anchor="w"
        )
        options_row = ttk.Frame(options_panel, style="Panel.TFrame")
        options_row.pack(fill="x", pady=(10, 0))
        ttk.Label(options_row, text="推理设备：", style="Body.TLabel").pack(
            side="left"
        )
        self.device_combo = ttk.Combobox(
            options_row,
            textvariable=self.device,
            values=("auto", "cuda", "cpu"),
            state="readonly",
            width=9,
        )
        self.device_combo.pack(side="left", padx=(4, 18))
        ttk.Checkbutton(
            options_row, text="视频哈希保护", variable=self.hash_videos
        ).pack(side="left")

        ffmpeg_row = ttk.Frame(options_panel, style="Panel.TFrame")
        ffmpeg_row.pack(fill="x", pady=(10, 0))
        ttk.Label(
            ffmpeg_row, text="FFmpeg（通常留空）：", style="Body.TLabel"
        ).pack(side="left")
        ttk.Entry(ffmpeg_row, textvariable=self.ffmpeg).pack(
            side="left", fill="x", expand=True, padx=(8, 8)
        )
        ttk.Button(
            ffmpeg_row,
            text="选择文件",
            command=self.choose_ffmpeg,
        ).pack(side="left")

        candidate_panel = ttk.Frame(
            self.work_tab, style="Panel.TFrame", padding=(18, 14)
        )
        candidate_panel.pack(fill="both", expand=True, pady=(0, 10))
        candidate_panel.columnconfigure(0, weight=1)
        candidate_panel.rowconfigure(2, weight=1)
        ttk.Label(candidate_panel, text="扫描结果", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            candidate_panel, textvariable=self.summary, style="Body.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(4, 8))
        columns = ("state", "video", "target", "format")
        self.candidate_tree = ttk.Treeview(
            candidate_panel, columns=columns, show="headings", height=8
        )
        self.candidate_tree.heading("state", text="状态")
        self.candidate_tree.heading("video", text="视频")
        self.candidate_tree.heading("target", text="目标 TXT")
        self.candidate_tree.heading("format", text="输出格式")
        self.candidate_tree.column("state", width=95, stretch=False)
        self.candidate_tree.column("video", width=260)
        self.candidate_tree.column("target", width=260)
        self.candidate_tree.column("format", width=140, stretch=False)
        scrollbar = ttk.Scrollbar(
            candidate_panel, orient="vertical", command=self.candidate_tree.yview
        )
        self.candidate_tree.configure(yscrollcommand=scrollbar.set)
        self.candidate_tree.grid(row=2, column=0, sticky="nsew")
        scrollbar.grid(row=2, column=1, sticky="ns")

        action_panel = ttk.Frame(self.work_tab, style="App.TFrame")
        action_panel.pack(fill="x", pady=(0, 2), before=candidate_panel)
        self.start_button = ttk.Button(
            action_panel,
            text="开始补全并替换",
            style="Primary.TButton",
            command=self.start_transcription,
        )
        self.start_button.pack(side="left")
        self.resume_button = ttk.Button(
            action_panel,
            text="继续中断任务",
            style="Action.TButton",
            command=self.resume_run,
        )
        self.resume_button.pack(side="left", padx=(10, 0))
        self.open_runs_button = ttk.Button(
            action_panel,
            text="打开运行目录",
            style="Action.TButton",
            command=self.open_runs_directory,
        )
        self.open_runs_button.pack(side="left", padx=(8, 0))
        self.cancel_button = ttk.Button(
            action_panel,
            text="停止当前任务",
            style="Action.TButton",
            command=self.cancel_task,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=(8, 0))
        self.progress = ttk.Progressbar(action_panel, mode="indeterminate", length=180)
        self.progress.pack(side="right", padx=(12, 0))
        ttk.Label(
            action_panel, textvariable=self.status, style="Subtitle.TLabel"
        ).pack(side="right")

    def _build_text_tab(self, parent: ttk.Frame, content: str) -> None:
        text = scrolledtext.ScrolledText(
            parent,
            wrap="word",
            relief="flat",
            borderwidth=0,
            background=PANEL,
            foreground=INK,
            font=("Microsoft YaHei UI", 10),
            padx=22,
            pady=20,
            spacing1=3,
            spacing3=7,
        )
        text.pack(fill="both", expand=True)
        text.insert("1.0", content)
        text.configure(state="disabled")

    def _build_log_tab(self) -> None:
        toolbar = ttk.Frame(self.log_tab, style="App.TFrame")
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="清空日志", command=self.clear_log).pack(side="right")
        self.log_text = scrolledtext.ScrolledText(
            self.log_tab,
            wrap="word",
            background="#101828",
            foreground="#e4e7ec",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            padx=12,
            pady=12,
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _gui_options(self) -> GuiOptions:
        ffmpeg_text = self.ffmpeg.get().strip().strip('"')
        return GuiOptions(
            device=self.device.get(),
            hash_videos=self.hash_videos.get(),
            ffmpeg=Path(ffmpeg_text).resolve() if ffmpeg_text else None,
        )

    def _selected_root(self) -> Path:
        value = self.scan_path.get().strip().strip('"')
        if not value:
            raise ValueError("请先选择需要扫描的视频文件夹。")
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"扫描目录不存在：{path}")
        return path

    def _ensure_environment(self) -> bool:
        if self.environment_ready and self.paths.venv_python.is_file():
            return True
        messagebox.showinfo(
            "运行环境尚未就绪",
            "请先点击“安装 / 修复运行环境”，等待环境检查通过。",
            parent=self.master,
        )
        return False

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for widget in (
            self.setup_button,
            self.choose_button,
            self.scan_button,
            self.start_button,
            self.resume_button,
        ):
            widget.configure(state=state)
        self.cancel_button.configure(state="normal" if busy else "disabled")
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()
        if status is not None:
            self.status.set(status)

    def _start_worker(
        self, worker: Callable[[], None], status: str, *, log_tab: bool = False
    ) -> None:
        if self.busy:
            return
        self.cancel_requested = False
        self.stop_triggered = False
        self._set_busy(True, status)
        if log_tab:
            self.notebook.select(self.log_tab)
        threading.Thread(target=worker, daemon=True).start()

    def _process_environment(self) -> dict[str, str]:
        return command_environment(self.paths)

    def _run_capture(self, command: list[str]) -> str:
        if self.cancel_requested:
            raise TaskCancelled("任务已由用户停止。")
        process = subprocess.Popen(
            command,
            cwd=str(self.paths.data_root),
            env=self._process_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
        self.process = process
        assert process.stdout is not None
        return self._finish_process(process, "".join(process.stdout))

    def _run_streaming(self, command: list[str]) -> str:
        if self.cancel_requested:
            raise TaskCancelled("任务已由用户停止。")
        self.events.put(("log", "\n▶ " + " ".join(command[1:]) + "\n"))
        process = subprocess.Popen(
            command,
            cwd=str(self.paths.data_root),
            env=self._process_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
        self.process = process
        assert process.stdout is not None
        lines: list[str] = []
        for line in process.stdout:
            lines.append(line)
            self.events.put(("log", line))
        return self._finish_process(process, "".join(lines))

    def _finish_process(
        self, process: subprocess.Popen[str], output: str
    ) -> str:
        return_code = process.wait()
        self.process = None
        if self.stop_triggered:
            raise TaskCancelled(
                output.strip() or "任务已由用户停止。运行产物可以稍后继续。"
            )
        if self.cancel_requested and return_code != 0:
            raise TaskCancelled(
                output.strip() or "任务已由用户停止。运行产物可以稍后继续。"
            )
        if return_code != 0:
            raise RuntimeError(output.strip() or f"任务执行失败，退出代码：{return_code}")
        return output

    def _scan_report(self, root: Path, options: GuiOptions) -> dict[str, Any]:
        self.events.put(("status", "正在扫描目录…"))
        output = self._run_capture(scan_command(self.paths, root, options))
        try:
            report = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"无法解析扫描结果：{exc}\n{output[:1200]}") from exc
        self.events.put(("scan_report", report))
        return report

    def choose_directory(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.master, title="选择需要扫描的视频文件夹", mustexist=True
        )
        if not selected:
            return
        self.scan_path.set(str(Path(selected).resolve()))
        if self._ensure_environment():
            self.scan_directory()

    def choose_ffmpeg(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.master,
            title="选择 ffmpeg.exe",
            filetypes=(("FFmpeg", "ffmpeg.exe"), ("可执行文件", "*.exe")),
        )
        if selected:
            self.ffmpeg.set(selected)

    def check_environment(self) -> None:
        def worker() -> None:
            if not self.paths.venv_python.is_file():
                self.events.put(("environment", (False, "尚未安装")))
                self.events.put(("idle", "请先安装运行环境"))
                return
            accelerator = self.accelerator.get()
            try:
                output = self._run_capture(
                    environment_check_command(self.paths, accelerator)
                )
            except TaskCancelled as exc:
                self.events.put(
                    (
                        "cancelled",
                        failure_info_from_command(
                            [], exc, fallback_stage="环境检查"
                        ),
                    )
                )
                return
            except Exception as exc:
                detail = str(exc).strip() or "需要安装或修复"
                label = (
                    "GPU 环境未就绪"
                    if accelerator == "cuda"
                    else "需要安装或修复"
                )
                self.events.put(("environment", (False, label)))
                self.events.put(("log", f"环境检查失败：{detail}\n"))
            else:
                self.events.put(
                    (
                        "environment",
                        (True, environment_status_label(accelerator, output)),
                    )
                )
                self.events.put(("log", output + "\n"))
            self.events.put(("idle", "就绪"))

        self._start_worker(worker, "正在检查运行环境…")

    def _find_base_python(self) -> list[str]:
        for candidate in python_candidates():
            completed = subprocess.run(
                [
                    *candidate,
                    "-c",
                    (
                        "import sys; "
                        "raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)"
                    ),
                ],
                capture_output=True,
                text=True,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
            if completed.returncode == 0:
                return candidate
        raise RuntimeError(
            "未找到 Python 3.10、3.11 或 3.12。请先安装 Python，并勾选 Add Python to PATH。"
        )

    def setup_environment(self) -> None:
        def worker() -> None:
            chunks: list[str] = []
            try:
                self.paths.data_root.mkdir(parents=True, exist_ok=True)
                base_python = self._find_base_python()
                accelerator = self.accelerator.get()
                commands = setup_commands(self.paths, base_python, accelerator)
                self.events.put(("log", "开始安装运行环境。此过程可能需要较长时间。\n"))
                if accelerator == "cuda":
                    self.events.put(
                        (
                            "log",
                            "将强制重装 GPU 版 PyTorch，以便覆盖已安装的 CPU 包。\n",
                        )
                    )
                for index, command in enumerate(commands, start=1):
                    self.events.put(
                        ("status", f"正在安装组件 {index}/{len(commands)}…")
                    )
                    chunks.append(self._run_streaming(command))
                output = self._run_capture(
                    environment_check_command(self.paths, accelerator)
                )
                chunks.append(output)
                self.events.put(("log", output + "\n"))
                self.events.put(
                    (
                        "environment",
                        (True, environment_status_label(accelerator, output)),
                    )
                )
                self.events.put(
                    (
                        "done",
                        ("安装完成", "运行环境安装并检查通过，可以开始扫描视频。"),
                    )
                )
            except TaskCancelled as exc:
                self.events.put(
                    (
                        "cancelled",
                        failure_info_from_command(
                            chunks, exc, fallback_stage="安装运行环境"
                        ),
                    )
                )
            except Exception as exc:
                self.events.put(
                    (
                        "error",
                        (
                            "环境安装失败",
                            failure_info_from_command(
                                chunks,
                                exc,
                                fallback_stage="安装运行环境",
                            ),
                        ),
                    )
                )

        self._start_worker(worker, "正在准备运行环境…", log_tab=True)

    def scan_directory(self) -> None:
        if not self._ensure_environment():
            return
        try:
            root = self._selected_root()
            options = self._gui_options()
        except ValueError as exc:
            messagebox.showerror("路径无效", str(exc), parent=self.master)
            return

        def worker() -> None:
            try:
                self._scan_report(root, options)
                self.events.put(("idle", "扫描完成"))
            except TaskCancelled as exc:
                self.events.put(
                    (
                        "cancelled",
                        failure_info_from_command(
                            [], exc, fallback_stage="扫描"
                        ),
                    )
                )
            except Exception as exc:
                self.events.put(
                    (
                        "error",
                        (
                            "扫描失败",
                            failure_info_from_command(
                                [], exc, fallback_stage="扫描"
                            ),
                        ),
                    )
                )

        self._start_worker(worker, "正在扫描目录…")

    def start_transcription(self) -> None:
        if not self._ensure_environment():
            return
        try:
            root = self._selected_root()
            options = self._gui_options()
        except ValueError as exc:
            messagebox.showerror("无法开始", str(exc), parent=self.master)
            return

        def worker() -> None:
            chunks: list[str] = []
            scanning = True
            try:
                report = self._scan_report(root, options)
                scanning = False
                count = int(report.get("candidate_count", 0))
                if count == 0:
                    self.events.put(
                        (
                            "done",
                            ("无需处理", "所选目录中没有符合条件的空字幕 TXT。"),
                        )
                    )
                    return
                self.events.put(
                    ("log", f"扫描发现 {count} 个待补全字幕，开始双模型处理。\n")
                )
                self.events.put(("status", "正在生成并替换字幕…"))
                chunks.append(
                    self._run_streaming(
                        transcribe_command(self.paths, root, options)
                    )
                )
                self.events.put(
                    (
                        "done",
                        (
                            "字幕补全完成",
                            f"已完成 {count} 个字幕文件的生成、替换和复核。",
                        ),
                    )
                )
            except TaskCancelled as exc:
                self.events.put(
                    (
                        "cancelled",
                        failure_info_from_command(
                            chunks,
                            exc,
                            fallback_stage=(
                                "扫描" if scanning else "转写与写回"
                            ),
                        ),
                    )
                )
            except Exception as exc:
                self.events.put(
                    (
                        "error",
                        (
                            "字幕补全失败",
                            failure_info_from_command(
                                chunks,
                                exc,
                                fallback_stage=(
                                    "扫描" if scanning else "转写与写回"
                                ),
                            ),
                        ),
                    )
                )

        self._start_worker(worker, "正在扫描并准备处理…", log_tab=True)

    def resume_run(self) -> None:
        if not self._ensure_environment():
            return
        self.paths.runs_root.mkdir(parents=True, exist_ok=True)
        selected = filedialog.askdirectory(
            parent=self.master,
            title="选择需要继续的运行目录",
            initialdir=str(self.paths.runs_root),
            mustexist=True,
        )
        if not selected:
            return
        run_dir = Path(selected).resolve()
        if not (run_dir / "manifest.json").is_file():
            messagebox.showerror(
                "运行目录无效",
                "所选目录中没有 manifest.json。",
                parent=self.master,
            )
            return
        self._resume_run_dir(run_dir)

    def _resume_run_dir(self, run_dir: Path) -> None:
        if self.busy:
            return

        def worker() -> None:
            chunks: list[str] = []
            try:
                self.events.put(("status", "正在继续中断任务…"))
                chunks.append(
                    self._run_streaming(resume_command(self.paths, run_dir))
                )
                self.events.put(
                    (
                        "done",
                        ("任务已完成", "中断任务已继续执行并完成最终复核。"),
                    )
                )
            except TaskCancelled as exc:
                self.events.put(
                    (
                        "cancelled",
                        failure_info_from_command(
                            chunks,
                            exc,
                            fallback_stage="继续中断任务",
                            run_dir=run_dir,
                        ),
                    )
                )
            except Exception as exc:
                self.events.put(
                    (
                        "error",
                        (
                            "继续任务失败",
                            failure_info_from_command(
                                chunks,
                                exc,
                                fallback_stage="继续中断任务",
                                run_dir=run_dir,
                            ),
                        ),
                    )
                )

        self._start_worker(worker, "正在继续中断任务…", log_tab=True)

    def cancel_task(self) -> None:
        if not self.busy:
            return
        self.cancel_requested = True
        process = self.process
        if process is not None and process.poll() is None:
            self.stop_triggered = True
            terminate_process_tree(process)
        self.status.set("正在停止任务…")
        self._append_log("\n用户请求停止当前任务。\n")

    def open_runs_directory(self) -> None:
        self._open_directory(self.paths.runs_root)

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, value: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", value)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _display_report(self, report: dict[str, Any]) -> None:
        self.last_report = report
        for item in self.candidate_tree.get_children():
            self.candidate_tree.delete(item)
        candidates = report.get("candidates", [])
        for candidate in candidates:
            self.candidate_tree.insert(
                "",
                "end",
                values=(
                    candidate.get("state", ""),
                    candidate.get("video_path", ""),
                    candidate.get("target_path", ""),
                    "MM:SS / H:MM:SS",
                ),
            )
        for conflict in report.get("conflicts", []):
            for video in conflict.get("videos", []):
                self.candidate_tree.insert(
                    "",
                    "end",
                    values=(
                        "冲突",
                        video,
                        conflict.get("target_path", ""),
                        "—",
                    ),
                )
        self.summary.set(
            "视频 {video} 个 · 待补全 {candidate} 个 · 冲突 {conflict} 个 · 受保护非空 TXT {protected} 个".format(
                video=report.get("video_count", 0),
                candidate=report.get("candidate_count", 0),
                conflict=report.get("conflict_count", 0),
                protected=report.get("protected_nonempty_txt_count", 0),
            )
        )

    def _enrich_failure(self, info: FailureInfo) -> FailureInfo:
        if info.run_dir is None:
            return info
        if info.file and info.stage != "质量门禁":
            return info
        sample = first_failed_sample(info.run_dir)
        if sample is None:
            return info
        video, reason = sample
        return replace(
            info,
            file=info.file or video,
            reason=reason if info.stage == "质量门禁" else info.reason,
        )

    def _result_dialog(self, title: str) -> tk.Toplevel:
        dialog = tk.Toplevel(self.master)
        dialog.title(title)
        dialog.configure(background=PANEL)
        dialog.resizable(False, False)
        dialog.transient(self.master)
        dialog.grab_set()
        dialog.update_idletasks()
        x = self.master.winfo_rootx() + max(
            0, (self.master.winfo_width() - dialog.winfo_reqwidth()) // 2
        )
        y = self.master.winfo_rooty() + max(
            0, (self.master.winfo_height() - dialog.winfo_reqheight()) // 3
        )
        dialog.geometry(f"+{x}+{y}")
        return dialog

    def _dialog_row(
        self, parent: ttk.Frame, row: int, label: str, value: str
    ) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").grid(
            row=row, column=0, sticky="nw", pady=(4, 0)
        )
        ttk.Label(
            parent,
            text=value,
            style="Body.TLabel",
            wraplength=500,
            justify="left",
        ).grid(row=row, column=1, sticky="nw", pady=(4, 0), padx=(10, 0))

    def _copy_error(
        self, dialog: tk.Toplevel, title: str, info: FailureInfo
    ) -> None:
        self.master.clipboard_clear()
        self.master.clipboard_append(error_text(title, info))
        dialog.destroy()

    def _view_log(self, dialog: tk.Toplevel) -> None:
        self.notebook.select(self.log_tab)
        dialog.destroy()

    def _open_run_dir(self, dialog: tk.Toplevel, info: FailureInfo) -> None:
        target = info.run_dir if info.run_dir is not None else self.paths.runs_root
        self._open_directory(target)
        dialog.destroy()

    def _resume_from_dialog(
        self, dialog: tk.Toplevel, info: FailureInfo
    ) -> None:
        run_dir = info.run_dir
        dialog.destroy()
        if run_dir is not None and run_is_resumable(run_dir):
            self._resume_run_dir(run_dir)

    def _open_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def _show_failure_dialog(self, title: str, info: FailureInfo) -> None:
        dialog = self._result_dialog(title)
        body = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 18, 20, 6))
        body.pack(fill="x")
        body.columnconfigure(1, weight=1)
        row = 0
        for label, value in (
            ("文件：", info.file),
            ("阶段：", info.stage),
            ("原因：", info.reason),
        ):
            if not value:
                continue
            self._dialog_row(body, row, label, value)
            row += 1
        if info.run_dir is not None:
            self._dialog_row(body, row, "运行目录：", str(info.run_dir))
            row += 1
        if info.done_count is not None and info.total_count is not None:
            note = (
                f"已处理 {info.done_count}/{info.total_count} 个视频，"
                "全部产物已暂存，可点击「继续中断任务」恢复，已完成的文件不会重跑。"
            )
            ttk.Label(
                body, text=note, style="Body.TLabel", wraplength=500
            ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(12, 0))
        actions = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 12, 20, 18))
        actions.pack(fill="x")
        ttk.Button(
            actions,
            text="复制错误信息",
            style="Action.TButton",
            command=lambda: self._copy_error(dialog, title, info),
        ).pack(side="left")
        ttk.Button(
            actions,
            text="查看日志",
            style="Action.TButton",
            command=lambda: self._view_log(dialog),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text="打开运行目录",
            style="Action.TButton",
            command=lambda: self._open_run_dir(dialog, info),
        ).pack(side="left", padx=(8, 0))
        if run_is_resumable(info.run_dir):
            ttk.Button(
                actions,
                text="继续中断任务",
                style="Action.TButton",
                command=lambda: self._resume_from_dialog(dialog, info),
            ).pack(side="right")
        ttk.Button(
            actions,
            text="关闭",
            style="Action.TButton",
            command=dialog.destroy,
        ).pack(side="right", padx=(0, 8))

    def _show_stopped_dialog(self, info: FailureInfo) -> None:
        dialog = self._result_dialog("任务已停止")
        body = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 18, 20, 6))
        body.pack(fill="x")
        body.columnconfigure(1, weight=1)
        resumable = run_is_resumable(info.run_dir)
        ttk.Label(
            body,
            text=stopped_message(info),
            style="Body.TLabel",
            wraplength=500,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        if info.run_dir is not None:
            self._dialog_row(body, 1, "运行目录：", str(info.run_dir))
        actions = ttk.Frame(dialog, style="Panel.TFrame", padding=(20, 12, 20, 18))
        actions.pack(fill="x")
        if resumable:
            ttk.Button(
                actions,
                text="继续中断任务",
                style="Action.TButton",
                command=lambda: self._resume_from_dialog(dialog, info),
            ).pack(side="left")
        ttk.Button(
            actions,
            text="打开运行目录",
            style="Action.TButton",
            command=lambda: self._open_run_dir(dialog, info),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text="关闭",
            style="Action.TButton",
            command=dialog.destroy,
        ).pack(side="right")

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "log":
                    self._append_log(str(payload))
                elif event == "status":
                    self.status.set(str(payload))
                elif event == "scan_report":
                    self._display_report(payload)
                elif event == "environment":
                    ready, label = payload
                    self.environment_ready = bool(ready)
                    self.environment_status.set(str(label))
                elif event == "idle":
                    self._set_busy(False, str(payload))
                elif event == "done":
                    title, message = payload
                    self._set_busy(False, "就绪")
                    messagebox.showinfo(title, message, parent=self.master)
                elif event == "cancelled":
                    info = payload
                    self._set_busy(False, "任务已停止")
                    self._append_log(stopped_message(info) + "\n")
                    self._show_stopped_dialog(info)
                elif event == "error":
                    title, info = payload
                    self._set_busy(False, "发生错误")
                    self._append_log(f"\n错误：{info.reason}\n")
                    self._show_failure_dialog(title, self._enrich_failure(info))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _on_close(self) -> None:
        if self.busy:
            close = messagebox.askyesno(
                "任务仍在运行",
                "关闭软件会停止当前任务。确定要关闭吗？",
                parent=self.master,
            )
            if not close:
                return
            self.cancel_task()
        self.master.destroy()


def main() -> None:
    paths = runtime_paths()
    paths.data_root.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    CCCoverApp(root, paths)
    root.mainloop()


if __name__ == "__main__":
    main()
