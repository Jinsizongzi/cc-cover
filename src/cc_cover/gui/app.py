from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

from cc_cover import __version__
from cc_cover.core.pipeline import (
    PipelineError,
    load_optional_json,
    run_completion_stats,
    run_status_label,
    write_summary,
)
from cc_cover.gui.background import (
    CancelledOutcome,
    DoneOutcome,
    ErrorOutcome,
    IdleOutcome,
    TaskCancelled,
    WorkerOutcome,
    run_in_background,
)
from cc_cover.gui.candidate_list import CandidateListPanel, scan_confirmation_stats
from cc_cover.gui.commands import resume_command, scan_command, transcribe_command
from cc_cover.gui.content import FEATURE_TEXT, GUIDE_TEXT
from cc_cover.gui.data_root import (
    RuntimePaths,
    apply_data_root,
    default_data_root,
    ensure_data_root,
    resolve_data_root,
    runtime_paths,
)
from cc_cover.gui.device import (
    device_probe_commands,
    parsed_device,
    parsed_nvidia_probe,
)
from cc_cover.gui.dialogs import DialogHost, enrich_failure
from cc_cover.gui.environment import (
    NOT_INSTALLED_STATUS_LABEL,
    environment_check_command,
    environment_status_label,
    needs_force_reinstall_prompt,
    outdated_packages,
    parse_installed_versions,
    python_candidates,
    reinstall_scope,
    setup_commands,
)
from cc_cover.gui.human_readable import format_size, strip_ansi_escapes
from cc_cover.gui.progress import (
    InstallProgressTracker,
    ProgressTracker,
    failure_info_from_command,
    failure_info_from_run,
    install_progress_text,
    parse_event_line,
    progress_text,
    run_dir_from_events,
    stopped_message,
)
from cc_cover.gui.settings import (
    GUI_DEVICE_CHOICES,
    GuiSettings,
    SettingsError,
    load_gui_settings,
    resolve_default_device,
    save_gui_settings,
)
from cc_cover.gui.storage import (
    CLEANUP_WARNING_BYTES,
    clean_model_cache,
    clear_all_data_text,
    clear_local_data,
    delete_runs,
    directory_size,
    disk_precheck,
    disk_precheck_text,
    estimate_install_required_bytes,
    install_download_bytes,
    list_runs,
    local_data_usage,
    model_cache_cleanup_text,
    runs_total_size,
)
from cc_cover.gui.tasks import (
    TaskRunner,
    play_completion_sound,
    should_play_completion_sound,
)
from cc_cover.gui.win_native import (
    SingleInstanceLock,
    focus_existing_window,
    open_in_explorer,
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


class CCCoverApp(ttk.Frame):
    def __init__(self, master: tk.Tk, paths: RuntimePaths, lock: SingleInstanceLock):
        super().__init__(master, padding=0)
        self.master = master
        self.paths = paths
        self.lock = lock
        self.default_root = default_data_root()
        self.events: queue.Queue[tuple[str, Any] | WorkerOutcome] = queue.Queue()
        self.tasks = TaskRunner(self.paths, self.events)
        self.busy = False
        self.environment_ready = False

        settings = self._load_settings()
        self._saved_device = settings.device
        self.scan_path = tk.StringVar(value=settings.scan_path)
        self.device = tk.StringVar(value=resolve_default_device(settings.device, None))
        self.device_auto = settings.device not in GUI_DEVICE_CHOICES
        self.ffmpeg = tk.StringVar(value=settings.ffmpeg)
        self.hash_videos = tk.BooleanVar(value=settings.hash_videos)
        self.hf_token = tk.StringVar(value=settings.hf_token)
        self.status = tk.StringVar(value="正在检查运行环境…")
        self.environment_status = tk.StringVar(value="检查中")
        self.summary = tk.StringVar(value="尚未选择扫描目录")
        self.progress_var = tk.StringVar(value="")
        self._progress_tracker: ProgressTracker | None = None
        self._session_started_at: float | None = None
        self.cache_size_var = tk.StringVar(value="检查中")
        self._install_tracker: InstallProgressTracker | None = None
        self._save_after_id: str | None = None
        self._applying_device_value = False
        self._device_recheck_after_id: str | None = None
        self._prompt_device_recheck = False

        self._configure_window()
        self._configure_styles()
        self._build_interface()

        self.candidates = CandidateListPanel(
            self.candidate_tree, self.select_all_var, on_change=self._refresh_summary
        )
        self.candidate_tree.bind(
            "<Button-1>", lambda event: self.candidates.on_click(event)
        )
        self.candidate_tree.bind(
            "<Button-3>",
            lambda event: self.candidates.show_context_menu(
                self.master, event, open_in_explorer=open_in_explorer
            ),
        )
        self.dialogs = DialogHost(
            self.master,
            notebook=self.notebook,
            log_tab=self.log_tab,
            runs_root=self.paths.runs_root,
            open_directory=self._open_directory,
            resume_run_dir=self._resume_run_dir,
        )

        for variable in (
            self.scan_path,
            self.device,
            self.ffmpeg,
            self.hash_videos,
            self.hf_token,
        ):
            variable.trace_add("write", self._on_settings_changed)
        self.device.trace_add("write", self._on_device_changed)
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)
        self.after(200, self._refresh_cache_display)
        self.after(350, self.check_environment)

    def _load_settings(self) -> GuiSettings:
        try:
            return load_gui_settings(self.paths.data_root)
        except SettingsError as exc:
            # exc 在 except 块退出后被 Python 删除，必须先把消息绑定出来，
            # 否则延迟回调（after）执行时引用 exc 会抛 NameError，警告框不显示。
            message = f"无法读取 settings.json，已使用默认设置。\n\n{exc}"
            self.master.after(
                0,
                lambda: messagebox.showwarning(
                    "设置文件无效", message, parent=self.master
                ),
            )
            return GuiSettings()

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
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

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

    def _panel(
        self, parent: ttk.Frame, padding: tuple[int, int] = (18, 14)
    ) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=padding)
        panel.pack(fill="x", pady=(0, 10))
        return panel

    def _build_work_tab(self) -> None:
        environment_panel = self._panel(self.work_tab)
        environment_panel.columnconfigure(1, weight=1)
        ttk.Label(environment_panel, text="运行环境", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.environment_label = ttk.Label(
            environment_panel,
            textvariable=self.environment_status,
            style="Body.TLabel",
        )
        self.environment_label.grid(row=0, column=1, sticky="w", padx=(14, 0))
        device_box = ttk.Frame(environment_panel, style="Panel.TFrame")
        device_box.grid(row=0, column=2, padx=(12, 8))
        ttk.Label(device_box, text="运行设备：", style="Body.TLabel").pack(side="left")
        self.device_gpu_radio = ttk.Radiobutton(
            device_box,
            text="NVIDIA GPU",
            variable=self.device,
            value="cuda",
        )
        self.device_gpu_radio.pack(side="left", padx=(6, 0))
        self.device_cpu_radio = ttk.Radiobutton(
            device_box,
            text="CPU",
            variable=self.device,
            value="cpu",
        )
        self.device_cpu_radio.pack(side="left", padx=(8, 0))
        self.setup_button = ttk.Button(
            environment_panel,
            text="安装 / 修复运行环境",
            style="Action.TButton",
            command=self.setup_environment,
        )
        self.setup_button.grid(row=0, column=3, sticky="e")
        self.data_root_path = tk.StringVar(value=str(self.paths.data_root))
        ttk.Label(environment_panel, text="数据目录：", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(
            environment_panel,
            textvariable=self.data_root_path,
            style="Body.TLabel",
            wraplength=380,
            justify="left",
        ).grid(row=1, column=1, columnspan=2, sticky="w", padx=(14, 0), pady=(8, 0))
        self.data_root_button = ttk.Button(
            environment_panel,
            text="更改…",
            command=self.change_data_root,
        )
        self.data_root_button.grid(row=1, column=3, sticky="e", pady=(8, 0))
        ttk.Label(environment_panel, text="模型缓存：", style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(
            environment_panel,
            textvariable=self.cache_size_var,
            style="Body.TLabel",
        ).grid(row=2, column=1, sticky="w", padx=(14, 0), pady=(8, 0))
        self.open_cache_button = ttk.Button(
            environment_panel,
            text="打开缓存位置",
            style="Action.TButton",
            command=self.open_model_cache,
        )
        self.open_cache_button.grid(row=2, column=2, padx=(12, 8), pady=(8, 0))
        self.clear_cache_button = ttk.Button(
            environment_panel,
            text="清理模型缓存",
            style="Action.TButton",
            command=self.clear_model_cache,
        )
        self.clear_cache_button.grid(row=2, column=3, sticky="e", pady=(8, 0))
        self.clear_all_data_button = ttk.Button(
            environment_panel,
            text="清理全部本地数据",
            command=self.clear_all_data,
        )
        self.clear_all_data_button.grid(row=3, column=3, sticky="e", pady=(8, 0))
        ttk.Label(
            environment_panel, text="HF Token（可选）：", style="Body.TLabel"
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.hf_token_entry = ttk.Entry(
            environment_panel, textvariable=self.hf_token, show="*"
        )
        self.hf_token_entry.grid(
            row=4, column=1, columnspan=2, sticky="ew", padx=(14, 0), pady=(8, 0)
        )

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
        self.hash_check = ttk.Checkbutton(
            options_row, text="视频哈希保护", variable=self.hash_videos
        )
        self.hash_check.pack(side="left")

        ffmpeg_row = ttk.Frame(options_panel, style="Panel.TFrame")
        ffmpeg_row.pack(fill="x", pady=(10, 0))
        ttk.Label(ffmpeg_row, text="FFmpeg（通常留空）：", style="Body.TLabel").pack(
            side="left"
        )
        self.ffmpeg_entry = ttk.Entry(ffmpeg_row, textvariable=self.ffmpeg)
        self.ffmpeg_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        self.ffmpeg_button = ttk.Button(
            ffmpeg_row,
            text="选择文件",
            command=self.choose_ffmpeg,
        )
        self.ffmpeg_button.pack(side="left")

        candidate_panel = ttk.Frame(
            self.work_tab, style="Panel.TFrame", padding=(18, 14)
        )
        candidate_panel.pack(fill="both", expand=True, pady=(0, 10))
        candidate_panel.columnconfigure(0, weight=1)
        candidate_panel.rowconfigure(2, weight=1)
        ttk.Label(candidate_panel, text="扫描结果", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        selection_row = ttk.Frame(candidate_panel, style="Panel.TFrame")
        selection_row.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        ttk.Label(selection_row, textvariable=self.summary, style="Body.TLabel").pack(
            side="left"
        )
        self.select_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            selection_row,
            text="全选",
            variable=self.select_all_var,
            command=lambda: self.candidates.toggle_all(),
        ).pack(side="right")
        columns = (
            "state",
            "video",
            "target",
            "duration",
            "size",
            "estimate",
            "format",
        )
        self.candidate_tree = ttk.Treeview(
            candidate_panel, columns=columns, show="tree headings", height=8
        )
        self.candidate_tree.heading("#0", text="")
        self.candidate_tree.column(
            "#0", width=36, minwidth=30, stretch=False, anchor="center"
        )
        self.candidate_tree.heading("state", text="状态")
        self.candidate_tree.heading("video", text="视频")
        self.candidate_tree.heading("target", text="目标 TXT")
        self.candidate_tree.heading("duration", text="时长")
        self.candidate_tree.heading("size", text="大小")
        self.candidate_tree.heading("estimate", text="粗估处理时间")
        self.candidate_tree.heading("format", text="输出格式")
        self.candidate_tree.column("state", width=95, stretch=False)
        self.candidate_tree.column("video", width=200)
        self.candidate_tree.column("target", width=200)
        self.candidate_tree.column("duration", width=70, stretch=False)
        self.candidate_tree.column("size", width=80, stretch=False)
        self.candidate_tree.column("estimate", width=100, stretch=False)
        self.candidate_tree.column("format", width=100, stretch=False)
        self.candidate_tree.tag_configure("excluded", foreground=MUTED)
        self.candidate_tree.tag_configure("conflict", foreground=ERROR)
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
        self.cleanup_runs_button = ttk.Button(
            action_panel,
            text="运行目录清理",
            style="Action.TButton",
            command=self.cleanup_runs,
        )
        self.cleanup_runs_button.pack(side="left", padx=(8, 0))
        self.cancel_button = ttk.Button(
            action_panel,
            text="停止当前任务",
            style="Action.TButton",
            command=self.cancel_task,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=(8, 0))
        ttk.Label(action_panel, textvariable=self.status, style="Subtitle.TLabel").pack(
            side="right"
        )
        progress_row = ttk.Frame(self.work_tab, style="App.TFrame")
        progress_row.pack(fill="x", pady=(0, 2), before=candidate_panel)
        self.progress = ttk.Progressbar(progress_row, mode="indeterminate", length=260)
        self.progress.pack(side="left", padx=(0, 10))
        ttk.Label(
            progress_row,
            textvariable=self.progress_var,
            style="Subtitle.TLabel",
        ).pack(side="left")

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

    def _current_settings(self) -> GuiSettings:
        return GuiSettings(
            scan_path=self.scan_path.get().strip(),
            device=self.device.get(),
            ffmpeg=self.ffmpeg.get().strip(),
            hash_videos=self.hash_videos.get(),
            hf_token=self.hf_token.get().strip(),
        )

    def _on_settings_changed(self, *_args: Any) -> None:
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
        self._save_after_id = self.after(400, self._save_settings)

    def _on_device_changed(self, *_args: Any) -> None:
        if self._applying_device_value:
            return
        self.device_auto = False
        if self._device_recheck_after_id is not None:
            self.after_cancel(self._device_recheck_after_id)
        self._device_recheck_after_id = self.after(400, self._recheck_device)

    def _recheck_device(self) -> None:
        self._device_recheck_after_id = None
        if self.busy:
            self._device_recheck_after_id = self.after(400, self._recheck_device)
            return
        self._start_device_recheck()

    def _recheck_detected_device(self) -> None:
        if self.busy:
            self.after(100, self._recheck_detected_device)
            return
        self._start_device_recheck()

    def _start_device_recheck(self) -> None:
        self.environment_ready = False
        self.environment_status.set("正在检查运行设备…")
        self._prompt_device_recheck = True
        self.check_environment()

    def _save_settings(self) -> None:
        self._save_after_id = None
        try:
            save_gui_settings(self.paths.data_root, self._current_settings())
        except (OSError, SettingsError) as exc:
            messagebox.showwarning("无法保存设置", str(exc), parent=self.master)

    def _flush_settings(self) -> None:
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
            self._save_after_id = None
        with contextlib.suppress(OSError, SettingsError):
            save_gui_settings(self.paths.data_root, self._current_settings())

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
            self.device_gpu_radio,
            self.device_cpu_radio,
            self.setup_button,
            self.data_root_button,
            self.open_cache_button,
            self.clear_cache_button,
            self.clear_all_data_button,
            self.path_entry,
            self.choose_button,
            self.scan_button,
            self.hash_check,
            self.ffmpeg_entry,
            self.ffmpeg_button,
            self.hf_token_entry,
            self.start_button,
            self.resume_button,
            self.cleanup_runs_button,
        ):
            widget.configure(state=state)
        self.cancel_button.configure(state="normal" if busy else "disabled")
        if busy:
            self.progress.configure(mode="indeterminate")
            self.progress.start(10)
        else:
            self.progress.stop()
            self._clear_progress()
        if status is not None:
            self.status.set(status)

    def _start_worker(
        self, worker: Callable[[], None], status: str, *, log_tab: bool = False
    ) -> None:
        if self.busy:
            return
        self.tasks.reset()
        self._set_busy(True, status)
        if log_tab:
            self.notebook.select(self.log_tab)
        threading.Thread(target=worker, daemon=True).start()

    def _hf_token_value(self) -> str:
        return self.hf_token.get().strip()

    def _scan_report(self, root: Path, settings: GuiSettings) -> dict[str, Any]:
        self.events.put(("status", "正在扫描目录…"))
        output = self.tasks.run_capture(
            scan_command(self.paths, root, settings), hf_token=self._hf_token_value()
        )
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

    def change_data_root(self) -> None:
        if self.busy:
            return
        self._flush_settings()
        confirmed = messagebox.askyesno(
            "更改数据根",
            "更改数据根后：\n\n"
            "• 新目录需要重新安装运行环境（venv 不自动迁移）；\n"
            "• 模型缓存不会自动迁移，可手动复制到新目录以跳过重新下载；\n"
            "• 旧目录中的设置文件与运行记录会保留。\n\n"
            "确定继续吗？",
            parent=self.master,
        )
        if not confirmed:
            return
        selected = filedialog.askdirectory(
            parent=self.master,
            title="选择新的数据目录",
            initialdir=str(self.paths.data_root),
            mustexist=True,
        )
        if not selected:
            return
        if Path(selected).expanduser().resolve() == self.paths.data_root.resolve():
            return
        new_lock = SingleInstanceLock(Path(selected))
        try:
            if not new_lock.acquire():
                messagebox.showerror(
                    "数据根被占用",
                    "另一个 CC-Cover 实例正在使用所选的数据目录。\n\n"
                    "请先关闭该实例，或选择其他目录。",
                    parent=self.master,
                )
                return
        except OSError as exc:
            messagebox.showerror("无法创建单实例锁", str(exc), parent=self.master)
            return
        try:
            new_root = apply_data_root(
                self.default_root, self.paths.data_root, Path(selected)
            )
        except (OSError, SettingsError) as exc:
            new_lock.release()
            messagebox.showerror("更改数据根失败", str(exc), parent=self.master)
            return
        self.paths = runtime_paths(data_root=new_root)
        self.tasks.paths = self.paths
        self.dialogs.runs_root = self.paths.runs_root
        try:
            ensure_data_root(self.paths)
        except OSError as exc:
            new_lock.release()
            messagebox.showerror("无法创建数据目录", str(exc), parent=self.master)
            return
        previous_lock = self.lock
        self.lock = new_lock
        previous_lock.release()
        self.data_root_path.set(str(self.paths.data_root))
        self._refresh_cache_display()
        self.environment_ready = False
        self.environment_status.set("需要重新安装")
        messagebox.showinfo(
            "数据根已更改",
            f"数据根已切换为：\n{self.paths.data_root}\n\n"
            "请点击“安装 / 修复运行环境”在新目录安装运行环境。\n"
            f"如需保留已下载的模型，请手动复制到：\n{self.paths.model_cache}",
            parent=self.master,
        )

    def _outdated_from_check_output(self, output: str) -> set[str]:
        return outdated_packages(parse_installed_versions(output))

    def check_environment(self) -> None:
        def worker() -> None:
            device = "auto"

            def run() -> None:
                nonlocal device
                if not self.paths.venv_python.is_file():
                    self.events.put(
                        ("environment", (False, NOT_INSTALLED_STATUS_LABEL))
                    )
                    self.events.put(IdleOutcome("请先安装运行环境"))
                    self._detect_and_report()
                    return
                device = self.device.get()
                output = self.tasks.run_capture(
                    environment_check_command(self.paths, device),
                    hf_token=self._hf_token_value(),
                )
                self._prompt_device_recheck = False
                has_update = bool(self._outdated_from_check_output(output))
                self.events.put(
                    (
                        "environment",
                        (
                            True,
                            environment_status_label(
                                device, output, outdated=has_update
                            ),
                        ),
                    )
                )
                self.events.put(("log", output + "\n"))
                self._detect_and_report()
                self.events.put(IdleOutcome("就绪"))

            def on_cancel(exc: TaskCancelled) -> None:
                self.events.put(
                    CancelledOutcome(
                        info=failure_info_from_command(
                            [], exc, fallback_stage="环境检查"
                        )
                    )
                )

            def on_error(exc: Exception) -> None:
                # 环境未就绪是预期内状态，不当作任务失败：不生成 ErrorOutcome、
                # 不弹错误对话框，只更新状态后照常落回 idle，跟成功路径一致。
                # device 复用 run() 读到的值（而不是再读一次 self.device），
                # 避免 device_detected 事件恰好在两次读取之间被主线程处理、
                # 导致这里选错提示文案（不受 _set_busy 的按钮禁用保护）。
                detail = str(exc).strip() or "需要安装或修复"
                label = "GPU 环境未就绪" if device == "cuda" else "需要安装或修复"
                self.events.put(("environment", (False, label)))
                self.events.put(("log", f"环境检查失败：{detail}\n"))
                self._detect_and_report()
                if self._prompt_device_recheck:
                    self._prompt_device_recheck = False
                    self.events.put(("device_check_failed", label))
                self.events.put(IdleOutcome("就绪"))

            run_in_background(run, on_cancel=on_cancel, on_error=on_error)

        self._start_worker(worker, "正在检查运行环境…")

    def _detect_and_report(self) -> None:
        if not self.device_auto:
            return
        detected: str | None = None
        for command in device_probe_commands(self.paths):
            try:
                output = self.tasks.run_capture(
                    command, hf_token=self._hf_token_value()
                )
            except Exception:
                continue
            detected = parsed_device(output) or parsed_nvidia_probe(output)
            if detected is not None:
                break
        self.events.put(("device_detected", detected or "cpu"))

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
        if self.busy:
            return
        try:
            required = estimate_install_required_bytes(self.device.get())
            check = disk_precheck(self.paths.data_root, required)
            runs_bytes = runs_total_size(list_runs(self.paths.runs_root))
        except OSError as exc:
            messagebox.showerror("磁盘预检失败", str(exc), parent=self.master)
            return
        if not check.sufficient:
            proceed = messagebox.askyesno(
                "磁盘空间不足",
                disk_precheck_text(check, runs_bytes) + "\n\n仍要继续安装吗？",
                parent=self.master,
            )
            if not proceed:
                return

        def worker() -> None:
            chunks: list[str] = []

            def run() -> None:
                self.paths.data_root.mkdir(parents=True, exist_ok=True)
                base_python = self._find_base_python()
                device = self.device.get()
                outdated: set[str] | None = None
                if self.paths.venv_python.is_file():
                    try:
                        check_output = self.tasks.run_capture(
                            environment_check_command(self.paths, device),
                            hf_token=self._hf_token_value(),
                        )
                    except TaskCancelled:
                        raise
                    except Exception:
                        # 环境本身有问题（比如 CUDA 不可用），走全量重装最保险，
                        # 不尝试用一次失败的检查结果算精简重装范围。TaskCancelled
                        # 不在此列——用户主动停止要照常走 on_cancel，不能被这里
                        # 当成"环境检查失败"吞掉、退化成继续跑完整安装。
                        outdated = None
                    else:
                        outdated = self._outdated_from_check_output(check_output)
                        if (
                            needs_force_reinstall_prompt(outdated)
                            and self._confirm_force_reinstall()
                        ):
                            outdated = None
                commands = setup_commands(
                    self.paths, base_python, device, outdated=outdated
                )
                include_torch, include_asr = reinstall_scope(outdated)
                self.events.put(("log", "开始安装运行环境。此过程可能需要较长时间。\n"))
                if device == "cuda" and include_torch:
                    self.events.put(
                        (
                            "log",
                            "将强制重装 GPU 版 PyTorch，以便覆盖已安装的 CPU 包。\n",
                        )
                    )
                self.events.put(
                    (
                        "install_start",
                        (
                            install_download_bytes(
                                device,
                                include_torch=include_torch,
                                include_asr=include_asr,
                            ),
                            len(commands),
                        ),
                    )
                )
                for index, command in enumerate(commands, start=1):
                    self.events.put(
                        ("status", f"正在安装组件 {index}/{len(commands)}…")
                    )
                    self.events.put(("install_component", (index, len(commands))))
                    chunks.append(
                        self.tasks.run_streaming(
                            command, hf_token=self._hf_token_value()
                        )
                    )
                output = self.tasks.run_capture(
                    environment_check_command(self.paths, device),
                    hf_token=self._hf_token_value(),
                )
                chunks.append(output)
                self.events.put(("log", output + "\n"))
                self.events.put(
                    (
                        "environment",
                        (True, environment_status_label(device, output)),
                    )
                )
                self._detect_and_report()
                self.events.put(
                    DoneOutcome(
                        title="安装完成",
                        message="运行环境安装并检查通过，可以开始扫描视频。",
                        run_dir=None,
                    )
                )

            def on_cancel(exc: TaskCancelled) -> None:
                self.events.put(
                    CancelledOutcome(
                        info=failure_info_from_command(
                            chunks, exc, fallback_stage="安装运行环境"
                        )
                    )
                )

            def on_error(exc: Exception) -> None:
                self.events.put(
                    ErrorOutcome(
                        title="环境安装失败",
                        info=failure_info_from_command(
                            chunks, exc, fallback_stage="安装运行环境"
                        ),
                    )
                )

            run_in_background(run, on_cancel=on_cancel, on_error=on_error)

        self._start_worker(worker, "正在准备运行环境…", log_tab=True)

    def scan_directory(self) -> None:
        if not self._ensure_environment():
            return
        try:
            root = self._selected_root()
            settings = self._current_settings()
        except ValueError as exc:
            messagebox.showerror("路径无效", str(exc), parent=self.master)
            return

        def worker() -> None:
            def run() -> None:
                self._scan_report(root, settings)
                self.events.put(IdleOutcome("扫描完成"))

            def on_cancel(exc: TaskCancelled) -> None:
                self.events.put(
                    CancelledOutcome(
                        info=failure_info_from_command([], exc, fallback_stage="扫描")
                    )
                )

            def on_error(exc: Exception) -> None:
                self.events.put(
                    ErrorOutcome(
                        title="扫描失败",
                        info=failure_info_from_command([], exc, fallback_stage="扫描"),
                    )
                )

            run_in_background(run, on_cancel=on_cancel, on_error=on_error)

        self._start_worker(worker, "正在扫描目录…")

    def start_transcription(self) -> None:
        if not self._ensure_environment():
            return
        try:
            root = self._selected_root()
            settings = self._current_settings()
        except ValueError as exc:
            messagebox.showerror("无法开始", str(exc), parent=self.master)
            return
        excluded_paths = self.candidates.excluded_paths()

        def worker() -> None:
            chunks: list[str] = []
            scanning = True
            exclude_file: Path | None = None

            def run() -> None:
                nonlocal scanning, exclude_file
                report = self._scan_report(root, settings)
                scanning = False
                fresh_candidates = report.get("candidates", [])
                excluded_set = set(excluded_paths)
                excluded_matches = [
                    candidate
                    for candidate in fresh_candidates
                    if str(candidate.get("video_path", "")) in excluded_set
                ]
                count = len(fresh_candidates) - len(excluded_matches)
                if excluded_paths:
                    self.paths.data_root.mkdir(parents=True, exist_ok=True)
                    descriptor, name = tempfile.mkstemp(
                        prefix="cc-cover-excluded-",
                        suffix=".json",
                        dir=str(self.paths.data_root),
                    )
                    os.close(descriptor)
                    exclude_file = Path(name)
                    exclude_file.write_text(
                        json.dumps(excluded_paths, ensure_ascii=False),
                        encoding="utf-8",
                    )
                # 报告自带排除（同 stem 冲突）+ 本次 GUI 勾选排除，合计为已排除数。
                excluded_count = scan_confirmation_stats(report)[1] + len(
                    excluded_matches
                )
                if count == 0:
                    self.events.put(
                        DoneOutcome(
                            title="无需处理",
                            message="没有需要处理的候选（所有候选均已排除或没有候选），本次不处理。",
                            run_dir=None,
                        )
                    )
                    return
                if not self._confirm_start(count, excluded_count):
                    self.events.put(IdleOutcome("已取消开始"))
                    return
                self.events.put(
                    (
                        "log",
                        f"扫描发现 {len(fresh_candidates)} 个候选，"
                        f"已排除 {excluded_count} 个，本次处理 {count} 个。\n",
                    )
                )
                self.events.put(("progress_start", count))
                self.events.put(("status", "正在生成并替换字幕…"))
                chunks.append(
                    self.tasks.run_streaming(
                        transcribe_command(
                            self.paths,
                            root,
                            settings,
                            exclude_file=exclude_file,
                        ),
                        hf_token=self._hf_token_value(),
                    )
                )
                run_dir = run_dir_from_events(chunks[-1])
                self.events.put(
                    DoneOutcome(
                        title="字幕补全完成",
                        message=f"已完成 {count} 个字幕文件的生成、替换和复核。",
                        run_dir=run_dir,
                    )
                )

            def on_cancel(exc: TaskCancelled) -> None:
                info = failure_info_from_run(
                    chunks,
                    exc,
                    fallback_stage=("扫描" if scanning else "转写与写回"),
                )
                self._best_effort_summary(info.run_dir)
                self.events.put(CancelledOutcome(info=info))

            def on_error(exc: Exception) -> None:
                self.events.put(
                    ErrorOutcome(
                        title="字幕补全失败",
                        info=failure_info_from_run(
                            chunks,
                            exc,
                            fallback_stage=("扫描" if scanning else "转写与写回"),
                        ),
                    )
                )

            try:
                run_in_background(run, on_cancel=on_cancel, on_error=on_error)
            finally:
                if exclude_file is not None:
                    with contextlib.suppress(OSError):
                        exclude_file.unlink()

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

            def run() -> None:
                self.events.put(("status", "正在继续中断任务…"))
                manifest = load_optional_json(run_dir / "manifest.json") or {}
                candidates = manifest.get("candidates") or []
                total = len(candidates) if isinstance(candidates, list) else 0
                if total:
                    self.events.put(("progress_start", total))
                chunks.append(
                    self.tasks.run_streaming(
                        resume_command(self.paths, run_dir),
                        hf_token=self._hf_token_value(),
                    )
                )
                self.events.put(
                    DoneOutcome(
                        title="任务已完成",
                        message="中断任务已继续执行并完成最终复核。",
                        run_dir=run_dir,
                    )
                )

            def on_cancel(exc: TaskCancelled) -> None:
                info = failure_info_from_run(
                    chunks,
                    exc,
                    fallback_stage="继续中断任务",
                    run_dir=run_dir,
                )
                self._best_effort_summary(info.run_dir)
                self.events.put(CancelledOutcome(info=info))

            def on_error(exc: Exception) -> None:
                self.events.put(
                    ErrorOutcome(
                        title="继续任务失败",
                        info=failure_info_from_run(
                            chunks,
                            exc,
                            fallback_stage="继续中断任务",
                            run_dir=run_dir,
                        ),
                    )
                )

            run_in_background(run, on_cancel=on_cancel, on_error=on_error)

        self._start_worker(worker, "正在继续中断任务…", log_tab=True)

    def cancel_task(self) -> None:
        if not self.busy:
            return
        self.tasks.cancel()
        self.status.set("正在停止任务…")
        self._append_log("\n用户请求停止当前任务。\n")

    def open_runs_directory(self) -> None:
        self._open_directory(self.paths.runs_root)

    def cleanup_runs(self) -> None:
        if self.busy:
            return
        runs = list_runs(self.paths.runs_root)
        dialog = self.dialogs.result_dialog("运行目录清理")
        body = ttk.Frame(dialog, style="Panel.TFrame", padding=(18, 14, 18, 6))
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text="勾选要删除的运行目录（本软件不会自动删除）：",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        columns = ("check", "run_id", "time", "status", "size")
        tree = ttk.Treeview(body, columns=columns, show="headings", height=12)
        for column, text, width, stretch in (
            ("check", "", 36, False),
            ("run_id", "运行 ID", 190, False),
            ("time", "时间", 150, False),
            ("status", "状态", 150, False),
            ("size", "大小", 80, False),
        ):
            tree.heading(column, text=text)
            tree.column(
                column,
                width=width,
                stretch=stretch,
                anchor="center" if column == "check" else "w",
            )
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y")

        by_iid: dict[str, Any] = {}
        for run in runs:
            iid = tree.insert(
                "",
                "end",
                values=(
                    "☐",
                    run.run_id,
                    run.created_at_utc or "（无记录）",
                    run_status_label(run.status),
                    format_size(run.size_bytes),
                ),
                tags=("unchecked",),
            )
            by_iid[iid] = run

        def update_delete_button() -> None:
            selected = [
                iid
                for iid in tree.get_children()
                if "checked" in tree.item(iid, "tags")
            ]
            delete_button.configure(state="normal" if selected else "disabled")

        def toggle_row(_event: Any) -> None:
            iid = tree.identify_row(_event.y)
            if not iid:
                return
            values = list(tree.item(iid, "values"))
            checked = values[0] == "☑"
            values[0] = "☐" if checked else "☑"
            tree.item(
                iid,
                values=values,
                tags=("checked" if not checked else "unchecked",),
            )
            update_delete_button()

        def confirm_delete() -> None:
            selected = [
                by_iid[iid]
                for iid in tree.get_children()
                if "checked" in tree.item(iid, "tags")
            ]
            if not selected:
                return
            total = format_size(sum(run.size_bytes for run in selected))
            confirmed = messagebox.askyesno(
                "确认删除",
                f"将永久删除 {len(selected)} 个运行目录（共 {total}），不可恢复。\n\n"
                "确定继续吗？",
                parent=dialog,
            )
            if not confirmed:
                return
            try:
                deleted = delete_runs(selected)
            except OSError as exc:
                messagebox.showerror(
                    "删除失败",
                    f"部分运行目录删除失败：\n{exc}",
                    parent=dialog,
                )
                return
            messagebox.showinfo(
                "删除完成", f"已删除 {deleted} 个运行目录。", parent=dialog
            )
            dialog.destroy()

        tree.bind("<Button-1>", toggle_row)

        footer = ttk.Frame(dialog, style="Panel.TFrame", padding=(18, 10, 18, 18))
        footer.pack(fill="x")
        total_size = runs_total_size(runs)
        ttk.Label(
            footer,
            text=f"总占用：{format_size(total_size)}",
            style="Body.TLabel",
        ).pack(side="left")
        if total_size > CLEANUP_WARNING_BYTES:
            ttk.Label(
                footer,
                text="⚠ 已超过 5GB，建议清理旧运行目录",
                style="Body.TLabel",
                foreground=WARNING,
            ).pack(side="left", padx=(12, 0))
        delete_button = ttk.Button(
            footer,
            text="删除所选",
            style="Action.TButton",
            command=confirm_delete,
            state="disabled",
        )
        delete_button.pack(side="right")
        ttk.Button(
            footer,
            text="关闭",
            style="Action.TButton",
            command=dialog.destroy,
        ).pack(side="right", padx=(0, 8))

    def _refresh_cache_display(self) -> None:
        self.cache_size_var.set(format_size(directory_size(self.paths.model_cache)))

    def open_model_cache(self) -> None:
        self._open_directory(self.paths.model_cache)

    def clear_model_cache(self) -> None:
        if self.busy:
            return
        size = directory_size(self.paths.model_cache)
        confirmed = messagebox.askyesno(
            "清理模型缓存",
            model_cache_cleanup_text(size),
            parent=self.master,
        )
        if not confirmed:
            return
        try:
            clean_model_cache(self.paths)
        except OSError as exc:
            messagebox.showerror("清理失败", str(exc), parent=self.master)
            return
        self._refresh_cache_display()
        messagebox.showinfo(
            "清理完成", "模型缓存已清理，下次运行将重新下载模型。", parent=self.master
        )

    def clear_all_data(self) -> None:
        if self.busy:
            return
        usage = local_data_usage(self.paths)
        confirmed = messagebox.askyesno(
            "清理全部本地数据", clear_all_data_text(usage), parent=self.master
        )
        if not confirmed:
            return
        try:
            clear_local_data(self.paths)
        except OSError as exc:
            messagebox.showerror("清理失败", str(exc), parent=self.master)
            return
        self._refresh_cache_display()
        self.environment_ready = False
        self.environment_status.set("需要重新安装")
        messagebox.showinfo(
            "清理完成",
            "venv、模型缓存、运行记录与临时文件已删除。\n"
            "请点击“安装 / 修复运行环境”重新安装。",
            parent=self.master,
        )

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, value: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", strip_ansi_escapes(value))
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _refresh_summary(self) -> None:
        self.summary.set(self.candidates.summary_text())

    def _open_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def _best_effort_summary(self, run_dir: Path | None) -> None:
        """任务被停止（子进程被终止，execute 的 finally 不会执行）时补写摘要。"""
        if run_dir is None:
            return
        with contextlib.suppress(OSError, PipelineError):
            write_summary(run_dir)

    def _confirm_start(self, candidate_count: int, excluded_count: int) -> bool:
        """在 GUI 线程弹出开始前确认框并等待结果；任务已停止时视为取消。"""
        result_queue: queue.Queue[bool] = queue.Queue()
        self.events.put(
            ("confirm_start", (candidate_count, excluded_count, result_queue))
        )
        confirmed = result_queue.get()
        if self.tasks.cancel_requested:
            return False
        return confirmed

    def _confirm_force_reinstall(self) -> bool:
        """版本比对全部匹配、没有可精简重装的目标时，询问是否强制完整重装。

        用于版本号没变但文件本身损坏（磁盘错误、杀软误隔离等）这类版本比对
        查不出来的场景——给用户一个手动逃生舱，而不是让"安装 / 修复运行环境"
        在这种情况下悄悄变成什么都不做。任务已停止时视为取消，跟 _confirm_start
        保持同一种语义。
        """
        result_queue: queue.Queue[bool] = queue.Queue()
        self.events.put(("confirm_force_reinstall", result_queue))
        confirmed = result_queue.get()
        if self.tasks.cancel_requested:
            return False
        return confirmed

    def _start_progress(self, total: int) -> None:
        self._session_started_at = time.monotonic()
        self._progress_tracker = ProgressTracker(
            total=total, started_at=self._session_started_at
        )
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.progress.stop()
        self._refresh_progress()
        self.after(500, self._schedule_progress_refresh)

    def _schedule_progress_refresh(self) -> None:
        if self._progress_tracker is None:
            return
        self._refresh_progress()
        self.after(500, self._schedule_progress_refresh)

    def _refresh_progress(self) -> None:
        tracker = self._progress_tracker
        if tracker is None:
            return
        snapshot = tracker.snapshot()
        self.progress.configure(value=snapshot.percent)
        self.progress_var.set(progress_text(snapshot))

    def _on_progress_line(self, line: str) -> None:
        if self._install_tracker is not None:
            self._install_tracker.on_output(line)
            self._refresh_install_progress()
            return
        if self._progress_tracker is None:
            return
        self._progress_tracker.on_event(parse_event_line(line))
        self._refresh_progress()

    def _clear_progress(self) -> None:
        if self._progress_tracker is None and self._install_tracker is None:
            return
        self._progress_tracker = None
        self._session_started_at = None
        self._install_tracker = None
        self.progress_var.set("")
        self.progress.configure(mode="indeterminate", value=0)

    def _start_install_progress(self, total_bytes: int, component_count: int) -> None:
        self._install_tracker = InstallProgressTracker(
            total_bytes=total_bytes,
            component_count=component_count,
            started_at=time.monotonic(),
        )
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.progress.stop()
        self._refresh_install_progress()
        self.after(500, self._schedule_install_progress_refresh)

    def _on_install_component(self, index: int, count: int) -> None:
        if self._install_tracker is not None:
            self._install_tracker.on_component(index)
            self._refresh_install_progress()

    def _schedule_install_progress_refresh(self) -> None:
        if self._install_tracker is None:
            return
        self._refresh_install_progress()
        self.after(500, self._schedule_install_progress_refresh)

    def _refresh_install_progress(self) -> None:
        tracker = self._install_tracker
        if tracker is None:
            return
        snapshot = tracker.snapshot()
        self.progress.configure(value=snapshot.percent)
        self.progress_var.set(install_progress_text(snapshot))

    def _session_elapsed(self) -> float | None:
        """本次运行会话的已耗时（进度开始到结束），用于完成提示音的判断。"""
        if self._session_started_at is None:
            return None
        return time.monotonic() - self._session_started_at

    def _handle_worker_outcome(self, outcome: WorkerOutcome) -> None:
        match outcome:
            case IdleOutcome(status=status):
                self._set_busy(False, status)
            case DoneOutcome(title=title, message=message, run_dir=run_dir):
                session_elapsed = self._session_elapsed()
                self._set_busy(False, "就绪")
                if run_dir is not None:
                    stats = run_completion_stats(run_dir)
                    if should_play_completion_sound(session_elapsed):
                        self.after(0, play_completion_sound)
                    self.dialogs.show_done_dialog(title, message, run_dir, stats)
                else:
                    self.dialogs.show_done_dialog(title, message, None)
            case CancelledOutcome(info=info):
                self._set_busy(False, "任务已停止")
                self._append_log(stopped_message(info) + "\n")
                self.dialogs.show_stopped_dialog(info)
            case ErrorOutcome(title=title, info=info):
                self._set_busy(False, "发生错误")
                self._append_log(f"\n错误：{info.reason}\n")
                self.dialogs.show_failure_dialog(title, enrich_failure(info))

    def _poll_events(self) -> None:
        try:
            while True:
                item = self.events.get_nowait()
                if isinstance(item, WorkerOutcome):
                    self._handle_worker_outcome(item)
                    continue
                event, payload = item
                if event == "log":
                    self._append_log(str(payload))
                    self._on_progress_line(str(payload))
                elif event == "progress_start":
                    self._start_progress(int(payload))
                elif event == "install_start":
                    total_bytes, component_count = payload
                    self._start_install_progress(total_bytes, component_count)
                elif event == "install_component":
                    index, count = payload
                    self._on_install_component(index, count)
                elif event == "confirm_start":
                    candidate_count, excluded_count, result_queue = payload
                    self.dialogs.show_confirm_start_dialog(
                        candidate_count, excluded_count, result_queue
                    )
                elif event == "confirm_force_reinstall":
                    self.dialogs.show_confirm_force_reinstall_dialog(payload)
                elif event == "status":
                    self.status.set(str(payload))
                elif event == "scan_report":
                    self.candidates.load(payload)
                elif event == "environment":
                    ready, label = payload
                    self.environment_ready = bool(ready)
                    self.environment_status.set(str(label))
                elif event == "device_detected":
                    if self.device_auto:
                        resolved = resolve_default_device(
                            self._saved_device, str(payload)
                        )
                        changed = resolved != self.device.get()
                        self._applying_device_value = True
                        self.device.set(resolved)
                        self._applying_device_value = False
                        self.device_auto = False
                        if changed:
                            self.after(0, self._recheck_detected_device)
                elif event == "device_check_failed":
                    self._prompt_device_recheck = False
                    label = str(payload)
                    repair = messagebox.askyesno(
                        "运行设备已切换",
                        f"当前运行环境无法匹配所选设备（{label}）。\n\n"
                        "是否现在重新安装 / 修复运行环境？",
                        parent=self.master,
                    )
                    if repair:
                        self.setup_environment()
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
        self._flush_settings()
        self.master.destroy()


def main() -> None:
    default_root = default_data_root()
    root = tk.Tk()
    root.withdraw()
    try:
        resolution = resolve_data_root(default_root)
    except SettingsError as exc:
        messagebox.showerror("设置文件无效", str(exc), parent=root)
        root.destroy()
        return
    if resolution.needs_choice:
        messagebox.showinfo(
            "选择数据目录",
            "CC-Cover 无法在程序所在目录或已配置的数据目录中保存数据"
            "（目录不可写或不可用）。\n\n请选择一个新的数据目录（例如"
            "其他磁盘上的 CC-Cover-Data 文件夹）。",
            parent=root,
        )
        selected = filedialog.askdirectory(
            parent=root,
            title="选择数据目录",
            initialdir=str(Path.home()),
            mustexist=True,
        )
        if not selected:
            root.destroy()
            return
        try:
            active_root = apply_data_root(default_root, resolution.root, Path(selected))
        except (OSError, SettingsError) as exc:
            messagebox.showerror("无法使用所选目录", str(exc), parent=root)
            root.destroy()
            return
    else:
        active_root = resolution.root
    paths = runtime_paths(data_root=active_root)
    try:
        ensure_data_root(paths)
    except OSError as exc:
        messagebox.showerror("无法创建数据目录", str(exc), parent=root)
        root.destroy()
        return
    lock = SingleInstanceLock(paths.data_root)
    try:
        if not lock.acquire():
            messagebox.showinfo(
                "CC-Cover 已在运行",
                "CC-Cover 已在运行，本实例即将退出。\n\n"
                "可切换到已打开的 CC-Cover 窗口继续操作。",
                parent=root,
            )
            focus_existing_window()
            root.destroy()
            return
    except OSError as exc:
        messagebox.showerror("无法创建单实例锁", str(exc), parent=root)
        root.destroy()
        return
    try:
        CCCoverApp(root, paths, lock)
        root.deiconify()
        root.mainloop()
    finally:
        lock.release()


if __name__ == "__main__":
    main()
