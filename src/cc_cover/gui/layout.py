from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any

from cc_cover import __version__
from cc_cover.gui.content import FEATURE_TEXT, GUIDE_TEXT

BACKGROUND = "#f5f7fb"
PANEL = "#ffffff"
INK = "#172033"
MUTED = "#667085"
ERROR = "#b42318"


def configure_window(app: Any) -> None:
    app.master.title(f"CC-Cover {__version__} · 双模型字幕补全")
    app.master.geometry("1080x760")
    app.master.minsize(920, 650)
    app.master.configure(background=BACKGROUND)
    app.pack(fill="both", expand=True)


def configure_styles(app: Any) -> None:
    style = ttk.Style(app.master)
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


def build_interface(app: Any) -> None:
    header = ttk.Frame(app, style="App.TFrame", padding=(28, 22, 28, 12))
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

    app.notebook = ttk.Notebook(app)
    app.notebook.pack(fill="both", expand=True, padx=24, pady=(0, 18))

    app.work_tab = ttk.Frame(app.notebook, style="App.TFrame", padding=4)
    app.feature_tab = ttk.Frame(app.notebook, style="App.TFrame", padding=12)
    app.guide_tab = ttk.Frame(app.notebook, style="App.TFrame", padding=12)
    app.log_tab = ttk.Frame(app.notebook, style="App.TFrame", padding=12)
    app.notebook.add(app.work_tab, text="  字幕补全  ")
    app.notebook.add(app.feature_tab, text="  功能说明  ")
    app.notebook.add(app.guide_tab, text="  操作指南  ")
    app.notebook.add(app.log_tab, text="  运行日志  ")

    build_work_tab(app)
    build_text_tab(app.feature_tab, FEATURE_TEXT)
    build_text_tab(app.guide_tab, GUIDE_TEXT)
    build_log_tab(app)


def panel(parent: ttk.Frame, padding: tuple[int, int] = (18, 14)) -> ttk.Frame:
    widget = ttk.Frame(parent, style="Panel.TFrame", padding=padding)
    widget.pack(fill="x", pady=(0, 10))
    return widget


def build_work_tab(app: Any) -> None:
    environment_panel = panel(app.work_tab)
    environment_panel.columnconfigure(1, weight=1)
    ttk.Label(environment_panel, text="运行环境", style="Section.TLabel").grid(
        row=0, column=0, sticky="w"
    )
    app.environment_label = ttk.Label(
        environment_panel,
        textvariable=app.environment_status,
        style="Body.TLabel",
    )
    app.environment_label.grid(row=0, column=1, sticky="w", padx=(14, 0))
    device_box = ttk.Frame(environment_panel, style="Panel.TFrame")
    device_box.grid(row=0, column=2, padx=(12, 8))
    ttk.Label(device_box, text="运行设备：", style="Body.TLabel").pack(side="left")
    app.device_gpu_radio = ttk.Radiobutton(
        device_box,
        text="NVIDIA GPU",
        variable=app.device,
        value="cuda",
    )
    app.device_gpu_radio.pack(side="left", padx=(6, 0))
    app.device_cpu_radio = ttk.Radiobutton(
        device_box,
        text="CPU",
        variable=app.device,
        value="cpu",
    )
    app.device_cpu_radio.pack(side="left", padx=(8, 0))
    app.setup_button = ttk.Button(
        environment_panel,
        text="安装 / 修复运行环境",
        style="Action.TButton",
        command=app.setup_environment,
    )
    app.setup_button.grid(row=0, column=3, sticky="e")
    app.data_root_path = tk.StringVar(value=str(app.paths.data_root))
    ttk.Label(environment_panel, text="数据目录：", style="Body.TLabel").grid(
        row=1, column=0, sticky="w", pady=(8, 0)
    )
    ttk.Label(
        environment_panel,
        textvariable=app.data_root_path,
        style="Body.TLabel",
        wraplength=380,
        justify="left",
    ).grid(row=1, column=1, columnspan=2, sticky="w", padx=(14, 0), pady=(8, 0))
    app.data_root_button = ttk.Button(
        environment_panel,
        text="更改…",
        command=app.change_data_root,
    )
    app.data_root_button.grid(row=1, column=3, sticky="e", pady=(8, 0))
    ttk.Label(environment_panel, text="模型缓存：", style="Body.TLabel").grid(
        row=2, column=0, sticky="w", pady=(8, 0)
    )
    ttk.Label(
        environment_panel,
        textvariable=app.cache_size_var,
        style="Body.TLabel",
    ).grid(row=2, column=1, sticky="w", padx=(14, 0), pady=(8, 0))
    app.open_cache_button = ttk.Button(
        environment_panel,
        text="打开缓存位置",
        style="Action.TButton",
        command=app.open_model_cache,
    )
    app.open_cache_button.grid(row=2, column=2, padx=(12, 8), pady=(8, 0))
    app.clear_cache_button = ttk.Button(
        environment_panel,
        text="清理模型缓存",
        style="Action.TButton",
        command=app.clear_model_cache,
    )
    app.clear_cache_button.grid(row=2, column=3, sticky="e", pady=(8, 0))
    app.clear_all_data_button = ttk.Button(
        environment_panel,
        text="清理全部本地数据",
        command=app.clear_all_data,
    )
    app.clear_all_data_button.grid(row=3, column=3, sticky="e", pady=(8, 0))
    ttk.Label(
        environment_panel, text="HF Token（可选）：", style="Body.TLabel"
    ).grid(row=4, column=0, sticky="w", pady=(8, 0))
    app.hf_token_entry = ttk.Entry(
        environment_panel, textvariable=app.hf_token, show="*"
    )
    app.hf_token_entry.grid(
        row=4, column=1, columnspan=2, sticky="ew", padx=(14, 0), pady=(8, 0)
    )

    path_panel = panel(app.work_tab)
    path_panel.columnconfigure(0, weight=1)
    ttk.Label(path_panel, text="扫描目录", style="Section.TLabel").grid(
        row=0, column=0, sticky="w", columnspan=3
    )
    app.path_entry = ttk.Entry(path_panel, textvariable=app.scan_path)
    app.path_entry.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    app.choose_button = ttk.Button(
        path_panel,
        text="选择文件夹",
        style="Action.TButton",
        command=app.choose_directory,
    )
    app.choose_button.grid(row=1, column=1, padx=(10, 0), pady=(10, 0))
    app.scan_button = ttk.Button(
        path_panel,
        text="重新扫描",
        style="Action.TButton",
        command=app.scan_directory,
    )
    app.scan_button.grid(row=1, column=2, padx=(8, 0), pady=(10, 0))

    options_panel = panel(app.work_tab)
    ttk.Label(options_panel, text="处理选项", style="Section.TLabel").pack(
        anchor="w"
    )
    options_row = ttk.Frame(options_panel, style="Panel.TFrame")
    options_row.pack(fill="x", pady=(10, 0))
    app.hash_check = ttk.Checkbutton(
        options_row, text="视频哈希保护", variable=app.hash_videos
    )
    app.hash_check.pack(side="left")

    ffmpeg_row = ttk.Frame(options_panel, style="Panel.TFrame")
    ffmpeg_row.pack(fill="x", pady=(10, 0))
    ttk.Label(ffmpeg_row, text="FFmpeg（通常留空）：", style="Body.TLabel").pack(
        side="left"
    )
    app.ffmpeg_entry = ttk.Entry(ffmpeg_row, textvariable=app.ffmpeg)
    app.ffmpeg_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
    app.ffmpeg_button = ttk.Button(
        ffmpeg_row,
        text="选择文件",
        command=app.choose_ffmpeg,
    )
    app.ffmpeg_button.pack(side="left")

    candidate_panel = ttk.Frame(app.work_tab, style="Panel.TFrame", padding=(18, 14))
    candidate_panel.pack(fill="both", expand=True, pady=(0, 10))
    candidate_panel.columnconfigure(0, weight=1)
    candidate_panel.rowconfigure(2, weight=1)
    ttk.Label(candidate_panel, text="扫描结果", style="Section.TLabel").grid(
        row=0, column=0, sticky="w"
    )
    selection_row = ttk.Frame(candidate_panel, style="Panel.TFrame")
    selection_row.grid(row=1, column=0, sticky="ew", pady=(4, 8))
    ttk.Label(selection_row, textvariable=app.summary, style="Body.TLabel").pack(
        side="left"
    )
    app.select_all_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        selection_row,
        text="全选",
        variable=app.select_all_var,
        command=lambda: app.candidates.toggle_all(),
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
    app.candidate_tree = ttk.Treeview(
        candidate_panel, columns=columns, show="tree headings", height=8
    )
    app.candidate_tree.heading("#0", text="")
    app.candidate_tree.column(
        "#0", width=36, minwidth=30, stretch=False, anchor="center"
    )
    app.candidate_tree.heading("state", text="状态")
    app.candidate_tree.heading("video", text="视频")
    app.candidate_tree.heading("target", text="目标 TXT")
    app.candidate_tree.heading("duration", text="时长")
    app.candidate_tree.heading("size", text="大小")
    app.candidate_tree.heading("estimate", text="粗估处理时间")
    app.candidate_tree.heading("format", text="输出格式")
    app.candidate_tree.column("state", width=95, stretch=False)
    app.candidate_tree.column("video", width=200)
    app.candidate_tree.column("target", width=200)
    app.candidate_tree.column("duration", width=70, stretch=False)
    app.candidate_tree.column("size", width=80, stretch=False)
    app.candidate_tree.column("estimate", width=100, stretch=False)
    app.candidate_tree.column("format", width=100, stretch=False)
    app.candidate_tree.tag_configure("excluded", foreground=MUTED)
    app.candidate_tree.tag_configure("conflict", foreground=ERROR)
    scrollbar = ttk.Scrollbar(
        candidate_panel, orient="vertical", command=app.candidate_tree.yview
    )
    app.candidate_tree.configure(yscrollcommand=scrollbar.set)
    app.candidate_tree.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns")

    action_panel = ttk.Frame(app.work_tab, style="App.TFrame")
    action_panel.pack(fill="x", pady=(0, 2), before=candidate_panel)
    app.start_button = ttk.Button(
        action_panel,
        text="开始补全并替换",
        style="Primary.TButton",
        command=app.start_transcription,
    )
    app.start_button.pack(side="left")
    app.resume_button = ttk.Button(
        action_panel,
        text="继续中断任务",
        style="Action.TButton",
        command=app.resume_run,
    )
    app.resume_button.pack(side="left", padx=(10, 0))
    app.open_runs_button = ttk.Button(
        action_panel,
        text="打开运行目录",
        style="Action.TButton",
        command=app.open_runs_directory,
    )
    app.open_runs_button.pack(side="left", padx=(8, 0))
    app.cleanup_runs_button = ttk.Button(
        action_panel,
        text="运行目录清理",
        style="Action.TButton",
        command=app.cleanup_runs,
    )
    app.cleanup_runs_button.pack(side="left", padx=(8, 0))
    app.cancel_button = ttk.Button(
        action_panel,
        text="停止当前任务",
        style="Action.TButton",
        command=app.cancel_task,
        state="disabled",
    )
    app.cancel_button.pack(side="left", padx=(8, 0))
    ttk.Label(action_panel, textvariable=app.status, style="Subtitle.TLabel").pack(
        side="right"
    )
    progress_row = ttk.Frame(app.work_tab, style="App.TFrame")
    progress_row.pack(fill="x", pady=(0, 2), before=candidate_panel)
    app.progress = ttk.Progressbar(progress_row, mode="indeterminate", length=260)
    app.progress.pack(side="left", padx=(0, 10))
    ttk.Label(
        progress_row,
        textvariable=app.progress_var,
        style="Subtitle.TLabel",
    ).pack(side="left")


def build_text_tab(parent: ttk.Frame, content: str) -> None:
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


def build_log_tab(app: Any) -> None:
    toolbar = ttk.Frame(app.log_tab, style="App.TFrame")
    toolbar.pack(fill="x", pady=(0, 8))
    ttk.Button(toolbar, text="清空日志", command=app.clear_log).pack(side="right")
    app.log_text = scrolledtext.ScrolledText(
        app.log_tab,
        wrap="word",
        background="#101828",
        foreground="#e4e7ec",
        insertbackground="#ffffff",
        font=("Consolas", 9),
        padx=12,
        pady=12,
    )
    app.log_text.pack(fill="both", expand=True)
    app.log_text.configure(state="disabled")
