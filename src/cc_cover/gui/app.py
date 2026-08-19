from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from cc_cover.core.pipeline import (
    PipelineError,
    load_optional_json,
    run_completion_stats,
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
from cc_cover.gui.data_root import (
    RuntimePaths,
    apply_data_root,
    default_data_root,
    ensure_data_root,
    resolve_data_root,
    runtime_paths,
)
from cc_cover.gui.dialogs import DialogHost, enrich_failure
from cc_cover.gui.environment import EnvironmentController
from cc_cover.gui.human_readable import format_size, strip_ansi_escapes
from cc_cover.gui.layout import build_interface, configure_styles, configure_window
from cc_cover.gui.progress import (
    ProgressPresenter,
    failure_info_from_command,
    failure_info_from_run,
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
    clean_model_cache,
    clear_all_data_text,
    clear_local_data,
    directory_size,
    local_data_usage,
    model_cache_cleanup_text,
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
PRIMARY = "#3157d5"
PRIMARY_DARK = "#2445b3"
SUCCESS = "#17803d"
WARNING = "#b54708"


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
        self.cache_size_var = tk.StringVar(value="检查中")
        self._save_after_id: str | None = None
        self._applying_device_value = False
        self._device_recheck_after_id: str | None = None

        configure_window(self)
        configure_styles(self)
        build_interface(self)

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
        self.environment = EnvironmentController(
            self.paths,
            self.events,
            self.tasks,
            self.dialogs,
            is_device_auto=lambda: self.device_auto,
            current_device=self.device.get,
            hf_token=self._hf_token_value,
        )
        self.progress_presenter = ProgressPresenter(self.progress, self.progress_var)

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
        self.environment.request_recheck_prompt()
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
            self.progress_presenter.start_busy()
        else:
            self.progress_presenter.stop_busy()
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

    def check_environment(self) -> None:
        self._start_worker(self.environment.build_check_worker(), "正在检查运行环境…")

    def setup_environment(self) -> None:
        if not self.environment.precheck_setup():
            return
        self._start_worker(
            self.environment.build_setup_worker(), "正在准备运行环境…", log_tab=True
        )

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
        self.dialogs.show_cleanup_dialog()

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

    def _handle_worker_outcome(self, outcome: WorkerOutcome) -> None:
        match outcome:
            case IdleOutcome(status=status):
                self._set_busy(False, status)
            case DoneOutcome(title=title, message=message, run_dir=run_dir):
                session_elapsed = self.progress_presenter.elapsed()
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
                    self.progress_presenter.on_line(str(payload))
                elif event == "progress_start":
                    self.progress_presenter.start(int(payload))
                elif event == "install_start":
                    total_bytes, component_count = payload
                    self.progress_presenter.start_install(total_bytes, component_count)
                elif event == "install_component":
                    index, count = payload
                    self.progress_presenter.on_install_component(index, count)
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
                    self.environment.clear_recheck_prompt()
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
