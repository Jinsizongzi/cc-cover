# 从 CCCoverApp 抽出 ProgressPresenter

`CCCoverApp`（`gui/app.py`）里 `_start_progress`/`_schedule_progress_refresh`/`_refresh_progress`/`_on_progress_line`/`_clear_progress`/`_start_install_progress`/`_on_install_component`/`_schedule_install_progress_refresh`/`_refresh_install_progress`/`_session_elapsed` 共 10 个方法，只是围着已经很深的 `ProgressTracker`/`InstallProgressTracker`（`gui/progress.py`）转的 Tk 挂件（`self.progress` 这个 `ttk.Progressbar` + `self.progress_var` 这个 `StringVar`）+ `after()` 轮询调度——跟 CandidateListPanel/DialogHost/TaskRunner 当初被抽出去之前是同一种"挂件+状态"形状。决定抽出第五个 collaborator：`ProgressPresenter`，构造参数是已经建好的 `progress: ttk.Progressbar`/`progress_var: tk.StringVar`（`_build_work_tab` 照旧建这两个控件，建完直接交给 `ProgressPresenter`）。公开接口 7 个：`start_busy()`/`stop_busy()`（替代 `_set_busy` 里直接操作 `self.progress` 的那两行）、`start(total)`/`start_install(bytes, count)`、`on_line(line)`/`on_install_component(i, n)`、`elapsed()`。原来的 4 个刷新/调度方法（`_schedule_progress_refresh`/`_refresh_progress`/`_schedule_install_progress_refresh`/`_refresh_install_progress`）变成内部实现，不再对外暴露；`ProgressPresenter` 自己持有 `self.progress`，调度直接用 `self.progress.after(...)`，不需要额外注入 master 或调度器。

## Considered Options

- `_set_busy` 里"忙碌转圈"那两行（busy=True 时 `self.progress.configure(mode="indeterminate"); self.progress.start(10)`，busy=False 时 `self.progress.stop()`）要不要也归 `ProgressPresenter`：考虑过留在 `CCCoverApp` 里直接操作、`ProgressPresenter` 只管确定进度条那半——被否决，一个控件被两个对象同时摸，以后谁在 `self.progress` 上加新逻辑得先想清楚该找哪个对象；改为整个控件都归 `ProgressPresenter`，`_set_busy` 只调 `start_busy()`/`stop_busy()`，跟 DialogHost 独占 Toplevel、TaskRunner 独占 `subprocess.Popen` 是同一个原则——一个控件只有一个主人。
- `stop_busy()` 要不要把原来 `_clear_progress()` 的重置逻辑（清两个 tracker、清 `_session_started_at`、清 `progress_var`、把控件设回 indeterminate/0）单独拆成一个 `clear()` 方法：查过调用点，`_clear_progress()` 在原代码里只被 `_set_busy(False, ...)` 一处调用，没有第二个独立调用方——不拆，直接并进 `stop_busy()`，7 个方法而不是 8 个；等真出现需要单独清空、不停转圈的场景再拆。

## Consequences

- **依赖方向**：`ProgressPresenter` 依赖 `gui/progress.py` 的 `ProgressTracker`/`InstallProgressTracker`（已是深模块，不变）；不反向依赖 `CCCoverApp`。
- **顺序约束**：`elapsed()` 必须在调用方触发 `stop_busy()` **之前**读——`stop_busy()` 会把 `_session_started_at` 清空。`_handle_worker_outcome` 的 `DoneOutcome` 分支保持"先读 elapsed，再切 busy"这个既有顺序不变。
- **不受影响的行为**：`elapsed()` 只在跑过 `start()`（转写进度）时才非 `None`——`start_install()`（装环境）不设这个计时锚点，装环境永远不触发"完成提示音"，这条行为原样保留。
- **测试**：新测试加进 `tests/test_progress.py`（`ProgressTracker`/`InstallProgressTracker` 已在这个文件里，`ProgressPresenter` 是围着它们转的新 collaborator，同一个文件），用真实（或 headless）`tk.Tk()` 实例，跟 `test_candidate_list.py`/`test_dialogs.py` 的既有先例一致；`tests/test_gui_smoke.py` 不需要改，`CCCoverApp` 对外构造签名不变。
- **来源**：本次决策产出于架构评审第三轮（候选②），经两个独立 agent 核实过范围与推理。
