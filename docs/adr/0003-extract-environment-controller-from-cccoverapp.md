# 从 CCCoverApp 抽出 EnvironmentController

`CCCoverApp`（`gui/app.py`）经 ADR-0002 拆成组合根 + 三个 collaborator（CandidateListPanel/DialogHost/TaskRunner）后，仍留着 8 个环境/设备相关方法（`check_environment`/`setup_environment`/`_detect_and_report`/`_find_base_python`/`_outdated_from_check_output`/`_recheck_device`/`_recheck_detected_device`/`_start_device_recheck`），只被 `tests/test_gui_smoke.py` 的 3 个冒烟测试间接覆盖，增量装/全量装选择、强制重装门槛、GPU 不可用提示这些分支逻辑没有专项单测。决定抽出第四个 collaborator：`EnvironmentController`，放进已有的 `gui/environment.py`（目前只有 `setup_commands`/`outdated_packages`/`reinstall_scope`/`needs_force_reinstall_prompt` 等纯函数），构造参数为 `paths`/`events`/`tasks`（TaskRunner）/`dialogs`（DialogHost）/`is_device_auto`（读 `device_auto` 当前值的零参回调，只读不写）。公开接口 5 个：`precheck_setup()`（磁盘预检，不够时通过 `dialogs` 弹窗确认，主线程同步执行，返回是否继续）、`build_setup_worker()`/`build_check_worker()`（只造还没执行的 worker 闭包，不负责派发到线程）、`request_recheck_prompt()`/`clear_recheck_prompt()`（原 `_prompt_device_recheck` 标志位，改由 EnvironmentController 自己持有）。

## Considered Options

- 磁盘预检的两处弹窗（原来直接 `messagebox.*`，用 `parent=self.master`）：考虑过让 `EnvironmentController` 直接拿 `master` 自己弹——被否决，会让它变成第二个"自己能弹窗"的 collaborator，跟 `DialogHost` 已确立的"弹窗内容归 DialogHost"分工冲突；也考虑过注入两个窄回调而不是整个 `dialogs` 引用——被否决，为两个弹窗多开两个构造参数，不如直接给 `DialogHost` 加两个方法（`show_disk_precheck_error`/`confirm_low_disk_space`，接收原始数据、内部自己拼文案，跟 `show_confirm_start_dialog` 等既有方法同一个模式）。
- 设备重检的防抖轮询（`_recheck_device`/`_recheck_detected_device`/`_start_device_recheck`）：考虑过整体搬进 `EnvironmentController`——被否决，这 3 个方法响应的是 `CCCoverApp` 自己的变量 trace（`self.device.trace_add`）和事件分发（`_poll_events` 处理 `"device_detected"`/`"device_check_failed"`），是设备相关 UI 状态变化的响应式胶水，不是环境检查领域逻辑本身，留在 `CCCoverApp`。
- 起线程、切换 busy 状态：考虑过让 `EnvironmentController` 自己调度（拿一个 `start_worker` 回调，或整个重新实现一遍）——被否决，`_set_busy` 牵动 17 个控件是 `CCCoverApp` 的地盘，复制一份或反向注入回调都会让"谁管界面状态"变含糊；改为 `EnvironmentController` 只暴露 `build_*_worker()` 返回未执行的闭包，`CCCoverApp` 照旧调 `_start_worker` 派发——测试时直接同步调用闭包即可，不用起真线程，跟 `tests/test_background.py` 测 `run_in_background()` 的既有方式一致。
- `check()`/`setup()` 内重复的"读当前已过期的包"两行代码：考虑过保留重复，类比 `TaskRunner` docstring 里"不同入口的真实差异不该被强行收口"的先例——被否决，这两行是字面完全相同的调用，抽成 `_current_outdated()` 私有方法不会藏起任何差异，跟 `TaskRunner` 那条先例针对的情况（worker 生命周期怎么收尾）不是一回事。
- `_prompt_device_recheck` 标志位：考虑过维持是 `CCCoverApp` 属性、给 `EnvironmentController` 注入一对 getter/setter 回调——被否决，类比 `TaskRunner` 自己管 `cancel_requested`/`stop_triggered`（外部只读不改）的既有形状，状态应该由持有它的深模块自己拥有。
- `_confirm_force_reinstall`：考虑过跟 `CCCoverApp` 保留的 `_confirm_start` 抽一个共用的"主线程阻塞确认"小工具——被否决，两处传参形状不同（三元组 vs 单值），为 5 行代码抽公共函数不划算，等出现第三处同模式再抽；改为原样复制一份进 `EnvironmentController`。

## Consequences

- **依赖方向**：`EnvironmentController` 依赖 `TaskRunner`、`DialogHost`、`gui/environment.py` 与 `gui/device.py` 的既有纯函数；后两者不得反向依赖 `EnvironmentController`。
- **不变的部分**：`_start_worker`/`_set_busy`（起线程、切换 busy 状态）、`_recheck_device`/`_recheck_detected_device`/`_start_device_recheck`（设备重检防抖）、`device_auto`（设备是否自动挑选）都留在 `CCCoverApp`，`EnvironmentController` 对 `device_auto` 只读不改。
- **测试**：新测试加进已有的 `tests/test_environment.py`（不新开文件，跟 `tasks.py`/`dialogs.py` 的既有先例一致——一个文件装纯函数 + 一个 collaborator 类）；`tests/test_gui_smoke.py` 的 3 个冒烟测试不需要改，`CCCoverApp` 对外构造签名不变。
- **来源**：本次决策产出于架构评审第三轮（候选①），经两个独立 agent 核实过范围与推理，再经 3 轮 grilling 定下最终形状。
