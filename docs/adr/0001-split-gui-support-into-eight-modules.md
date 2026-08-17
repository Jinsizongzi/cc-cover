# 拆分 gui_support.py 为 8 个职责单一模块

`gui_support.py` 长到 1516 行，混杂多类互不相关的职责（架构评审报告草拟的拆分方案只归纳出 4 类：settings/commands/progress/win_native，但逐一核对代码实际调用关系后发现边界更细）。决定拆成 8 个扁平的兄弟模块：`human_readable.py`、`win_native.py`、`candidates.py`、`settings.py`（叶子，零依赖）、`progress.py`（依赖 `human_readable.py`）、`data_root.py`（依赖 `settings.py`）、`commands.py`（依赖 `data_root.py` + `settings.py`）、`storage.py`（依赖 `data_root.py` + `human_readable.py`）。`gui_support.py` 迁移完成后整体删除。

## Considered Options

- 按报告草案的 4 分法（settings/commands/progress/win_native）：被否决——`read_settings`/`write_settings` 实际同时被"数据根解析"和"GUI 偏好持久化"两条独立生命周期的逻辑复用，合并成一个 `settings.py` 会重新绑死两件节奏不同的事；候选列表统计文案（`selection_summary` 等）在草案里会被随手归进"进度"或"格式化"杂项，实际是围绕既有 Candidate 术语的独立一类，值得单独成 `candidates.py`。
- 格式化函数（`format_size`/`format_duration`）单独成层 `human_readable.py`：候选列表专用的格式化函数（`format_column_size`/`format_column_duration`/`format_estimate`）**不**并入这一层，留在 `candidates.py`——它们只服务候选列表这一个屏幕、不被其他模块复用，强行抽到 `human_readable.py` 会让一个本该零依赖的通用层长出候选列表专属逻辑。

## Consequences

- **依赖方向是硬约束**：`human_readable.py`/`win_native.py`/`candidates.py`/`settings.py` 是叶子模块，禁止反向 import 其他新模块（含彼此）；`progress.py`/`data_root.py`/`commands.py`/`storage.py` 只能沿上述方向单向依赖。以后如果新代码需要跨模块调用违反此方向，说明分类本身出了问题，应该回来改分类，而不是加一条新的反向 import。
- 迁移按依赖顺序分 8 张 ticket：`human_readable` → `win_native` → `candidates` → `settings` → `progress` → `data_root` → `commands` → `storage`（最后一张顺带删除 `gui_support.py`）。测试文件同步 1:1 新建（`test_human_readable.py` 等 8 个），旧的 5 个测试文件（`test_gui_support.py`/`test_failure_ux.py`/`test_install_disk.py`/`test_run_cleanup.py`/`test_run_feedback.py`）迁移完成后删除。
