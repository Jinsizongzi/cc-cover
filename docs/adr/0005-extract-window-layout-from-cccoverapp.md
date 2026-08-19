# 从 CCCoverApp 抽出界面搭建代码

`CCCoverApp`（`gui/app.py`）里 `_configure_window`/`_configure_styles`/`_build_interface`/`_panel`/`_build_work_tab`/`_build_text_tab`/`_build_log_tab` 共 7 个方法、约 350 行，是纯 Tk 控件树搭建，没有分支逻辑，混在真正有判断逻辑的方法中间。决定挪进新文件 `gui/layout.py`，作为不隶属任何类的模块级函数。这个候选评级是 Worth exploring 而非 Strong（架构评审第三轮候选④）：不带来可测性收益——挪不挪，`_build_work_tab` 建出来的控件树都只能靠 `tests/test_gui_smoke.py` 的真实 Tk 实例冒烟检查；纯粹是 locality 收益，读业务逻辑不用跳过控件搭建代码。

`build_interface`/`build_work_tab`/`build_log_tab`/`configure_window`/`configure_styles` 这 5 个函数需要在 `CCCoverApp` 实例上直接设置约 25 个控件属性（`app.progress`/`app.setup_button`/`app.candidate_tree` 等，后续被 `_set_busy`、`DialogHost`/`EnvironmentController`/`ProgressPresenter` 的构造调用等大量读取），因此接收 `app: Any` 并直接读写 `app.xxx`，而不是返回一个打包好的控件容器——25 个字段的容器类要么之后还是得在 `__init__` 里逐个解包挂回 `self`（没有省掉任何代码，纯粹多一层），要么把这 25 个字段的读取点也全部从 `self.xxx` 改成 `self.widgets.xxx`，牵动 `dialogs.py`/`environment.py`/`_set_busy` 等一大片调用点——对一个"不带可测性收益"的候选来说，代价和收益不成比例。`panel`/`build_text_tab` 这两个函数不碰 `app` 的任何属性，只用传入的 `parent`/`content` 参数，因此不接收 `app`。

`app: Any` 而非 `app: CCCoverApp`：`layout.py` 反过来 import `app.py` 的 `CCCoverApp` 做类型注解会成环（`app.py` 要 import `layout.py` 里的构建函数）。项目未接入 mypy/pyright（已核实 `pyproject.toml` 无相关配置），这类注解不会被任何工具验证，加一层 `TYPE_CHECKING` 守卫的字符串类型只是仪式，不做。

## Consequences

- **依赖方向**：`layout.py` 依赖 `cc_cover.gui.content`（`FEATURE_TEXT`/`GUIDE_TEXT`）；不依赖、也不能依赖 `cc_cover.gui.app`。
- **随迁移的常量**：`BACKGROUND`/`PANEL`/`INK`/`MUTED`/`ERROR` 这 5 个颜色常量随控件搭建代码一起搬进 `layout.py`（原来只被这 7 个方法引用）；`app.py` 里另外 4 个颜色常量（`PRIMARY`/`PRIMARY_DARK`/`SUCCESS`/`WARNING`）在本次迁移前就已经没有任何引用点，属于既有的死代码，不在这次候选范围内，原样保留未动。
- **测试**：不新增测试——这个候选本身不改变任何控件的搭建结果，`tests/test_gui_smoke.py` 的既有冒烟测试原样覆盖。
- **来源**：本次决策产出于架构评审第三轮（候选③，评级 Worth exploring）。
