# 拆分 src/cc_cover 为 core/ 与 gui/ 两个子包，并拆分 pipeline.py

`src/cc_cover/` 17 个模块全部平铺在一层目录里，但其中 6 个（`models`/`discovery`/`formats`/`engines`/`pipeline`/`cli`）不依赖 `tkinter`、能独立跑通整条字幕补全流程，另外 9 个 + `gui.py` 只服务 Tkinter 界面——这条边界本来就是 ADR-0001 给 gui_support 拆分定单向依赖规则时隐含假设的前提，只是没有落到目录结构上。决定把前者收进 `cc_cover/core/`，后者收进 `cc_cover/gui/`。同时 `pipeline.py`（1426 行）本身也是同一种"一个文件身兼数职"的问题（原子写文件、run 摘要拼接、Options 序列化、热词提取、双引擎比对审计、字幕段校验、`SubtitlePipeline` 主类混在一起），按职责拆成 `core/pipeline/` 下的 `errors`/`io`/`options`/`hotwords`/`audit`/`validate`/`summary`/`run` 八个文件，对外仍通过 `core/pipeline/__init__.py` 一次性 re-export，调用方只多打一个 `core.` 前缀。

## Considered Options

- 只拆目录、不拆 `pipeline.py`：被否决——目录分层解决的是"哪些文件属于引擎、哪些属于界面"，`pipeline.py` 内部七八件不相关的事挤在一个文件里的问题在同一层级依然存在，两者同源（一个文件/一层目录职责过多），一起解决成本更低。
- `CCCoverApp`（`gui.py`）本身的拆分（候选列表状态/对话框/任务编排）：这次一并做，但具体接口形状（尤其是 `gui/tasks.py` 里 `self.events`/`self.process`/`self.busy` 这坨状态怎么封装）不在这条 ADR 里锁死，留给实现时用 codebase-design 的"设计两遍"决定，避免边写边改签名。
- 这次拆分实际上是把 2026-08 归档评审"候选 3"（拆 `CCCoverApp` 上帝对象）当年没做完的事补上——那次评审只做了控制流去重（建 `background.py`，把 5 处取消/报错样板收进 `run_in_background()`/`WorkerOutcome`），没有触碰"一个类身兼窗口构建/候选列表/对话框/任务编排/进度/设置/事件分发七职责"这件事本身，完成后 `CCCoverApp` 方法数从 73 涨到了 90。

## Consequences

- **依赖方向**：`core/` 内部不得 import `cc_cover.gui` 任何东西（GUI 通过 subprocess 调用 CLI，不是 import，这条边界本来就存在，只是现在有目录边界背书）；`core/pipeline/` 内部 `errors` 是唯一的公共叶子，`options`/`audit` 独立、被 `run` 依赖，`io`/`hotwords`/`validate` 依赖 `errors`，`summary` 依赖 `io`，`run` 依赖以上全部——无环。以后新代码需要反方向 import，说明分类本身出了问题，应该回来改分类，而不是加一条新的反向依赖。
- **入口点**：`pyproject.toml` 的两个 console script（`cc-cover`/`cc-cover-gui`）、`gui_launcher.py`、`src/cc_cover/__main__.py` 都要同步改 import 路径；打包配置（`CC-Cover.spec`/`CC-Cover-verify.spec`/`.github/workflows/windows-gui.yml`）不用动——三者都是整棵 `src` 树打包，跟内部目录结构无关，已核实。
- **测试**：23 个测试文件的 import 路径需要同步改（只改路径前缀，不改导入的具体名字）；`SubtitlePipeline`/`PipelineOptions` 等公开名字对使用方（`core/cli.py`、`gui/tasks.py`）保持不变，只是模块路径多了 `core.` 前缀。
