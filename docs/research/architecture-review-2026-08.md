# 架构评审归档（2026-08-16）

来源：一次针对 `gui.py`、`gui_support.py`、`pipeline.py`（近 50 次提交里改动最频繁的三个文件）的架构评审，产出 5 个候选改动。评审报告原件只生成在本地临时目录，未纳入版本控制，本文归档报告结论与后续处理结果，避免报告随临时文件消失后无据可查。

## 候选与结论

| # | 候选 | 报告评级 | 处理结果 |
|---|---|---|---|
| 1 | 给管道加一个带类型的事件接口 | Strong | 已完成 — #65-#70 |
| 2 | 拆分 gui_support.py（1446 行杂物抽屉）为职责单一模块 | Worth exploring | 已完成 — #71-#79 |
| 3 | 拆掉 CCCoverApp 这个 73 方法的上帝对象 | Strong | 已完成 — #80-#87 |
| 4 | GUI 直接读 manifest.json 裸字典，越过进程边界 | Worth exploring | **不做**（见下） |
| 5 | 配置项要在 6 个文件里手工翻译一遍 | Worth exploring | **不做**（见下） |

候选 1/2/3 的实现过程见对应 GitHub issue；`CONTEXT.md` 中的 Event / WorkerOutcome / Candidate / Options / Protected Text / Stage / Phase 词条已在这三个候选的域建模阶段同步钉死。

## 候选 4、5 为什么不做

两个候选发起时都在 "Worth exploring" 档（低于候选 1/3 的 "Strong"），复核代码现状后确认报告本身把问题规模夸大了：

**候选 4**：报告称 GUI 侧存在"第二套 schema 解析器"。实测只有 3 处调用点、约 6 个字段（`progress.py` 的 `first_failed_sample`/`run_is_resumable`，加上报告未提及的 `gui.py:1315` 的 `manifest.get("candidates")`）。更重要的是 pipeline.py 里并不存在 `Manifest`/`StageReport` 类型可供"照猫画虎"——`manifest` 本身是个持续被 `update_manifest(**changes)` 增量修改的裸 `dict[str, Any]`。报告建议的方案（新增 `read_manifest()`/`read_stage_report()` 返回类型化对象）等于要先为 3 个读取点凭空发明一整套类型系统，且需处理增量 mutate 与不可变 dataclass 的天然冲突——投入产出比不划算，按报告方案做属于 over-engineering。若未来重提，应把范围收窄成给这 3 处各写一个小 helper 函数，而非引入新类型。

**候选 5**：报告称 13 个配置字段要在 7 层手工翻译。实测：`GuiOptions` 只有 3 个字段（device/hash_videos/ffmpeg），`GuiSettings` 只有 4 个（多一个 scan_path），另外 10 个字段（language/funasr_model/… 等）根本不在 GUI 中暴露，只经由 `cli.py` 里一个通用的 `DEFAULTS` 字典循环处理，本来就不是逐字段手写。真正沿多层重复的字段只有 3 个（device/hash_videos/ffmpeg）。且 `options_to_dict`/`options_from_dict` 是声明式的 1:1 镜像，换成报告建议的 `to_argv()`/`from_namespace()` 代码量基本不会减少——过不了报告自己定义的 deletion test（"内联后逻辑该以更糟的形式在别处冒出来"）。判定为报告误判，不建议做。

## 候选 2 的定位说明

候选 2 虽然完成了，但价值评估应留痕：报告自身的 deletion test 已指出"拆分不会让复杂度消失也不会转移，只是组织债"，即该候选不修复任何运行时风险，纯粹是可读性/locality 收益。与候选 1、3（两者都在实现过程中通过审查/测试挖出了真实缺陷）不是同一量级的必要性。

## 结论

评审报告在"Strong"档的判断（候选 1、3）是准确的，两者都在实现中被证明修复了真实的静默失败风险。"Worth exploring"档三个候选里，候选 2 已完成、价值有限但无害；候选 4、5 经复核后确认报告高估了问题规模，均不推进。本次架构评审到此归档完结。
