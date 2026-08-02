# Windows 桌面分发与签名方案（研究单 #3）

## 现状

- PyInstaller `--onefile`：启动时解压到临时目录（慢），杀软误报率明显高于 onedir；未签名 → SmartScreen 显示「未知发布者」。
- 运行期要求用户系统已装 Python 3.10–3.12，再联网建 venv 安装 PyTorch / funasr / faster-whisper 并下载模型（数 GB）——对非技术用户门槛高、失败面大。

## 方案对比

| 方案 | 成本 | 优点 | 风险 / 代价 | 结论 |
|---|---|---|---|---|
| A. 维持现状（onefile + 首跑装环境） | 0 | 改动最小 | AV 误报、启动慢、SmartScreen；首跑安装易失败 | 仅短期过渡 |
| B. onedir + Inno Setup 安装器，保留首跑安装 ASR 环境 | 0 | 启动快、误报显著降低；安装器带卸载/版本信息；模型仍按需下载 | 发布物从单个 exe 变成目录+安装器，需要更新 CI | 推荐（近期） |
| C. B + 预构建 venv/运行时随安装器分发 | 0（产物巨大） | 用户零依赖、首跑即用 | 安装包 4–8GB（PyTorch+模型），发布/更新成本高 | 可选（远期离线包） |
| D. 代码签名：Azure Trusted Signing ~$9.99/月；OV ~$100–300/年；EV 已无「即时信誉」优势 | 低~中 | 消除/缓解 SmartScreen 拦截 | 需要身份验证与预算；免费阶段可先提交 Defender / VirusTotal 误报申诉 | 在 B 之后做 |

要点（来源：Microsoft Learn 代码签名选项、PyInstaller 生态实践、GlobalSign/segmentfault 2026 年签名文章）：

- onefile 的内存解压是杀软启发式扫描的高危信号；onedir + 安装器是真实应用的常见做法。
- 关闭 UPX（`--noupx`），并在 exe 中写入公司/产品/版本/版权信息，可降低误报。
- Microsoft 对非商店分发推荐的签名方式是 Azure Trusted Signing（约 9.99 美元/月，身份验证需数个工作日）；传统 OV 需要累积下载量逐步建立信誉（可能数周），EV 在 2026 年已取消「即时信誉」特权，性价比低。

## 推荐

1. 近期：方案 B —— onedir（`--noupx`、版本信息、图标）+ Inno Setup 安装器；保留首跑安装 ASR 运行环境的逻辑（模型下载不变，避免安装包膨胀到数 GB）。
2. 免费阶段：给 exe 写全版本信息；把误报样本提交 Windows Defender 与 VirusTotal。
3. 有预算后：Azure Trusted Signing 或 OV 证书签名（不建议再买 EV）。
4. 远期可选：提供「含运行时的离线安装包」作为额外大文件下载，而不是默认安装器。
