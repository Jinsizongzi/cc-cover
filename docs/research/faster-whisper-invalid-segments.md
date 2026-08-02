# faster-whisper 无效字幕段根因（研究单 #2）

## 结论

`FasterWhisperEngine.transcribe`（`src/cc_cover/engines.py`）把 `raw.start` 换算成毫秒时只做了下界保护（`max(0, ...)`），没有上界保护。当模型在音频末尾多吐出一个段、且其 `start` 略微超过 WAV 时长时，`end` 会被 clamp 到 `duration`，而 `start` 保持超界，于是 `end <= start`，`validate_segments`（`src/cc_cover/pipeline.py`）抛「引擎字幕段无效：<index>」。该报错不含视频路径、段内容等上下文，且异常会中止整批。

## 证据

1. `debug.txt`（2026-08-01）：视频 `47_skills实操_使用其他的skill工具.mp4` 在 faster-whisper 第 45/66 个时输出「错误：引擎字幕段无效：#20」，随后整批退出码 1。
2. 该运行目录最终产物（resume 后）：`CC-MISSING-00047` 的 faster-whisper 输出恰好 20 段（索引 0–19），末段是明显幻觉（韩文/西里尔字符混排，`avg_logprob=-4.18`），`end_ms` 恰好等于视频时长 551701ms。失败那次模型多吐了第 21 段（索引 20），形态符合「start 越过时长边界」。
3. 代码路径：`start = max(0, round(raw.start*1000))`；`end = min(duration_ms, max(start+1, round(raw.end*1000)))`。start 无上界，end 有上界 → `end < start` 必然可被触发。

## 最小复现

```python
duration_ms = round(551.701312 * 1000)  # 551701
# 当前换算逻辑
start = max(0, round(551.7018 * 1000))          # 551702
end = min(duration_ms, max(start + 1, round(551.7020 * 1000)))  # 551701
assert end <= start  # 触发 validate_segments 的「引擎字幕段无效」
```

修复后的换算把 start 也 clamp 到 `[0, duration_ms]`，零长度段直接跳过：

```python
start = min(duration_ms, max(0, round(551.7018 * 1000)))  # 551701
end = min(duration_ms, max(start + 1, round(551.7020 * 1000)))  # 551701
# end <= start -> 跳过该尾部段，不报错
```

## 修复建议

1. 换算时把 start clamp 到 `[0, duration_ms]`；若 `end <= start`（零长度尾部段）直接跳过该段并记日志。
2. `validate_segments` 的报错带上上下文：视频路径 / 样本号、段索引、start/end/duration、引擎名。
3. 在 `run_engine` 中把「候选级失败」与「整批失败」分开：单个候选的异常段可跳过该候选并标记 failed，而不是中止整批（联动 G1/G2）。
4. 审计侧：对 faster-whisper 段增加 `avg_logprob` / `no_speech_prob` 阈值，把低置信幻觉段标为 high_risk（联动 G4）。
