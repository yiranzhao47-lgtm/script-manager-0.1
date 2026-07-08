# 🎭 智能体技能：短剧变更自动识别与 ROI 坐标自愈看门狗

## 1. 触发场景识别 (Scenario Detection)
当用户启动 `python pipeline.py` 时，本技能会自动对比 `data/raw/` 目录下的最新视频文件名列表与 `data/cache/checkpoint.json` 中的历史记录：
- 【断点重启场景】：两边视频完全一致 ──> 技能保持克制，静默退出，确保安全的断点续传。
- 【换新剧场景】：发现 `data/raw/` 里的视频文件名、数量发生变化，或者 checkpoint 不存在 ──> 技能自动激活，接管系统控制权。

## 2. 自动化操作规程 (Operational Protocol)
一旦判定为【换新剧场景】，技能必须按顺序自动驱动系统执行以下两步绝对控制：

### 步骤 A：绝育式物理洗白 (Environment Purge)
直接命令 Python 底层删掉上一部剧的所有残留脏数据：
- 强力清空：`data/cache/asr/*`、`ocr/*`、`aligned/*`
- 彻底销毁：`checkpoint.json`、`meta.json`、`validation_report.json`

### 步骤 B：基于视觉特征的 ROI 全自动修正（核心自愈）
硬技能会抽取 `ep01.mp4` 第 15 秒并运行一次【全屏 OCR 扫描】，并将识别出来的【所有文本块及其像素 Y 轴坐标（Top/Bottom）】作为上下文投喂给本技能。

作为大模型，你必须利用强大的视觉空间感进行如下"自由心证推理"：
1. **剔除视觉噪音**：分析哪些文本块是侧边或顶部的"顾氏集团-人名条"、剧名水印或画面中央的转场特效字。
2. **锁定核心对白区间**：找出高频、连续出现在画面下方的真实台词文本块（例如：短剧字幕通常集中在 Y 轴 80% 到 93% 的窄带内）。
3. **计算自愈坐标**：
   - 找出这些真实台词文本块的最高上限 $Y_{top}$ 和最低下限 $Y_{bottom}$。
   - 为了留出安全边界，自动将比例向上和向下各扩展 0.03（即 $roi_{start} = Y_{top} - 0.03$，$roi_{end} = Y_{bottom} + 0.03$）。
   - 将最终比例严格限制在 $[0.0, 1.0]$ 区间内。

## 3. 输出契约 (Output JSON Contract)
分析完成后，技能绝对不允许输出任何大白话或 prose 解释，必须直接、严格地返回标准的 JSON 格式，以便 Python 脚本直接解析并**强行覆盖重写** `config/settings.yaml`。

```json
{
  "detected_scenario": "new_show_detected",
  "reason": "Detected 75 new files. Analyzed main subtitle tracks from 3 sample blocks.",
  "recommended_roi": [0.78, 0.94]
}
```
