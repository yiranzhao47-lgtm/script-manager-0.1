# cc_script_manager — 短剧字幕自动化流水线

> **80集全量测试通过（胜爱情战争）** · 双轨多语言翻译矩阵 · 检查点断点续传 · DeepSeek + Claude API 驱动

---

## 第一章　双模态多策略架构与统一数据契约

### 1.1　工业级五层流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│  pipeline.py  — 全局调度器 / 断点续传 / CLI                           │
└──────────┬───────────────────────────────────────────────────────────┘
           │
     Stage 0 ─ Pre-flight (LangDetector)
           │  检测视频字幕语言 vs. 配置 mode，不匹配立即中止
           │
     Stage 1+2 ─ Ingestion + Alignment
           │
     ┌─────┴─────────────────────────────────────────────┐
     │  src/ingestion/                                    │
     │    asr_runner.py   — Whisper 语音识别 (ASR)         │
     │    ocr_runner.py   — PaddleOCR 字幕帧识别 (OCR)    │
     │    ocr_dedup.py    — 跨帧去重 (cross_lang 专用)     │
     │  src/alignment/                                    │
     │    overlap_aligner.py — ASR↔OCR 时间轴合并          │
     └────────────────────────────────────────────────────┘
           │  输出: data/cache/aligned/{ep}_aligned.json
           │
     Stage 3 ─ MapReduce (人物实体提取)
           │
     ┌─────┴─────────────────────────────────────────────┐
     │  src/metadata/                                     │
     │    map_phase.py   — 逐批 LLM 抽取人物/别名          │
     │    reduce_phase.py — LLM 合并同音/OCR 手误变体       │
     └────────────────────────────────────────────────────┘
           │  输出: data/meta/meta.json
           │
     Stage 4 ─ Execution (LLM 精修)
           │
     ┌─────┴─────────────────────────────────────────────┐
     │  src/execution/                                    │
     │    episode_refiner.py — Jinja2 prompt + LLM 调用  │
     │    srt_validator.py   — 5项 SRT 格式校验            │
     └────────────────────────────────────────────────────┘
           │  输出: data/output/{ep}.srt
           │
     Stage 5 ─ Intelligence (可选，由 intelligence.drama_analysis.enabled 开关控制)
           │
     ┌─────┴─────────────────────────────────────────────┐
     │  src/intelligence/                                 │
     │    rhythm_analyzer.py — 两阶段 MapReduce 剧情拆分  │
     │    cost_auditor.py    — FinOps Token 成本核算      │
     └────────────────────────────────────────────────────┘
           │  输出: data/output/drama_structure_graph.json
           │         data/output/cost_report.json
           │
     Stage 6 ─ Translation (可选，由 intelligence.translation.enabled 开关控制)
           │
     ┌─────┴─────────────────────────────────────────────┐
     │  src/intelligence/                                 │
     │    translation_matrix.py — 双轨多语言翻译矩阵        │
     │  config/prompts/                                   │
     │    translate_en_skeleton.j2  — DeepSeek ZH→EN      │
     │    refine_en_claude.j2       — Claude EN 润色       │
     │    translate_minor_lang.j2   — DeepSeek 小语种      │
     │    localize_char_names.j2    — 人名西化一次性调用    │
     └────────────────────────────────────────────────────┘
           │  输出: data/output/translations/{ep}_{lang}.srt
           │         data/meta/char_name_en_override.json
```

| 层次 | 关键文件 | 职责 |
|------|---------|------|
| 调度 | `pipeline.py` | CLI 入口、4 阶段串联、checkpoint 驱动、tqdm 进度 |
| 摄取 | `asr_runner.py` | stable-whisper large-v3，逐集转录，GPU scope 保护 |
| 摄取 | `ocr_runner.py` | PaddleOCR，动态 ROI，动态 FPS，逐帧缓存 JSON |
| 摄取 | `ocr_dedup.py` | 相似度合并帧序列 → 字幕时间段（cross_lang 专用） |
| 对齐 | `overlap_aligner.py` | 双条件时间重叠打分，输出统一对齐 JSON；ASR 稀疏时自动升格 OCR 为 master |
| 元数据 | `map_phase.py` | 20 集/批 Map LLM，抽取人物 + 别名 + 角色描述 |
| 元数据 | `reduce_phase.py` | Reduce LLM，同音合并、OCR 手误去重、规范化 |
| 精修 | `episode_refiner.py` | 两轮 LLM + 幻觉/碎字/滞留三重防御 + 回退 SRT |
| 校验 | `srt_validator.py` | 5 项纯函数校验，fail-fast，零 I/O |
| 工具 | `gpu_manager.py` | GPU 序列化看门狗，pynvml 遥测 |
| 工具 | `checkpoint.py` | 原子状态机，BOM-free JSON |
| 工具 | `llm_client.py` | OpenAI 兼容封装，tenacity 重试，线程安全 FinOps 账本 |
| 工具 | `lang_detector.py` | 帧采样 → CJK 占比预检 |
| 工具 | `token_counter.py` | tiktoken 估算，超限警告 |
| 智能 | `rhythm_analyzer.py` | 两阶段 MapReduce 剧情因果链拆分，并行 Map + 单次 Reduce |
| 智能 | `cost_auditor.py` | FinOps 核算：按模块分类 token 用量，输出 CNY 成本表 |
| 翻译 | `translation_matrix.py` | 双轨多语言翻译矩阵：DeepSeek 骨架 + Claude 润色 + 多语言并发 |
| 工具 | `scripts/reset_checkpoint.py` | 安全回滚单集检查点状态（Python，无 BOM 风险） |

### 1.2　Strategy 模式：一个开关驱动两套行为

`config/settings.yaml` 的单字段 `pipeline.mode` 是全系统的**策略选择器**，所有组件在初始化时读取它，自动选择正确分支，不存在运行时 if-else 嗅探：

```yaml
pipeline:
  mode: "same_lang"   # ← 改为 "cross_lang" 即切换整套行为
```

| 组件 | same_lang 行为 | cross_lang 行为 |
|------|--------------|----------------|
| ASRRunner | 主轨：中文音频 → 时间轴 | 语义锚：role=semantic_anchor |
| OCRRunner | 从轨：中文字幕，fps=2，roi=[0.55,0.85] | 主轨：英文字幕，fps=5，roi=[0.80,0.95] |
| OCRDedup | 不启用 | 启用：帧序列 → 字幕时间段 |
| OverlapAligner | ASR 为 master，OCR 为 context | OCR 为 master，ASR 为 context |
| Jinja2 模板 | `refine_same_lang.j2` | `refine_cross_lang.j2` |
| GPUManager | 允许 Whisper scope | asr.role==disabled 时拒绝 Whisper scope |

### 1.3　统一数据契约：Master/Context 抽象

对齐层输出的每条记录都遵循**同一 JSON schema**，无论模式：

```json
{
  "segment_id":       "ep01_0042",
  "start":            34.92,
  "end":              35.30,
  "master_text":      "不好意思",
  "master_source":    "asr",
  "context_text":     "不好意思",
  "context_available": true,
  "ocr_candidates":   [{"text": "不好意思", "score": 0.97}]
}
```

- `master_text`：主权文本（same_lang=ASR，cross_lang=OCR）
- `context_text`：辅助文本（slave 轨最佳匹配，供 LLM 交叉验证）
- `context_available`：从轨是否有效，LLM prompt 据此决定是否引用

这一抽象将所有下游（精修器、提示模板、幻觉防御）从模式判断中**完全解耦**。两种模式共享一套精修逻辑，只有 Jinja2 模板文件不同。

---

## 第二章　核心算法规格与五大核心边界防御

### 防御 1　GPU 内存看门狗

**文件：** `src/utils/gpu_manager.py`

问题背景：Whisper (large-v3 ~3GB VRAM) 与 PaddleOCR (~2GB VRAM) 若同时驻留 GPU，在 6GB 显卡上会触发 OOM。Python GC 不保证 `__del__` 调用时序。

**三重释放保证：**
```python
# _ModelScope.__exit__ 内，scope 退出时无条件执行：
del model           # 1. 删除最后一个 Python 引用
gc.collect()        # 2. 强制 GC，确保 __del__ 触发
torch.cuda.empty_cache()  # 3. 归还 PyTorch 缓存至 CUDA 驱动
```

**序列化执行策略：**
```python
# gpu.enforce_sequential = true (默认)
# 尝试在已有模型 active 时进入第二个 scope → 立即抛出 GPUPolicyError
with gpu_manager.scope("whisper") as scope:
    model = load_model(...)
    scope.register(model)   # 必须调用，否则 scope 退出时发出 VRAM 泄漏警告
    results = model.transcribe(...)
# ← Whisper 在此处完整释放后，PaddleOCR scope 才可进入
```

**pynvml 遥测：** 每次 scope 进入/退出均打印 `used / free / total VRAM`，PRE-ALLOC 与 POST-FREE 对比可精确定位泄漏。

### 防御 2　归一化编辑距离帧聚合（cross_lang 专用）

**文件：** `src/ingestion/ocr_dedup.py`

**问题：** 5fps 采样时每条字幕产生约 8-25 帧，每帧 OCR 结果因压缩噪声、渐变帧轻微差异，直接字符串比对会误切字幕边界。

**归一化流程（先于相似度，不修改存储文本）：**
```
原始 OCR 文本
  → 剥除 HTML/格式标签           "<i>word</i>" → "word"
  → 移除音乐/特殊符号            "♪ hello ♪"  → "hello"
  → 修复行尾连字符               "contin-\nue" → "continue"
  → 折叠空白                     "hello  world" → "hello world"
  → 全小写                       "Hello" → "hello"
```

**帧决策表：**

| 帧类型 | 判定条件 | 动作 |
|--------|---------|------|
| empty | OCR 返回空 | flush 当前段（字幕消失） |
| transition | confidence < FADE_CONF_FLOOR | 仅延伸 end 时间，不更新文本（渐变帧） |
| normal，similarity ≥ 0.85 | SequenceMatcher ratio | 延伸当前段，更新平均置信度 |
| normal，similarity < 0.85 | — | flush 当前段；开启新段 |

后过滤：丢弃时长 < `min_segment_duration_sec`（0.4s）的孤立段（快速场切噪声）。

**配置（settings.yaml）：**
```yaml
cross_lang:
  dedup:
    similarity_threshold: 0.85
    min_segment_duration_sec: 0.4
    normalize_before_compare: true
```

### 防御 3　动态 ROI 空间降噪

**文件：** `src/ingestion/ocr_runner.py` + `config/settings.yaml`

**问题：** 不同短剧字幕位置差异极大（横屏 80-90%，竖屏 55-85%，顶部台词等），全帧 OCR 会抓取剧情文字、LOGO、水印等无关内容。

**ROI 机制：**
```python
# ocr_runner.py __init__ — 100% 动态，无硬编码
mode = cfg["pipeline"]["mode"]
ocr_cfg = cfg.get(mode, {}).get("ocr", {})          # 读取模式对应分支
roi_raw = ocr_cfg.get("roi", [0.80, 0.95])           # [y_start_ratio, y_end_ratio]
self._roi = (float(roi_raw[0]), float(roi_raw[1]))

# _iter_frames 内帧裁剪：
y_start = int(frame_h * roi[0])
y_end   = int(frame_h * roi[1])
cropped = frame[y_start:y_end, :]                    # 仅 OCR 字幕带
```

**换新剧只需改 yaml，不动代码：**
```yaml
same_lang:
  ocr:
    roi: [0.55, 0.85]   # 竖屏短剧，字幕在画面 55-85% 高度区间

cross_lang:
  ocr:
    roi: [0.80, 0.95]   # 横屏译制片，字幕在画面 80-95% 高度区间
```

### 防御 4　三重数据防御（集成测试验证）

**文件：** `src/execution/episode_refiner.py` — `_format_segments()` + `_merge_artifact_fragments()` + `_merge_short_fragments()`

这三道防御在 LLM 看到数据**之前**完成，属于预处理层，对 LLM 完全透明。

#### 4a　滞留 OCR 抑制（Stale OCR Guard）

**问题：** 字幕显示时长超过 ASR 片段，OCR 抓帧时旧字幕仍在屏幕上，对齐器将旧字幕内容绑定到下一句的 `context_text`，导致 LLM 用错误的 OCR 作为参考。

**算法：**
```python
recent_seen: list[tuple[str, str]] = []  # 滑动窗口：最近 5 条 (master, ctx)

for seg in segments:
    ctx = seg["context_text"]
    # 若 ctx 与最近 5 条的任意 master 或 ctx 匹配 → 判定为滞留残影
    recent_masters = [p[0] for p in recent_seen[-5:]]
    recent_ctxs    = [p[1] for p in recent_seen[-5:] if p[1]]
    if ctx in recent_masters or ctx in recent_ctxs:
        ctx_avail = False           # 对 LLM 隐藏此 context_text
    recent_seen = (recent_seen + [(master, ctx)])[-5:]
```

**效果：** ep01 由 18 处 OCR 污染降至 2 处（后 2 处为剧情合法重复）。

#### 4b　ASR 幻觉纠正（Hallucination Guard）

**问题：** Whisper 对不熟悉的场景音频（旁白、医院噪音）重复输出早期已见短语，出现"幻听"。ep05 医院场景 6 条对话全被替换为旁白句"非要把自己搞得伤痕累累才甘心"。

**算法：**
```python
phrase_counts: dict[str, int] = {}

# 条件：master_text 已出现 ≥3 次 AND 长度 ≥8 字 AND OCR 有不同内容
if (ctx_avail and ctx != master
        and len(master) >= 8
        and phrase_counts.get(master, 0) >= 3):
    effective_master = ctx      # 用 OCR 替换幻觉 ASR
```

**效果：** ep05 "孕妇发生血崩"、"血库里的血浆不够"、"需要紧急输血" 正确恢复。

#### 4c　短碎片合并（Short Fragment Merge）

**问题：** Whisper 逐字输出导致单字/双字条目时长 60-240ms（如"你"60ms → "说"120ms → "什么"240ms → "话"1720ms），`_merge_artifact_fragments` 只处理**相同文本**重复，无法合并不同文字。

**算法（迭代前向合并，直至稳定）：**
```python
def _merge_short_fragments(srt_text: str) -> str:
    # 反复扫描，将任何 duration < 300ms 的块合并入下一块
    changed = True
    while changed:
        changed = False
        new_blocks = []
        i = 0
        while i < len(blocks):
            start, end, text = blocks[i]
            dur = _ms(end) - _ms(start)
            if dur < 300 and i + 1 < len(blocks):
                n_start, n_end, n_text = blocks[i + 1]
                merged = text if n_text == text else text + n_text
                new_blocks.append([start, n_end, merged])
                i += 2
                changed = True
            else:
                new_blocks.append(blocks[i])
                i += 1
        blocks = new_blocks
```

**效果（集成测试）：**

| 集数 | 精修前 <300ms 条目 | 精修后 <300ms 条目 |
|------|-----------------|-----------------|
| ep01 | 14 | 1（末尾无下一块可合并） |
| ep02 | 2  | 0 |
| ep03 | 4  | 0 |
| ep04 | 0  | 0 |
| ep05 | 12 | 0 |

### 防御 5　双轮反射机制 + SRT 五项校验器

**文件：** `src/execution/episode_refiner.py` + `src/execution/srt_validator.py`

**SRTValidator 五项检查（fail-fast，纯函数，零 I/O）：**

| # | 检查项 | 错误示例 |
|---|-------|---------|
| 1 | 序号连续性 | `expected 3, got 5` |
| 2 | 时间戳格式 | `HH:MM:SS,mmm --> HH:MM:SS,mmm` |
| 3 | 时间顺序 | `end time is not after start time` |
| 4 | 文本非空 | `subtitle text is empty` |
| 5 | 粘连块检测 | 文本行内出现裸时间戳（缺少空行分隔符） |

**双轮反射执行流：**
```
LLM 调用 #1
  └─ SRTValidator OK  → 写入 .srt  → 完成
  └─ SRTValidator FAIL
       ↓
       构造纠错 prompt（含错误描述 + 问题片段）
       ↓
     LLM 调用 #2
       └─ OK  → 写入 .srt  → 完成
       └─ FAIL
            ↓
            Fallback 路径：
              直接从 master_text 组装合法 SRT（无 LLM）
              记录到 data/output/validation_report.json
              该集标记人工审核
```

fallback SRT 也经过 `_merge_artifact_fragments` + `_merge_short_fragments` 双重后处理，保证即使 LLM 两轮失败，输出也不存在碎片条目。

### 防御 6　Whisper 时长幻觉过滤

**文件：** `src/ingestion/asr_runner.py`

**问题：** Whisper large-v3 在静音/BGM 区段（无真实语音）会用训练数据中的高频短语填充，典型幻觉文本为"请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"。置信度反常地高（avg_probability 0.93–0.99），关键词过滤无效。

**根本特征：** 幻觉 segment 绑定到整段非语音时间，duration 异常长（10–20 s）；真实台词物理上不超过 4–5 s。

```python
_SEG_MAX_DURATION_SEC = 8.0   # 真实台词极少超过 4-5 s；8 s 以上几乎确定是幻觉

# _transcribe() 末尾 — 按时长过滤，与文本内容无关
long_segs = [s for s in segments if (s.end - s.start) > _SEG_MAX_DURATION_SEC]
if long_segs:
    segments = [s for s in segments if (s.end - s.start) <= _SEG_MAX_DURATION_SEC]
    logger.warning("[%s] %d over-long segment(s) removed ...", ...)
```

**为什么不用关键词过滤：** 时长是物理信号，与幻觉内容无关，对所有未知幻觉模式均有效；关键词表随模型版本/内容类型失效；真实台词合法含"点赞订阅"也不会被误杀。

**修复存量缓存（不重新跑 Whisper）：** 若 ASR JSON 已存在且含幻觉段，可直接过滤缓存文件后删除下游 aligned/SRT，再回滚 checkpoint 到 `ocr_done` 重跑对齐：

```python
import json, pathlib
path = pathlib.Path("data/cache/asr/28_asr.json")
data = json.loads(path.read_text(encoding="utf-8"))
data["segments"] = [s for s in data["segments"] if s["end"] - s["start"] <= 8.0]
for i, s in enumerate(data["segments"]): s["id"] = i
data["segment_count"] = len(data["segments"])
tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(path)
```

### 防御 7　ASR 稀疏救援（same_lang 专用）

**文件：** `src/alignment/overlap_aligner.py`

**问题：** 部分集数 BGM 声压偏大，Whisper 转录段数极少（实测最低 1 段/集），而 OCR 从帧中正常捕获了 19–42 条字幕块。same_lang 模式以 ASR 为 master，导致 SRT 字幕条目极度不足（文件 < 100 B）。

**触发条件：** `len(asr_segs) < asr_min_segments` 且 `len(ocr_blocks) ≥ 5`

```python
# _align_same_lang() 内，读取完 ASR/OCR 后立即判断
if (self._asr_sparse_rescue
        and len(asr_segs) < self._asr_min_segs
        and len(ocr_blocks) >= 5):
    return self._align_ocr_rescue(episode_id, asr_segs, ocr_blocks)
```

救援时 OCR block 升格为 master（`master_source="ocr"`），原 ASR 段降格为 context；输出 schema 与正常 same_lang 对齐完全一致，下游精修器无需感知。触发时日志打印 `WARNING [ep] ASR sparse (N < threshold M) — rescue: K OCR blocks promoted to master`。

**配置（settings.yaml）：**
```yaml
same_lang:
  alignment:
    asr_sparse_rescue: true       # 默认开启；设 false 可禁用救援
    asr_min_segments: 10          # 低于此段数且 OCR≥5 时触发
```

### 防御 8　OCR 救援路径同语言去重

**文件：** `src/alignment/overlap_aligner.py` — `_dedup_rescue_blocks()`

**问题：** ASR 稀疏救援（防御 7）触发时，OCR blocks 升格为 master。OCRDedup（防御 2）仅在 cross_lang 模式下运行，same_lang 模式下 OCR blocks 未经去重。字幕首帧/末帧 OCR 误读（如"干"读成"千"）会产生数个文本相似但起止时间不同的 block，全部升格后输出为 3 条重复字幕。

ep01 实测：`干了什么全忘了` 出现 3 次，blocks 19-21，持续时间各约 0.5 s。

**算法（前向合并，相似度 + 时间间隔双条件）：**

```python
_SIM_THRESH = 0.80   # SequenceMatcher ratio 阈值
_GAP_THRESH = 0.6    # 相邻 block 起止间隔（秒）

for blk in sorted_blocks:
    gap = blk["start"] - active["end"]
    if gap <= _GAP_THRESH:
        sim = SequenceMatcher(None, active["combined_text"], blk["combined_text"]).ratio()
        if sim >= _SIM_THRESH:
            active["end"] = blk["end"]       # 合并：保留更高置信度的文本
            continue
    merged.append(active); active = blk
```

**调用点：** `_align_ocr_rescue()` 入口第一行，去重后再走后续对齐逻辑，下游无感知。

### 防御 9　DeepSeek JSON 尾随逗号容错

**文件：** `src/utils/llm_client.py` — `_strip_trailing_commas()`

**问题：** DeepSeek 偶尔在 JSON 对象/数组末尾输出非法逗号（`"value",\n}`），导致 `json.loads()` 抛出 `JSONDecodeError`，整集翻译缓存写入失败。ep66、ep68 首次复现。

**修复：**

```python
_RE_TRAILING_COMMA = re.compile(r",\s*([}\]])")

def _strip_trailing_commas(s: str) -> str:
    return _RE_TRAILING_COMMA.sub(r"\1", s)

# extract_json / extract_json_array 均先尝试原始字符串，失败后尝试剥除版本
for candidate in (s, _strip_trailing_commas(s)):
    try:
        result = json.loads(candidate)
        ...
```

---

## 第三章　生产环境实战 SOP

### 3.1　环境准备

**Python 依赖（一次性安装）：**
```powershell
# 1. PyTorch (CUDA 12.1 示例 — 必须先于 PaddlePaddle 安装)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. PaddlePaddle GPU (CUDA 12.x)
pip install paddlepaddle-gpu==2.6.1.post120 `
    -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# 3. 其余依赖
pip install -r requirements.txt
```

**核心依赖速查：**

| 包 | 版本要求 | 用途 |
|----|---------|------|
| `stable-ts` | ≥2.14.0 | Whisper + 词级时间戳 |
| `paddleocr` | ≥2.7.3 | 字幕 OCR |
| `torch` | ≥2.1.0 | GPU 推理基础 |
| `openai` | ≥1.30.0 | DeepSeek API 客户端 |
| `tenacity` | ≥8.3.0 | LLM 调用重试 |
| `opencv-python` | ≥4.9.0 | 视频帧提取 |
| `pynvml` | ≥11.5.0 | GPU 内存遥测 |

**API Key 注入（每次 Shell 会话执行一次）：**
```powershell
$env:DEEPSEEK_API_KEY = "your_key_here"
```

> **安全规则：** API Key 只存在于环境变量 `DEEPSEEK_API_KEY`，绝不写入任何文件。`settings.yaml` 中的 `api_key_env` 字段存储的是变量**名称**，不是 Key 本身。

### 3.2　CLI 常用命令

```powershell
# 一键全量运行（读取 config/settings.yaml 默认配置）
python pipeline.py

# 切换到英文字幕模式（不修改 yaml，临时覆盖）
python pipeline.py --mode cross_lang

# 跳过语言预检（已手动确认视频语言，加快重启速度）
python pipeline.py --skip-preflight

# 指定外部视频目录（路径含空格时加引号）
python pipeline.py --video-dir "D:\drama\season2\raw"

# 使用自定义配置文件（多剧集项目并行管理）
python pipeline.py --config config/drama_B.yaml

# 以上选项可组合
python pipeline.py --mode cross_lang --skip-preflight --video-dir "E:\raw"
```

### 3.3　断点续传机制

checkpoint 状态机路径：

```
pending → asr_done → ocr_done → aligned → refined → complete
```

任何阶段中断后，直接重新执行 `python pipeline.py`，已完成的阶段自动跳过。

**查看当前进度：**
```powershell
cat data\cache\checkpoint.json
```

**安全回滚单集 checkpoint（推荐，避免 PowerShell BOM 污染）：**
```powershell
# 格式：python scripts/reset_checkpoint.py <episode_id> <state>
# 有效状态：pending  asr_done  ocr_done  aligned  refined  complete
python scripts/reset_checkpoint.py 01 ocr_done   # 将 ep01 回滚到 ocr_done
```

> **警告：** 不要用 PowerShell `Set-Content` / `ConvertTo-Json` 直接写入 `checkpoint.json`——PowerShell 5.1 的 UTF-8 编码默认带 BOM，Python `json.load()` 解析失败会把所有集数重置为 `pending`，触发全量重跑。始终用 `scripts/reset_checkpoint.py`。

**重跑单集精修（删除对应 SRT 文件，checkpoint 不受影响）：**
```powershell
Remove-Item data\output\ep03.srt
python pipeline.py --skip-preflight   # ep03 将重新执行 Stage 4
```

**重跑全部精修（保留 ASR/OCR/Alignment 缓存）：**
```powershell
# 删除所有 SRT，然后手动将 checkpoint.json 各集状态改为 "aligned"
# 注意：使用 UTF-8 无 BOM 编码写入 checkpoint.json
Remove-Item data\output\*.srt
# 编辑 data\cache\checkpoint.json，将所有 "complete"/"refined" 改为 "aligned"
python pipeline.py --skip-preflight
```

**完全重跑（清除所有缓存）：**
```powershell
Remove-Item -Recurse -Force data\cache\*
Remove-Item -Force data\output\*
Remove-Item -Force data\meta\*
python pipeline.py
```

**修复特定集数的 ASR 幻觉（不重新跑 Whisper）：**

适用场景：某几集 SRT 字幕极少（< 1 KB），确认 ASR JSON 中含 duration > 8 s 的幻觉段。

```powershell
# 1. 过滤存量 ASR 缓存（以 ep28 为例）
python -c "
import json, pathlib
for ep in ['28', '29']:   # 替换为实际问题集数
    p = pathlib.Path(f'data/cache/asr/{ep}_asr.json')
    d = json.loads(p.read_text(encoding='utf-8'))
    d['segments'] = [s for s in d['segments'] if s['end']-s['start'] <= 8.0]
    for i,s in enumerate(d['segments']): s['id'] = i
    d['segment_count'] = len(d['segments'])
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8'); t.replace(p)
    print(f'ep{ep}: {d[\"segment_count\"]} segments kept')
"

# 2. 删除下游缓存（aligned + SRT）
Remove-Item data\cache\aligned\28_aligned.json, data\cache\aligned\29_aligned.json
Remove-Item data\output\28.srt, data\output\29.srt

# 3. 回滚 checkpoint，让 pipeline 从对齐阶段重跑
python -c "
import json, pathlib
p = pathlib.Path('data/cache/checkpoint.json')
d = json.loads(p.read_text(encoding='utf-8'))
for ep in ['28', '29']: d['episodes'][ep] = 'ocr_done'
t = p.with_suffix('.tmp')
t.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8'); t.replace(p)
"

# 4. 重跑（只处理回滚的几集，耗时 < 1 分钟）
python pipeline.py --skip-preflight
```

> **注意：** 如果 ASR 稀疏救援（防御 7）已开启（默认），且 OCR 有足够块，pipeline 会自动用 OCR 升格为 master，无需手动干预。上述步骤仅当需要精确控制时使用。

### 3.4　换新剧配置检查清单

切换到新剧集时，按以下顺序检查 `config/settings.yaml`：

```yaml
pipeline:
  mode: "same_lang"        # ① 确认模式：中配中字 same_lang / 中配英字 cross_lang

same_lang:                 # ② 若 same_lang：
  ocr:
    roi: [0.55, 0.85]      #    调整字幕高度区间（目视视频确定字幕带位置）
    fps: 2                 #    竖屏短剧 2fps 足够；快速场切可提高至 3
  alignment:
    asr_sparse_rescue: true  #  默认开启：BGM 盖过语音导致 ASR 稀疏时自动用 OCR 救援
    asr_min_segments: 10     #  低于此段数触发救援（通常不需要调整）

cross_lang:                # ③ 若 cross_lang：
  ocr:
    roi: [0.80, 0.95]      #    英文字幕通常在画面底部 80-95%
    fps: 5                 #    英文字幕切换快，5fps 不要降

paths:
  raw_video_dir: "data/raw"  # ④ 视频目录（或用 --video-dir 临时覆盖）
```

ROI 值确定方法：用 VLC 截图，量取字幕上边距 / 画面总高度 和 字幕下边距 / 画面总高度，对应 `[y_start_ratio, y_end_ratio]`。

### 3.5　输出结果审核

**正常完成后目录结构：**
```
data/
  output/
    ep01.srt               ← 最终字幕文件
    ep02.srt
    ...
    validation_report.json ← 仅当有集数 LLM 两轮均失败时存在
  meta/
    meta.json              ← 人物名册，可手动修正后重跑 Stage 4
    meta_raw.json          ← Map 阶段原始输出（备份）
```

**validation_report.json 处理流程：**
```json
{
  "failures": [
    {
      "episode_id": "ep07",
      "reason": "LLM returned invalid SRT on both attempts",
      "fallback_used": true,
      "srt_path": "data/output/ep07.srt"
    }
  ]
}
```

1. 打开对应 `.srt`（fallback 版本已可用，内容为原始 ASR 文本）
2. 对照视频手动修正问题片段
3. 或删除该集 `.srt`，修改 `meta.json` 人物名后重新执行 `python pipeline.py --skip-preflight`

**meta.json 人工修正（人物名纠错）：**
```json
{
  "characters": {
    "霍建华": {
      "aliases": ["霍健华", "霍剑华", "货建华"],
      "description": "男主角，仲氏财团二公子"
    }
  }
}
```

修改后**只需重跑 Stage 4**（删除需要重精修的 `.srt` 文件），ASR/OCR/Alignment 缓存完整保留，无需重新跑 GPU 推理。

### 3.6　大规模运行建议（80+ 集）

- **首选 `--skip-preflight`**：第一次运行通过后，后续重试无需重跑语言检测
- **磁盘空间预估**：每集约占 ASR 缓存 5MB + OCR 缓存 10-30MB + aligned 3MB；80 集约 3-4GB
- **VRAM 监控**：日志中 `VRAM POST-FREE` 行显示每集释放后剩余显存；若连续下降，检查 pynvml 安装
- **LLM 费用预估**：DeepSeek-chat 约 1-3 元/集（8k 输入 + 2k 输出），80 集约 80-240 元
- **中断恢复**：Ctrl+C 中断后 checkpoint 已保存当前进度，直接 `python pipeline.py` 续跑

---

---

## 第四章　Stage 5 Intelligence — 剧情结构分析 + FinOps 成本核算

### 4.1　剧情因果链两阶段 MapReduce（RhythmAnalyzer）

**文件：** `src/intelligence/rhythm_analyzer.py`  
**Prompt 模板：** `config/prompts/map_conflict.j2` (Map)、`config/prompts/reduce_rhythm.j2` (Reduce)  
**开关：** `intelligence.drama_analysis.enabled: true`（默认 true；改为 false 跳过整个 Stage 5）

#### 架构：Map → Reduce

```
所有集数 SRT
     │
     ├─ (并行 4 workers) ──────────────────────────────────────────────┐
     │   _analyze_one_episode(ep_id)                                    │
     │     ├─ 缓存命中？data/cache/drama_map/{ep}_conflict.json → 直接返回│
     │     └─ 读取 SRT → 渲染 map_conflict.j2 → LLM → 解析 JSON        │
     │         输出字段：                                                 │
     │           scenes[]:                                               │
     │             scene_id          "ep01_sc_01"（零填充两位）           │
     │             location          场景地点                             │
     │             time              时间标记                             │
     │             scene_actions[]   关键行为列表                         │
     │             unresolved_debt   遗留因果债（可空）                   │
     │             pivot_signals[]   故事转折信号                         │
     │             structured_dialogues[]  逐句台词（仅存缓存）           │
     └─────────────────────────────────────────────────────────────────┘
                               │
                      汇总 conflict_map
                               │
     _build_conflict_chain()   │  ← 仅提取 scene facts，剔除 structured_dialogues
     （压缩为纯文本，控制 Reduce token 用量）
                               │
               渲染 reduce_rhythm.j2 → 单次 LLM 调用
                               │
                    Reduce 输出字段：
                      debt_chain_narrative    全剧因果债叙事综述
                      first_pinch             第一夹点（建议 ep 7-15）
                      second_pinch            第二夹点（建议 ep 20-30）
                      post_first_pinch_flow   Flow A（爆炸型）/ B（暗流型）
                      marketing_clips[]       商业剪辑推荐
                               │
               _assemble_and_write()
               输出: data/output/drama_structure_graph.json
```

#### 设计约束

- **无数字评分/百分比/强度等级**：所有分析均为纯叙事文字描述
- **Map 可断点续传**：每集单独原子写入 `drama_map/{ep}_conflict.json`，中断重跑仅补齐缺失集
- **Reduce 输入不含台词**：`_build_conflict_chain()` 刻意剔除 `structured_dialogues`，使 80+ 集系列的 Reduce prompt 保持在预算内
- **LLM 账本共享**：与 Stages 3-4 使用同一 LLMClient 实例，费用合并到 FinOps 报告

#### 配置（settings.yaml）

```yaml
intelligence:
  drama_analysis:
    enabled: true         # false = 跳过整个 Stage 5（SRT 不受影响）
    map_workers: 4        # 并行 LLM 线程数（受 API 速率限制约束）
    map_max_tokens: 6000  # 单集 Map LLM 最大输出 token
    reduce_max_tokens: 8000  # Reduce LLM 最大输出 token
    srt_char_limit: 15000    # 送入 LLM 前 SRT 截断字符上限
```

#### 输出文件结构

```json
{
  "total_episodes_analysed": 40,
  "total_episodes_in_series": 40,
  "macro_blueprint": {
    "debt_chain_narrative": "...",
    "first_pinch": { "episode_range": "ep07-ep10", "trigger": "..." },
    "second_pinch": { "episode_range": "ep22-ep26", "trigger": "..." },
    "post_first_pinch_flow": "B",
    "marketing_clips": [...]
  },
  "episode_conflicts": {
    "01": { "scenes": [...] },
    "02": { "scenes": [...] }
  }
}
```

缓存文件 `data/cache/drama_map/{ep}_conflict.json` 包含完整 `structured_dialogues`，最终 graph JSON 也包含（用于下游精读）。

---

### 4.2　FinOps Token 成本核算（CostAuditor）

**文件：** `src/intelligence/cost_auditor.py`  
**触发：** 每次 pipeline 运行结束时自动执行（无开关，轻量级）

#### 机制

LLMClient 在每次 API 调用后将 token 用量写入线程安全的内存账本（`_ledger_lock` 保护）：

```python
# llm_client.py — _record_usage()
with self._ledger_lock:
    self._ledger["total"]["input_tokens"]  += input_tokens
    self._ledger["total"]["output_tokens"] += output_tokens
    by_module = self._ledger["by_module"].setdefault(module_name, {...})
    by_module["input_tokens"]  += input_tokens
    by_module["output_tokens"] += output_tokens
```

`module_name` 由每个调用方传入，当前注册的模块：

| module_name | 来源 |
|-------------|------|
| `Subtitle_Refine` | EpisodeRefiner（Stage 4） |
| `MapReduce` | MapPhase / ReducePhase（Stage 3） |
| `Drama_Analysis` | RhythmAnalyzer Map 阶段 |
| `Drama_Blueprint` | RhythmAnalyzer Reduce 阶段 |
| `ROI_Auto_Heal` | ProjectInitializer（换新剧 ROI 推断） |

#### 终端输出样例

```
=====================================================================
  FinOps Cost Report  |  deepseek-chat  |  2026-07-08 01:44:38
=====================================================================
  Module              |  Input Tokens| Output Tokens|  Cost (CNY)
---------------------------------------------------------------------
  Subtitle_Refine     |        6,309 |           324|      0.0070
  Drama_Blueprint     |       13,474 |         1,138|      0.0158
---------------------------------------------------------------------
  TOTAL               |       19,783 |         1,462|      0.0227
=====================================================================
```

#### 定价配置（settings.yaml）

```yaml
pricing:
  deepseek-chat:
    input_cost_per_m:  1.0    # ¥/百万 input token
    output_cost_per_m: 2.0    # ¥/百万 output token
```

切换模型只需将 `execution.llm.model` 与 `pricing` 下同名 key 对应即可；模型名不匹配时成本列显示 ¥0.0000 并记录 WARNING。

#### 输出文件（data/output/cost_report.json）

```json
{
  "generated_at": "2026-07-08T01:44:38",
  "model": "deepseek-chat",
  "pricing_used": { "input_cost_per_m": 1.0, "output_cost_per_m": 2.0 },
  "by_module": [
    { "module": "Drama_Blueprint", "input_tokens": 13474, "output_tokens": 1138,
      "calls": 1, "cost_cny": 0.015812 },
    { "module": "Subtitle_Refine", "input_tokens": 6309, "output_tokens": 324,
      "calls": 4, "cost_cny": 0.006957 }
  ],
  "total": { "input_tokens": 19783, "output_tokens": 1462,
             "calls": 5, "cost_cny": 0.0227 }
}
```

---

## 第五章　Stage 6 Translation Matrix — 双轨多语言翻译

### 5.1　架构：三步双轨并发

```
精修中文 SRT (data/output/{ep}.srt)
         │
   Step 1 — DeepSeek   ZH → EN 骨架（faithful，保留所有剧情信息）
         │
         ├─ Track A (并发) — Claude    EN 骨架 → EN 润色（US English，移动端观感）
         └─ Track B (并发) — DeepSeek  EN 骨架 → th / vi / … （小语种，每种一线程）
         │
   Step 3 — 代码层校验：每条字幕 ≤3 行 / ≤40 字符/行 / ≤140 字符
               违规 → correction retry → 仍违规 → 截断兜底
         │
   Step 4 — 写入翻译缓存 + 输出 SRT
         │
   输出: data/cache/translation/{ep}_translation.json
         data/output/translations/{ep}_{lang}.srt
```

**关键设计决策：**

- **从精修 SRT 读取，不从 aligned JSON 读取**：`EpisodeRefiner` 的 `_merge_short_fragments()` 会将 <300ms 碎片合并，中文 SRT 条数 < aligned JSON 条数。翻译矩阵读精修 SRT 作为源，保证英文 SRT 与中文 SRT 条数完全一致。
- **小语种从骨架翻译，非从润色翻译**：骨架保留原文信息最完整，润色版已做美式英语适配，不宜作为泰语/越南语等的二次源。
- **缓存双重校验**：命中需同时满足 `target_languages` 覆盖 + `segment_count` 与当前精修 SRT 匹配；任一不符则自动重跑。

### 5.2　人名西化（一次性 LLM 调用）

**文件：** `_ensure_name_override()` in `translation_matrix.py`  
**缓存：** `data/meta/char_name_en_override.json`

首次运行时，对 `meta.json` 中所有人物发起一次 Claude 调用，按剧情规则分配西化英文名：

| 规则 | 示例 |
|------|------|
| 主角：剧中已有英文名 → 保留 | 陆子谦 → Ethan Lu（★ 剧中原名） |
| 主配角：WesternFirst + ChineseSurname | 林展虹 → Diana Lin，闻誉施 → Sophia Wen |
| 职位称呼：Title + Surname | 张副总 → VP Zhang，董事长 → Chairman |

**增量更新：** 后续运行检测 `meta.json` 中新出现的人物，仅对增量字符发起 LLM 调用，结果合并到现有文件，已有映射保持不变。

### 5.3　翻译质量保障

#### idx-scatter 定位

DeepSeek 偶尔合并相邻短句，返回条数 < 输入条数。旧代码顺序追加导致条目错位（内容偏移 + 末尾空白）。

**修复：** `_parse_translation_array()` 改为按 `idx` 字段散射定位：

```python
result = [{"idx": i, "text": ""} for i in range(expected_count)]
for item in arr:
    idx = int(item.get("idx", sequential_fallback))
    if 0 <= idx < expected_count:
        result[idx]["text"] = item["text"]
```

空位再由 `_fill_missing_skeletons()` 单独补发 retry 填平。

#### 两类空位补发

| 阶段 | 方法 | 触发条件 |
|------|------|---------|
| Step 1 骨架 | `_fill_missing_skeletons()` | idx-scatter 后仍有空槽 → 仅发送缺失 `(idx, zh)` 对给 DeepSeek |
| Step 2A 润色 | `_fill_missing_refined()` | Claude 返回条数不足 → 仅发送缺失骨架条目给 Claude 补润色 |

#### 提示词关键约束

`translate_en_skeleton.j2`（DeepSeek）：
- **IDIOMS / EPITHETS**：`小财迷` = "money grubber"，不得逐字直译；成语译意义；职场骂人话维持攻击强度
- **VENUE NAMES**：店名/场所名按含义意译（`晴天见` → "See You on a Sunny Day"），非拼音转写

`refine_en_claude.j2`（Claude）：
- 单行输出，禁止 `\n`（播放器自动换行）；代码层二次剥除保底
- TONE BY SCENE TYPE：职场撕逼用直白美式英语，霸总冷场短句，恋爱张力保热度
- 扩展 Chinglish 替换表："this matter / truly / let me tell you / not good" 等

#### 覆盖率报告

`run_all()` 结束时自动调用 `_log_coverage_report()`，扫描所有集数翻译缓存：

```
EN coverage gaps in 2/80 episode(s)  (3/3420 segments empty):
  [25] 1/11 EN segments empty
  [36] 2/18 EN segments empty
```
全部覆盖时打印 `EN coverage OK — all 80 episodes, 3420 segments, 0 empty`。

### 5.4　CLI 命令

```powershell
# 仅运行翻译（ASR/OCR/对齐/精修全部走缓存）
python pipeline.py --translate-only

# API Key 注入（每次 Shell 会话执行一次）
$env:DEEPSEEK_API_KEY    = "your_deepseek_key"
$env:OPENROUTER_API_KEY  = "your_openrouter_key"   # Claude 通过 OpenRouter 调用
```

**输出目录结构：**
```
data/output/translations/
  01_en.srt      ← 英文字幕（单行，Claude 润色）
  02_en.srt
  ...
data/meta/
  char_name_en_override.json   ← 人名西化映射，可手动修改后删缓存重跑
data/cache/translation/
  01_translation.json          ← 骨架 + 润色 + 小语种 完整翻译缓存
```

> **安全规则：** `DEEPSEEK_API_KEY` 与 `OPENROUTER_API_KEY` 只存在于环境变量，绝不写入任何文件。

---

*cc_script_manager — 为短剧字幕自动化而生*
