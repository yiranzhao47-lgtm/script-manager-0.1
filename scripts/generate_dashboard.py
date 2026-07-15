"""
Generate a self-contained HTML dashboard at data/output/dashboard.html.

Usage:
    python scripts/generate_dashboard.py

Reads:
    data/raw/<drama>/          — discovers all drama names
    data/cache/<drama>/        — stage detection
    data/meta/<drama>/         — meta.json, en_terms.json
    data/output/<drama>/       — SRT counts, creatives, cost_history.jsonl,
                                 drama_structure_graph.json
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR    = _ROOT / "data" / "raw"
_META_DIR   = _ROOT / "data" / "meta"
_OUTPUT_DIR = _ROOT / "data" / "output"
_CACHE_DIR  = _ROOT / "data" / "cache"

STAGE_LABELS = [
    "未开始",
    "ASR/OCR/对齐中",
    "MapReduce中",
    "LLM精修中",
    "术语锚定中",
    "翻译中",
    "翻译完成",
    "全部完成",
]


# ── stage detection ───────────────────────────────────────────────────────────

def _count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(pattern))


def detect_stage(drama: str) -> tuple[int, str, int, int]:
    """Return (stage_idx, stage_label, ep_done, ep_total)."""
    cache_dir  = _CACHE_DIR  / drama
    meta_dir   = _META_DIR   / drama
    output_dir = _OUTPUT_DIR / drama

    if not cache_dir.exists():
        return 0, STAGE_LABELS[0], 0, 0

    n_total = _count_files(cache_dir / "aligned", "*.json")

    if n_total == 0:
        asr_done = _count_files(cache_dir / "asr", "*.json")
        label = f"{STAGE_LABELS[1]} {asr_done}/??集" if asr_done else STAGE_LABELS[1]
        return 1, label, asr_done, 0

    if not (meta_dir / "meta.json").exists():
        asr_done = _count_files(cache_dir / "asr", "*.json")
        label = f"{STAGE_LABELS[1]} {asr_done}/{n_total}集"
        return 1, label, asr_done, n_total

    # Refined SRT count — check cn/ then en/
    n_refined = 0
    for sub in ("cn", "en"):
        srt_dir = output_dir / sub
        if srt_dir.exists():
            n_refined = _count_files(srt_dir, "*.srt")
            break

    if n_refined < n_total:
        label = f"{STAGE_LABELS[3]} {n_refined}/{n_total}集"
        return 3, label, n_refined, n_total

    if not (meta_dir / "en_terms.json").exists():
        return 4, STAGE_LABELS[4], n_total, n_total

    n_translated = _count_files(cache_dir / "translation", "*_translation.json")
    if n_translated < n_total:
        label = f"{STAGE_LABELS[5]} {n_translated}/{n_total}集"
        return 5, label, n_translated, n_total

    if _count_files(output_dir / "creatives", "ext_*.mp4") > 0:
        label = f"{STAGE_LABELS[7]}（{n_total}集）"
        return 7, label, n_total, n_total

    label = f"{STAGE_LABELS[6]}（{n_total}集）"
    return 6, label, n_total, n_total


def _drama_start_time(drama: str) -> Optional[datetime]:
    asr_dir = _CACHE_DIR / drama / "asr"
    if asr_dir.exists():
        mtimes = [p.stat().st_mtime for p in asr_dir.iterdir() if p.is_file()]
        if mtimes:
            return datetime.fromtimestamp(min(mtimes))
    cache_dir = _CACHE_DIR / drama
    if cache_dir.exists():
        return datetime.fromtimestamp(cache_dir.stat().st_mtime)
    return None


# ── paywall detection ─────────────────────────────────────────────────────────

def _parse_paywall(drama: str) -> tuple[Optional[int], Optional[int]]:
    graph = _OUTPUT_DIR / drama / "drama_structure_graph.json"
    if not graph.exists():
        return None, None
    try:
        d = json.loads(graph.read_text(encoding="utf-8"))
        mb = d.get("macro_blueprint", {})
        fp_scene = mb.get("first_pinch", {}).get("target_scene_id", "")
        sp_scene = mb.get("second_pinch", {}).get("target_scene_id", "")
        m1 = re.search(r"ep(\d+)", fp_scene)
        m2 = re.search(r"ep(\d+)", sp_scene)
        pay1 = int(m1.group(1)) + 1 if m1 else None
        pay2 = int(m2.group(1)) + 1 if m2 else None
        return pay1, pay2
    except Exception:
        return None, None


# ── cost aggregation ──────────────────────────────────────────────────────────

def aggregate_costs() -> dict:
    now = datetime.now()
    cutoff_24h = now.timestamp() - 86400

    costs_24h: dict[str, dict] = {}
    costs_all: dict[str, dict] = {}
    costs_by_drama: dict[str, dict[str, float]] = {}

    for jsonl in _OUTPUT_DIR.glob("*/cost_history.jsonl"):
        drama = jsonl.parent.name
        for raw in jsonl.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            model = rec.get("model", "unknown")
            total = rec.get("total", {})
            inp  = total.get("input_tokens", 0)
            out  = total.get("output_tokens", 0)
            cost = total.get("cost_cny", 0.0)
            ts_str = rec.get("generated_at", "")

            key = "Claude" if "claude" in model.lower() else "DeepSeek"

            if key not in costs_all:
                costs_all[key] = {"input": 0, "output": 0, "cost": 0.0}
            costs_all[key]["input"]  += inp
            costs_all[key]["output"] += out
            costs_all[key]["cost"]   += cost

            try:
                ts = datetime.fromisoformat(ts_str).timestamp()
                if ts >= cutoff_24h:
                    if key not in costs_24h:
                        costs_24h[key] = {"input": 0, "output": 0, "cost": 0.0}
                    costs_24h[key]["input"]  += inp
                    costs_24h[key]["output"] += out
                    costs_24h[key]["cost"]   += cost
            except Exception:
                pass

            if drama not in costs_by_drama:
                costs_by_drama[drama] = {}
            if key not in costs_by_drama[drama]:
                costs_by_drama[drama][key] = 0.0
            costs_by_drama[drama][key] += cost

    return {"24h": costs_24h, "all": costs_all, "by_drama": costs_by_drama}


# ── HTML rendering ────────────────────────────────────────────────────────────

_CSS = """
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 14px;
  background: #f0f2f5;
  color: #1a1a2e;
  margin: 0;
  padding: 0;
}
.container { max-width: 1440px; margin: 0 auto; padding: 20px; }
header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #fff;
  padding: 18px 24px;
  margin-bottom: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0,0,0,.18);
}
header h1 { margin: 0 0 4px; font-size: 20px; font-weight: 700; letter-spacing: -.02em; }
header p  { margin: 0; font-size: 12px; color: #94a3b8; }
header code { font-family: monospace; background: rgba(255,255,255,.1); padding: 1px 5px; border-radius: 3px; }
.card {
  background: #fff;
  border-radius: 10px;
  padding: 16px 20px;
  margin-bottom: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
h2 { font-size: 15px; font-weight: 700; margin: 0 0 14px; color: #1a1a2e; }
h3 { font-size: 13px; font-weight: 600; margin: 0 0 10px; color: #475569; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid #f0f2f5; vertical-align: middle; }
th {
  background: #f8fafc;
  font-weight: 600;
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: .05em;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: #fafbff; }
.drama-name { font-weight: 500; max-width: 240px; word-break: break-word; }
.pbar { display: flex; gap: 3px; margin-bottom: 5px; }
.seg { height: 7px; flex: 1; border-radius: 4px; }
.seg-done    { background: #22c55e; }
.seg-active  { background: #3b82f6; }
.seg-pending { background: #e2e8f0; }
.progress-cell { min-width: 200px; }
.stage-label { font-size: 12px; color: #555; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-ok   { background: #f1f5f9; color: #94a3b8; }
.badge-warn { background: #fef3c7; color: #92400e; border: 1px solid #fbbf24; }
.paywall { font-family: "SF Mono", "Cascadia Code", monospace; font-size: 13px; color: #334155; }
.cost-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px; }
.cost-card { background: #f8fafc; border-radius: 8px; padding: 14px 16px; }
.cost-table { font-size: 13px; }
.cost-table th { font-size: 11px; }
.cost-table td { padding: 7px 10px; }
.total-row td { font-weight: 700; border-top: 2px solid #e2e8f0 !important; }
.drama-costs {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  margin-top: 4px;
  overflow: hidden;
}
.drama-costs > summary {
  padding: 9px 14px;
  cursor: pointer;
  font-size: 13px;
  color: #475569;
  font-weight: 600;
  background: #f8fafc;
  user-select: none;
  list-style: none;
}
.drama-costs > summary::before { content: "\\25B6  "; font-size: 10px; }
.drama-costs[open] > summary::before { content: "\\25BC  "; }
@media (max-width: 800px) {
  .cost-grid { grid-template-columns: 1fr; }
}
"""

_SEG_LABELS = ["ASR/OCR", "MapReduce", "LLM精修", "术语锚定", "翻译", "翻译完成", "全部完成"]


def _progress_bar(stage_idx: int) -> str:
    segs = []
    for i, label in enumerate(_SEG_LABELS):
        seg_stage = i + 1
        if seg_stage < stage_idx:
            cls = "seg-done"
        elif seg_stage == stage_idx:
            cls = "seg-active"
        else:
            cls = "seg-pending"
        segs.append(f'<div class="seg {cls}" title="{label}"></div>')
    return '<div class="pbar">' + "".join(segs) + "</div>"


def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("'", "&#x27;").replace('"', "&quot;"))


def _drama_row(idx: int, drama: str) -> str:
    stage_idx, stage_label, _, _ = detect_stage(drama)
    start_dt  = _drama_start_time(drama)
    start_str = start_dt.strftime("%Y-%m-%d") if start_dt else "—"
    pay1, pay2 = _parse_paywall(drama)
    paywall_str = f"ep{pay1} / ep{pay2}" if (pay1 and pay2) else "— / —"
    pbar = _progress_bar(stage_idx)
    return (
        f"<tr><td>{idx}</td>"
        f'<td class="drama-name">{_esc(drama)}</td>'
        f"<td>{start_str}</td>"
        f'<td><div class="progress-cell">{pbar}'
        f'<span class="stage-label">{_esc(stage_label)}</span></div></td>'
        f'<td><span class="badge badge-ok">—</span></td>'
        f'<td class="paywall">{paywall_str}</td></tr>'
    )


def _cost_summary_table(bucket: dict) -> str:
    rows = ""
    total_cost = 0.0
    for key in ("Claude", "DeepSeek"):
        if key not in bucket:
            continue
        b = bucket[key]
        cost = b["cost"]
        total_cost += cost
        rows += (
            f"<tr><td>{key}</td>"
            f"<td>{_fmt_tok(b['input'])}</td>"
            f"<td>{_fmt_tok(b['output'])}</td>"
            f"<td>¥{cost:.2f}</td></tr>\n"
        )
    rows += f"<tr class='total-row'><td>合计</td><td>—</td><td>—</td><td>¥{total_cost:.2f}</td></tr>"
    return (
        "<table class='cost-table'><thead><tr>"
        "<th>API</th><th>Input</th><th>Output</th><th>费用</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _drama_costs_table(costs_by_drama: dict) -> str:
    rows = ""
    for drama, models in sorted(costs_by_drama.items(),
                                 key=lambda kv: -sum(kv[1].values())):
        total = sum(models.values())
        parts = " &nbsp;|&nbsp; ".join(
            f"{k}: ¥{v:.2f}" for k, v in sorted(models.items())
        )
        rows += (
            f"<tr><td>{_esc(drama)}</td>"
            f"<td>{parts}</td>"
            f"<td>¥{total:.2f}</td></tr>"
        )
    return (
        "<table class='cost-table'><thead><tr>"
        "<th>剧名</th><th>明细</th><th>小计</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def render_html(dramas: list[str], costs: dict) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _sort_key(d: str) -> tuple:
        idx, _, _, _ = detect_stage(d)
        t = _drama_start_time(d)
        return (idx >= 7, -(t.timestamp() if t else 0))

    dramas_sorted = sorted(dramas, key=_sort_key)
    rows_html = "\n".join(_drama_row(i + 1, d) for i, d in enumerate(dramas_sorted))

    thead = (
        "<thead><tr>"
        "<th>#</th><th>剧名</th><th>开始时间</th><th>处理进度</th>"
        "<th>运营接入</th><th>一卡 / 二卡</th>"
        "</tr></thead>"
    )
    c24h = _cost_summary_table(costs["24h"])
    call = _cost_summary_table(costs["all"])
    cdr  = _drama_costs_table(costs["by_drama"])

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cc_script_manager 运营仪表盘</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">

<header>
  <h1>cc_script_manager 运营仪表盘</h1>
  <p>最后更新: {now_str} &nbsp;|&nbsp; 刷新: <code>python scripts/generate_dashboard.py</code></p>
</header>

<div class="card">
  <h2>活跃任务（{len(dramas_sorted)}）</h2>
  <table>{thead}<tbody>{rows_html}</tbody></table>
</div>

<div class="card">
  <h2>费用监控</h2>
  <div class="cost-grid">
    <div class="cost-card"><h3>近 24 小时</h3>{c24h}</div>
    <div class="cost-card"><h3>历史全量</h3>{call}</div>
  </div>
  <details class="drama-costs">
    <summary>按剧明细（点击展开）</summary>
    {cdr}
  </details>
</div>

</div>
</body>
</html>"""


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    dramas = sorted(p.name for p in _RAW_DIR.iterdir() if p.is_dir())
    if not dramas:
        print("No dramas found in data/raw/")
        return
    costs    = aggregate_costs()
    html     = render_html(dramas, costs)
    out_path = _OUTPUT_DIR / "dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {out_path}")


if __name__ == "__main__":
    main()
