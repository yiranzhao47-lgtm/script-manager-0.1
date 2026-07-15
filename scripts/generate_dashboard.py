"""
generate_dashboard.py — cc_script_manager project dashboard

Produces data/output/dashboard.html — a self-contained HTML report showing:
  • Active tasks:   per-drama stage progress, operator flags, paywall settings
  • Archived tasks: completed dramas with no activity for 48 h
  • Cost monitor:   24 h + all-time token spend by API, with per-drama breakdown

Usage:
    python scripts/generate_dashboard.py
    (also auto-called by pipeline.py at the end of every run)
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR    = _PROJECT_ROOT / "data" / "raw"
_META_DIR   = _PROJECT_ROOT / "data" / "meta"
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "output"
_CACHE_DIR  = _PROJECT_ROOT / "data" / "cache"

# 7 named stages (index 0 = not started, 7 = fully complete)
_STAGE_NAMES = [
    "未开始",    # 0
    "ASR/OCR",  # 1
    "MapReduce", # 2
    "LLM精修",   # 3
    "术语锚定",  # 4
    "翻译",      # 5
    "翻译完成",  # 6
    "全部完成",  # 7
]


@dataclass
class DramaStatus:
    name: str
    start_time: Optional[datetime]
    stage_idx: int           # 0-7
    stage_label: str         # human-readable, may include "X/N集"
    ep_done: int
    ep_total: int
    operator_flags: list     # e.g. ["CN字幕待审核"]
    first_paywall_ep: Optional[int]
    second_paywall_ep: Optional[int]
    is_archived: bool
    last_modified: Optional[datetime]


# ── Stage / paywall detection ─────────────────────────────────────────────────

def _parse_paywall_eps(drama_name: str):
    """Return (first_pay_ep, second_pay_ep) from drama_structure_graph.json."""
    graph = _OUTPUT_DIR / drama_name / "drama_structure_graph.json"
    if not graph.exists():
        return None, None
    try:
        data = json.loads(graph.read_text(encoding="utf-8"))
        # first_pinch / second_pinch live inside macro_blueprint
        blueprint = data.get("macro_blueprint") or data

        def _ep(pinch: dict) -> Optional[int]:
            # target_scene_id: "ep08_sc_02" → episode 8 → pay ep = 9
            tsid = pinch.get("target_scene_id", "") or ""
            m = re.match(r"ep(\d+)", tsid)
            if m:
                return int(m.group(1)) + 1
            # fallback: last episode number in episode_range + 1
            ep_range = pinch.get("episode_range", "") or ""
            nums = re.findall(r"ep(\d+)", ep_range)
            if nums:
                return int(nums[-1]) + 1
            return None

        first  = blueprint.get("first_pinch")  or {}
        second = blueprint.get("second_pinch") or {}
        return _ep(first), _ep(second)
    except Exception:
        return None, None


def _max_mtime(directory: Path) -> Optional[datetime]:
    max_ts = 0.0
    try:
        for f in directory.rglob("*"):
            if f.is_file():
                ts = f.stat().st_mtime
                if ts > max_ts:
                    max_ts = ts
    except PermissionError:
        pass
    return datetime.fromtimestamp(max_ts) if max_ts > 0.0 else None


def detect_drama_status(drama_name: str) -> DramaStatus:
    cache_dir  = _CACHE_DIR / drama_name
    meta_dir   = _META_DIR  / drama_name
    output_dir = _OUTPUT_DIR / drama_name

    # ── Start time: earliest ASR file mtime ──────────────────────────────
    start_time: Optional[datetime] = None
    asr_dir = cache_dir / "asr"
    if asr_dir.exists():
        asr_files = [f for f in asr_dir.iterdir() if f.is_file()]
        if asr_files:
            start_time = datetime.fromtimestamp(
                min(f.stat().st_mtime for f in asr_files)
            )
    if start_time is None and cache_dir.exists():
        start_time = datetime.fromtimestamp(cache_dir.stat().st_mtime)

    # ── Episode counts ────────────────────────────────────────────────────
    # n_total: true total from raw video files (accurate even mid-run)
    raw_dir = _RAW_DIR / drama_name
    n_total = len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0

    aligned_dir = cache_dir / "aligned"
    n_aligned = len(list(aligned_dir.glob("*.json"))) if aligned_dir.exists() else 0

    trans_cache = cache_dir / "translation"
    n_translated = (
        len(list(trans_cache.glob("*_translation.json"))) if trans_cache.exists() else 0
    )

    # Prefer cn/ (zh drama), fall back to en/ (EN drama)
    cn_dir = output_dir / "cn"
    en_dir = output_dir / "en"
    refined_dir = cn_dir if cn_dir.exists() else en_dir
    n_refined = len(list(refined_dir.glob("*.srt"))) if refined_dir.exists() else 0

    # ── Operator flags ─────────────────────────────────────────────────────
    operator_flags = []
    review_pending = (meta_dir / "REVIEW_PENDING").exists()
    if review_pending:
        operator_flags.append("CN字幕待审核")

    # ── Stage detection ────────────────────────────────────────────────────
    # Use n_aligned as the "processed" denominator within Stage 1-3;
    # use n_total (raw) as the overall total for display.
    # IMPORTANT: check translation completeness BEFORE en_terms.json so dramas
    # processed before Stage 4.5 was introduced are not falsely stuck at stage 4.
    if not cache_dir.exists():
        stage_idx, stage_label = 0, "未开始"
    elif n_aligned == 0:
        stage_idx = 1
        stage_label = f"ASR/OCR/对齐中 0/{n_total}集"
    elif n_aligned < n_total:
        stage_idx = 1
        stage_label = f"ASR/OCR/对齐中 {n_aligned}/{n_total}集"
    elif not (meta_dir / "meta.json").exists():
        stage_idx, stage_label = 2, "MapReduce中"
    elif n_refined < n_aligned:
        stage_idx = 3
        stage_label = f"精修中 {n_refined}/{n_total}集"
    elif review_pending:
        stage_idx = 3
        stage_label = f"精修完成（{n_total}集）— 待审核"
    elif n_translated >= n_aligned:
        # Translation complete (check before en_terms so pre-Stage-4.5 dramas aren't blocked)
        creatives_dir = output_dir / "creatives"
        has_ext = creatives_dir.exists() and bool(list(creatives_dir.glob("ext_*.mp4")))
        if has_ext:
            stage_idx = 7
            stage_label = f"全部完成（{n_total}集）"
        else:
            stage_idx = 6
            stage_label = f"翻译完成（{n_total}集）"
    elif not (meta_dir / "en_terms.json").exists():
        stage_idx, stage_label = 4, "术语锚定中"
    else:
        stage_idx = 5
        stage_label = f"翻译中 {n_translated}/{n_total}集"

    # ep_done: show refinement progress through stage 4; translation from stage 5+
    ep_done = n_refined if stage_idx <= 4 else n_translated

    # ── Paywall settings ───────────────────────────────────────────────────
    first_pay_ep, second_pay_ep = _parse_paywall_eps(drama_name)

    # ── Archive: stage >= 6 and no file activity in the last 48 h ─────────
    is_archived = False
    last_modified: Optional[datetime] = None
    if stage_idx >= 6:
        for d in (output_dir, meta_dir, cache_dir):
            mt = _max_mtime(d) if d.exists() else None
            if mt and (last_modified is None or mt > last_modified):
                last_modified = mt
        if last_modified is not None:
            cutoff = datetime.now() - timedelta(hours=48)
            is_archived = last_modified < cutoff

    return DramaStatus(
        name=drama_name,
        start_time=start_time,
        stage_idx=stage_idx,
        stage_label=stage_label,
        ep_done=ep_done,
        ep_total=n_total,
        operator_flags=operator_flags,
        first_paywall_ep=first_pay_ep,
        second_paywall_ep=second_pay_ep,
        is_archived=is_archived,
        last_modified=last_modified,
    )


# ── Cost aggregation ──────────────────────────────────────────────────────────

def aggregate_costs() -> dict:
    """
    Scan all cost_history.jsonl files and return:
      {"24h": {model: {...}}, "all": {model: {...}}, "by_drama": {drama: {model: cost}}}
    """
    costs_24h: dict  = {}
    costs_all: dict  = {}
    costs_by_drama: dict = {}

    cutoff_24h = datetime.now() - timedelta(hours=24)

    def _add(store: dict, model: str, total: dict) -> None:
        if model not in store:
            store[model] = {"input_tokens": 0, "output_tokens": 0, "cost_cny": 0.0}
        store[model]["input_tokens"]  += total.get("input_tokens", 0)
        store[model]["output_tokens"] += total.get("output_tokens", 0)
        store[model]["cost_cny"]      += total.get("cost_cny", 0.0)

    if not _OUTPUT_DIR.exists():
        return {"24h": {}, "all": {}, "by_drama": {}}

    for drama_dir in _OUTPUT_DIR.iterdir():
        if not drama_dir.is_dir():
            continue
        drama_name = drama_dir.name
        drama_model_costs: dict = {}

        history_file = drama_dir / "cost_history.jsonl"
        if not history_file.exists():
            continue

        for raw_line in history_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            model = entry.get("model", "unknown")
            total = entry.get("total", {})

            _add(costs_all, model, total)

            # 24 h filter (generated_at is stored as naive local time)
            gen_str = entry.get("generated_at", "")
            try:
                gen_at = datetime.fromisoformat(gen_str)
                if gen_at >= cutoff_24h:
                    _add(costs_24h, model, total)
            except (ValueError, TypeError):
                pass

            drama_model_costs[model] = (
                drama_model_costs.get(model, 0.0) + total.get("cost_cny", 0.0)
            )

        if drama_model_costs:
            costs_by_drama[drama_name] = drama_model_costs

    return {"24h": costs_24h, "all": costs_all, "by_drama": costs_by_drama}


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _fmt_dt(dt: Optional[datetime], fmt: str = "%Y-%m-%d") -> str:
    return dt.strftime(fmt) if dt else "—"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _fmt_cny(v: float) -> str:
    return f"¥{v:.2f}"


def _model_label(model: str) -> str:
    lm = model.lower()
    if "claude" in lm:
        return "Claude"
    if "deepseek" in lm:
        return "DeepSeek"
    return model


def _progress_bar_html(stage_idx: int, has_alert: bool) -> str:
    segs = []
    for i in range(1, 8):
        if i < stage_idx or stage_idx == 7:
            css = "seg-done"
        elif i == stage_idx:
            css = "seg-alert" if has_alert else "seg-active"
        else:
            css = "seg-pending"
        segs.append(f'<div class="seg {css}" title="{_STAGE_NAMES[i]}"></div>')
    return '<div class="pbar">' + "".join(segs) + "</div>"


def _operator_badges_html(flags: list) -> str:
    if not flags:
        return '<span class="badge badge-ok">—</span>'
    return " ".join(
        f'<span class="badge badge-warn">⚠ {html.escape(f)}</span>'
        for f in flags
    )


def _drama_row_html(idx: int, s: DramaStatus) -> str:
    bar  = _progress_bar_html(s.stage_idx, bool(s.operator_flags))
    p1   = f"ep{s.first_paywall_ep}"  if s.first_paywall_ep  else "—"
    p2   = f"ep{s.second_paywall_ep}" if s.second_paywall_ep else "—"
    return (
        "<tr>"
        f"<td>{idx}</td>"
        f'<td class="drama-name">{html.escape(s.name)}</td>'
        f"<td>{_fmt_dt(s.start_time)}</td>"
        f'<td><div class="progress-cell">'
        f'{bar}'
        f'<span class="stage-label">{html.escape(s.stage_label)}</span>'
        f"</div></td>"
        f"<td>{_operator_badges_html(s.operator_flags)}</td>"
        f'<td class="paywall">{p1} / {p2}</td>'
        "</tr>"
    )


def _task_table_html(statuses: list) -> str:
    if not statuses:
        return "<p>暂无数据</p>"
    rows = "\n".join(_drama_row_html(i + 1, s) for i, s in enumerate(statuses))
    return (
        "<table>"
        "<thead><tr>"
        "<th>#</th><th>剧名</th><th>开始时间</th>"
        "<th>处理进度</th><th>运营接入</th><th>一卡 / 二卡</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


def _cost_section_html(costs: dict) -> str:
    def _summary_rows(store: dict) -> str:
        if not store:
            return "<tr><td colspan='4' class='empty'>暂无数据</td></tr>"
        rows = []
        total_cost = 0.0
        for model, v in sorted(store.items()):
            rows.append(
                "<tr>"
                f"<td>{html.escape(_model_label(model))}</td>"
                f"<td>{_fmt_tokens(v['input_tokens'])}</td>"
                f"<td>{_fmt_tokens(v['output_tokens'])}</td>"
                f"<td>{_fmt_cny(v['cost_cny'])}</td>"
                "</tr>"
            )
            total_cost += v["cost_cny"]
        rows.append(
            "<tr class='total-row'>"
            "<td>合计</td><td>—</td><td>—</td>"
            f"<td>{_fmt_cny(total_cost)}</td>"
            "</tr>"
        )
        return "\n".join(rows)

    def _cost_table(store: dict) -> str:
        return (
            '<table class="cost-table">'
            "<thead><tr>"
            "<th>API</th><th>Input</th><th>Output</th><th>费用</th>"
            "</tr></thead>"
            f"<tbody>{_summary_rows(store)}</tbody>"
            "</table>"
        )

    # Per-drama breakdown
    by_drama = costs.get("by_drama", {})
    drama_rows = []
    for drama, model_costs in sorted(by_drama.items(), key=lambda x: -sum(x[1].values())):
        total = sum(model_costs.values())
        parts = " &nbsp;|&nbsp; ".join(
            f"{_model_label(m)}: {_fmt_cny(v)}"
            for m, v in sorted(model_costs.items())
        )
        drama_rows.append(
            "<tr>"
            f"<td>{html.escape(drama)}</td>"
            f"<td>{parts}</td>"
            f"<td>{_fmt_cny(total)}</td>"
            "</tr>"
        )
    drama_detail = ""
    if drama_rows:
        drama_detail = (
            '<details class="drama-costs">'
            "<summary>按剧明细（点击展开）</summary>"
            '<table class="cost-table">'
            "<thead><tr><th>剧名</th><th>明细</th><th>小计</th></tr></thead>"
            f"<tbody>{''.join(drama_rows)}</tbody>"
            "</table>"
            "</details>"
        )

    grid = (
        '<div class="cost-grid">'
        f'<div class="cost-card"><h3>近 24 小时</h3>{_cost_table(costs.get("24h", {}))}</div>'
        f'<div class="cost-card"><h3>历史全量</h3>{_cost_table(costs.get("all", {}))}</div>'
        "</div>"
    )
    return grid + drama_detail


# ── Full HTML ─────────────────────────────────────────────────────────────────

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
/* 7-segment progress bar */
.pbar { display: flex; gap: 3px; margin-bottom: 5px; }
.seg { height: 7px; flex: 1; border-radius: 4px; }
.seg-done    { background: #22c55e; }
.seg-active  { background: #3b82f6; }
.seg-alert   { background: #f59e0b; }
.seg-pending { background: #e2e8f0; }
.progress-cell { min-width: 200px; }
.stage-label { font-size: 12px; color: #555; }
/* Badges */
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-ok   { background: #f1f5f9; color: #94a3b8; }
.badge-warn { background: #fef3c7; color: #92400e; border: 1px solid #fbbf24; }
/* Paywall */
.paywall { font-family: "SF Mono", "Cascadia Code", monospace; font-size: 13px; color: #334155; }
/* Archived section */
.archived-toggle {
  background: #f8fafc;
  border-radius: 10px;
  margin-bottom: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.archived-toggle > summary {
  padding: 13px 20px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  user-select: none;
  color: #64748b;
  border-radius: 10px;
  list-style: none;
}
.archived-toggle > summary::before { content: "▶  "; font-size: 10px; }
.archived-toggle[open] > summary::before { content: "▼  "; }
.archived-toggle[open] > summary { border-bottom: 1px solid #e2e8f0; border-radius: 10px 10px 0 0; }
.archived-inner { padding: 16px 20px; }
/* Cost section */
.cost-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px; }
.cost-card { background: #f8fafc; border-radius: 8px; padding: 14px 16px; }
.cost-table { font-size: 13px; }
.cost-table th { font-size: 11px; }
.cost-table td { padding: 7px 10px; }
.total-row td { font-weight: 700; border-top: 2px solid #e2e8f0 !important; }
td.empty { color: #94a3b8; font-style: italic; padding: 12px; }
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
.drama-costs > summary::before { content: "▶  "; font-size: 10px; }
.drama-costs[open] > summary::before { content: "▼  "; }
@media (max-width: 800px) {
  .cost-grid { grid-template-columns: 1fr; }
}
"""


def render_html(active: list, archived: list, costs: dict) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    active_table  = _task_table_html(active)
    cost_section  = _cost_section_html(costs)

    archived_section = ""
    if archived:
        archived_section = (
            '<details class="archived-toggle">'
            f"<summary>归档任务（{len(archived)}）— 已完成且 48 h 无改动</summary>"
            f'<div class="archived-inner">{_task_table_html(archived)}</div>'
            "</details>"
        )

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
  <h2>活跃任务（{len(active)}）</h2>
  {active_table}
</div>

{archived_section}

<div class="card">
  <h2>费用监控</h2>
  {cost_section}
</div>

</div>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not _RAW_DIR.exists():
        print(f"[dashboard] data/raw/ not found at {_RAW_DIR} — skipping")
        return

    drama_names = sorted(d.name for d in _RAW_DIR.iterdir() if d.is_dir())
    if not drama_names:
        print("[dashboard] No dramas found in data/raw/")
        return

    statuses = [detect_drama_status(n) for n in drama_names]

    active = sorted(
        [s for s in statuses if not s.is_archived],
        key=lambda s: s.start_time or datetime.min,
        reverse=True,
    )
    archived = sorted(
        [s for s in statuses if s.is_archived],
        key=lambda s: s.last_modified or datetime.min,
        reverse=True,
    )

    costs = aggregate_costs()
    page  = render_html(active, archived, costs)

    out = _OUTPUT_DIR / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[dashboard] Generated → {out}")


if __name__ == "__main__":
    main()
