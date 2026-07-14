"""
Process marketing clips with subtitle erasure and English subtitle burning.

For Chinese subtitle dramas (source_language="zh"):
  Each segment is processed through:
    1. ffmpeg extraction (frame-accurate, re-encoded)
    2. LaMa subtitle erasure  (PaddleOCR detect + LaMa inpaint, ROI-only)
    3. English SRT window extraction + timestamp offset (srt_clipper)
    4. ffmpeg subtitle burn-in  (libass hard subtitles)
  Segments are then concatenated into the final creative via stream-copy.

For English subtitle dramas (source_language != "zh"):
  Delegates directly to assemble_clips.main() — subtitles are already English
  and burned in; no erasure or additional burning is needed.

Usage:
    python scripts/process_creatives.py <drama_name> [clip_ids...]

Examples:
    python scripts/process_creatives.py "my drama"        # all clips
    python scripts/process_creatives.py "my drama" 1 3    # clips 1 and 3

Reads:
    config/settings.yaml                         pipeline.source_language + ROI
    data/output/<drama>/clip_plans.json
    data/raw/<drama>/<episode_id>.mp4
    data/output/<drama>/translations/en/ep<id>_en.srt

Output:
    data/output/<drama>/creatives/ext_<N>_<title>_en.mp4
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent

# Hard-coded subtitle style matching user spec
_SUBTITLE_STYLE = (
    "FontName=Bitter,FontSize=13,Outline=2,Shadow=2,"
    "MarginL=20,MarginR=20,MarginV=60,Alignment=2"
)


# ══════════════════════════════════════════════════════════════════════════════
#  Shared ffmpeg helpers
# ══════════════════════════════════════════════════════════════════════════════


def _srt_to_ffmpeg(ts: str) -> str:
    return ts.replace(",", ".")


def _ts_to_sec(ts: str) -> float:
    try:
        ts = ts.replace(",", ".")
        h, m, rest = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)
    except Exception:
        return 0.0


def _find_video(raw_dir: Path, ep_id: str) -> Path | None:
    candidates = [raw_dir / f"{ep_id}.mp4"]
    try:
        candidates.append(raw_dir / f"{int(ep_id):02d}.mp4")
    except ValueError:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in s).strip()[:35]


def _run(cmd: list[str]) -> tuple[bool, str]:
    r = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return r.returncode == 0, r.stderr


def _extract_segment(video: Path, start: str, end: str, out: Path) -> bool:
    ok, stderr = _run([
        "ffmpeg", "-i", str(video),
        "-ss", _srt_to_ffmpeg(start),
        "-to", _srt_to_ffmpeg(end),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-y", str(out),
    ])
    if not ok:
        tail = stderr.strip().splitlines()
        print(f"         ffmpeg extract error: {tail[-1] if tail else '(empty)'}")
    return ok


def _burn_subtitles(video: Path, srt: Path, out: Path) -> bool:
    """Burn SRT into video with libass (hard subtitles)."""
    # Windows: escape drive-letter colon for ffmpeg filter path (C: → C\:)
    srt_posix = srt.as_posix()
    if len(srt_posix) >= 2 and srt_posix[1] == ":":
        srt_posix = srt_posix[0] + "\\:" + srt_posix[2:]

    ok, stderr = _run([
        "ffmpeg", "-i", str(video),
        "-vf", f"subtitles={srt_posix}:force_style='{_SUBTITLE_STYLE}'",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-y", str(out),
    ])
    if not ok:
        tail = stderr.strip().splitlines()
        print(f"         ffmpeg burn error: {tail[-1] if tail else '(empty)'}")
    return ok


def _concat(segment_paths: list[Path], out: Path) -> bool:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for p in segment_paths:
            escaped = p.as_posix().replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        list_path = Path(f.name)
    ok, stderr = _run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        "-movflags", "+faststart",
        "-y", str(out),
    ])
    list_path.unlink(missing_ok=True)
    if not ok:
        tail = stderr.strip().splitlines()
        print(f"         concat error: {tail[-1] if tail else '(empty)'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Per-clip processing (Chinese drama path)
# ══════════════════════════════════════════════════════════════════════════════


def _process_zh_clip(
    plan: dict,
    raw_dir: Path,
    en_srt_dir: Path,
    out_dir: Path,
    tmp: Path,
    eraser,       # SubtitleEraser instance (models loaded once, reused)
) -> bool:
    from src.creative.srt_clipper import clip_srt

    cid      = plan.get("clip_id", "?")
    title    = str(plan.get("clip_title", f"clip_{cid}"))
    segments = plan.get("segments", [])
    est_sec  = plan.get("estimated_total_sec", "?")
    cliff    = plan.get("cliffhanger_scene_id", "?")
    cliff_reason = plan.get("cliffhanger_reason", "")

    cid_str  = str(cid).zfill(2)
    out_name = f"ext_{cid_str}_{_safe_name(title)}_en.mp4"
    out_path = out_dir / out_name

    if out_path.exists():
        print(f"[{cid_str}] {title}  — already exists, skipping")
        return True

    print(f"[{cid_str}] {title}")
    print(f"     {len(segments)} segments  |  ~{est_sec}s  |  cliffhanger → {cliff}")

    final_segs: list[Path] = []
    cumulative_offset = 0.0   # seconds accumulated in the output clip so far

    for j, seg in enumerate(segments, 1):
        ep_id     = seg.get("episode_id", "")
        start     = seg.get("start_time", "")
        end       = seg.get("end_time", "")
        note      = seg.get("note", "")
        start_sec = _ts_to_sec(start)
        end_sec   = _ts_to_sec(end)
        dur       = end_sec - start_sec

        video = _find_video(raw_dir, ep_id)
        if video is None:
            print(f"     seg {j}/{len(segments)}: ep{ep_id} — video not found, skipped")
            continue

        note_str = f"  ({note})" if note else ""
        print(f"     seg {j}/{len(segments)}: ep{ep_id}  {start}→{end}  [{dur:.0f}s]{note_str}")

        # ── Step 1: extract raw segment ───────────────────────────────────
        raw_seg = tmp / f"c{cid_str}_s{j:02d}_raw.mp4"
        if not _extract_segment(video, start, end, raw_seg):
            print(f"     seg {j}: extraction failed — skipped")
            continue

        # ── Step 2: LaMa subtitle erasure ────────────────────────────────
        clean_seg = tmp / f"c{cid_str}_s{j:02d}_clean.mp4"
        print(f"          erasing subtitles...")
        erasure_ok = eraser.process_video(raw_seg, clean_seg)
        if not erasure_ok:
            print(f"     seg {j}: erasure failed — using raw (original subtitles kept)")
            clean_seg = raw_seg   # fallback: keep original

        # ── Step 3: clip English SRT for this segment ─────────────────────
        # Translation output: translations/en/ep{id}_en.srt
        srt_candidates = [
            en_srt_dir / f"ep{ep_id}_en.srt",
        ]
        try:
            srt_candidates.append(en_srt_dir / f"ep{int(ep_id):02d}_en.srt")
        except ValueError:
            pass
        srt_src = next((p for p in srt_candidates if p.exists()), None)

        srt_content = ""
        if srt_src is not None:
            srt_content = clip_srt(srt_src, start_sec, end_sec, cumulative_offset)

        # ── Step 4: burn English subtitles (or just copy if no SRT) ──────
        final_seg = tmp / f"c{cid_str}_s{j:02d}_final.mp4"
        if srt_content:
            tmp_srt = tmp / f"c{cid_str}_s{j:02d}.srt"
            tmp_srt.write_text(srt_content, encoding="utf-8")
            n_entries = srt_content.count("\n\n") + 1
            print(f"          burning {n_entries} EN subtitle entries...")
            if not _burn_subtitles(clean_seg, tmp_srt, final_seg):
                print(f"     seg {j}: burn failed — using clean video (no subtitles)")
                shutil.copy2(clean_seg, final_seg)
        else:
            if srt_src is None:
                print(f"          EN SRT not found for ep{ep_id} — no subtitles")
            else:
                print(f"          no EN subtitle entries in [{start}→{end}]")
            shutil.copy2(clean_seg, final_seg)

        final_segs.append(final_seg)
        cumulative_offset += dur

    if not final_segs:
        print("     ERROR: no segments processed\n")
        return False

    # ── Concatenate all processed segments ────────────────────────────────
    print(f"     Concatenating {len(final_segs)} segments → {out_name}")
    if len(final_segs) == 1:
        shutil.copy2(final_segs[0], out_path)
        ok = True
    else:
        ok = _concat(final_segs, out_path)

    if ok:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"     OK  {out_name}  ({size_mb:.1f} MB  |  ~{cumulative_offset:.0f}s)")
        if cliff_reason:
            print(f"     Hook: {cliff_reason}")
    else:
        print("     ERROR: concat failed")
    print()
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main(drama_name: str, filter_ids: set[int] | None) -> int:
    # ── Load config ───────────────────────────────────────────────────────
    cfg_path = _ROOT / "config" / "settings.yaml"
    source_language = "zh"
    roi = (0.78, 0.94)

    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        source_language = cfg.get("pipeline", {}).get("source_language", "zh")
        roi_list = (
            cfg.get("same_lang", {}).get("ocr", {}).get("roi", [0.78, 0.94])
        )
        roi = (float(roi_list[0]), float(roi_list[1]))

    # ── English dramas: no erasure needed — delegate to assemble_clips ────
    if source_language != "zh":
        from scripts.assemble_clips import main as _assemble
        return _assemble(drama_name, filter_ids)

    # ── Chinese drama paths ───────────────────────────────────────────────
    plans_path = _ROOT / "data" / "output" / drama_name / "clip_plans.json"
    raw_dir    = _ROOT / "data" / "raw"    / drama_name
    en_srt_dir = _ROOT / "data" / "output" / drama_name / "translations" / "en"
    out_dir    = _ROOT / "data" / "output" / drama_name / "creatives"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not plans_path.exists():
        print(
            f"ERROR: clip_plans.json not found.\n"
            f"Run: python scripts/plan_clips.py \"{drama_name}\" first."
        )
        return 1

    with plans_path.open(encoding="utf-8") as f:
        data = json.load(f)

    all_plans = data.get("clip_plans", [])
    plans = [
        p for p in all_plans
        if filter_ids is None or p.get("clip_id") in filter_ids
    ]

    print(f"Drama  : {drama_name}  (source_language={source_language})")
    print(f"Plans  : {len(plans)}/{len(all_plans)} selected")
    print(f"EN SRT : {en_srt_dir}")
    print(f"Output : {out_dir}")
    print()

    # Load LaMa + PaddleOCR once — reused across all clips
    from src.creative.subtitle_eraser import SubtitleEraser
    eraser = SubtitleEraser(roi=roi, use_gpu=True, ocr_lang="ch")

    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        for plan in plans:
            ok = _process_zh_clip(
                plan, raw_dir, en_srt_dir, out_dir, tmp, eraser
            )
            if not ok:
                errors.append(f"clip {plan.get('clip_id', '?')}: processing failed")

    ok_count = len(plans) - len(errors)
    print(f"Done: {ok_count}/{len(plans)} clips processed in {out_dir}")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/process_creatives.py <drama_name> [clip_ids...]")
        sys.exit(1)
    drama = sys.argv[1]
    ids = {int(x) for x in sys.argv[2:]} if len(sys.argv) > 2 else None
    sys.exit(main(drama, ids))
