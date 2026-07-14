"""
Process marketing clips with subtitle erasure and English subtitle burning.

For Chinese subtitle dramas (source_language="zh"):
  Each segment is processed through:
    1. ffmpeg extraction (frame-accurate, re-encoded)
    2. Subtitle erasure  (PaddleOCR detect + cv2.inpaint TELEA, ROI-only)
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
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Hard-coded subtitle style matching user spec
_SUBTITLE_STYLE = (
    "FontName=Bitter,FontSize=13,Outline=2,Shadow=2,"
    "MarginL=20,MarginR=20,MarginV=60,Alignment=2"
)

# GPU encoder — probed once at startup, used for extract and burn steps.
_nvenc_available: bool | None = None

def _nvenc() -> bool:
    global _nvenc_available
    if _nvenc_available is None:
        r = subprocess.run(
            ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.04",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True,
        )
        _nvenc_available = r.returncode == 0
        tag = "h264_nvenc" if _nvenc_available else "libx264 (nvenc not found)"
        print(f"Video encoder: {tag}")
    return _nvenc_available

def _venc_args() -> list[str]:
    if _nvenc():
        return ["-c:v", "h264_nvenc", "-cq", "18", "-preset", "p4", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"]


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


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in s).strip()[:35]


def _run(cmd: list[str], cwd: str | None = None) -> tuple[bool, str]:
    r = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=cwd,
    )
    return r.returncode == 0, r.stderr


def _extract_segment(video: Path, start: str, end: str, out: Path) -> bool:
    ok, stderr = _run([
        "ffmpeg", "-i", str(video),
        "-ss", _srt_to_ffmpeg(start),
        "-to", _srt_to_ffmpeg(end),
        *_venc_args(),
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
    # Use only the filename (no drive letter) in the subtitles filter and set
    # cwd so ffmpeg resolves it relative to the SRT directory.  This sidesteps
    # the Windows drive-letter colon ambiguity in ffmpeg's filter option parser.
    ok, stderr = _run([
        "ffmpeg", "-i", str(video),
        "-vf", f"subtitles={srt.name}:force_style='{_SUBTITLE_STYLE}'",
        *_venc_args(),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-y", str(out),
    ], cwd=str(srt.parent))
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

    t_clip_start = time.perf_counter()
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
        t_seg_start = time.perf_counter()

        # ── Step 1: extract raw segment ───────────────────────────────────
        raw_seg = tmp / f"c{cid_str}_s{j:02d}_raw.mp4"
        t0 = time.perf_counter()
        if not _extract_segment(video, start, end, raw_seg):
            print(f"     seg {j}: extraction failed — skipped")
            continue
        print(f"          extract    {_fmt(time.perf_counter() - t0)}")

        # ── Step 2: subtitle erasure ──────────────────────────────────────
        clean_seg = tmp / f"c{cid_str}_s{j:02d}_clean.mp4"
        t0 = time.perf_counter()
        erasure_ok = eraser.process_video(raw_seg, clean_seg)
        erase_elapsed = time.perf_counter() - t0
        if erasure_ok:
            print(f"          erase      {_fmt(erase_elapsed)}  ({dur/erase_elapsed:.1f}× realtime)")
        else:
            print(f"          erase      FAILED ({_fmt(erase_elapsed)}) — keeping original")
            clean_seg = raw_seg

        # ── Step 3: clip English SRT for this segment ─────────────────────
        srt_candidates = [
            en_srt_dir / f"{ep_id}_en.srt",
            en_srt_dir / f"ep{ep_id}_en.srt",
        ]
        try:
            srt_candidates.append(en_srt_dir / f"{int(ep_id):02d}_en.srt")
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
            t0 = time.perf_counter()
            if not _burn_subtitles(clean_seg, tmp_srt, final_seg):
                print(f"          burn       FAILED — using clean video")
                shutil.copy2(clean_seg, final_seg)
            else:
                print(f"          burn       {_fmt(time.perf_counter() - t0)}  ({n_entries} entries)")
        else:
            if srt_src is None:
                print(f"          burn       skipped (EN SRT not found for ep{ep_id})")
            else:
                print(f"          burn       skipped (no entries in [{start}→{end}])")
            shutil.copy2(clean_seg, final_seg)

        seg_elapsed = time.perf_counter() - t_seg_start
        print(f"          seg total  {_fmt(seg_elapsed)}")
        final_segs.append(final_seg)
        cumulative_offset += dur

    if not final_segs:
        print("     ERROR: no segments processed\n")
        return False

    # ── Concatenate all processed segments ────────────────────────────────
    t0 = time.perf_counter()
    print(f"     Concatenating {len(final_segs)} segments → {out_name}")
    if len(final_segs) == 1:
        shutil.copy2(final_segs[0], out_path)
        ok = True
    else:
        ok = _concat(final_segs, out_path)

    clip_elapsed = time.perf_counter() - t_clip_start
    if ok:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        concat_elapsed = time.perf_counter() - t0
        print(f"     OK  {out_name}  ({size_mb:.1f} MB  |  ~{cumulative_offset:.0f}s content  |  concat {_fmt(concat_elapsed)})")
        print(f"     clip total: {_fmt(clip_elapsed)}")
        if cliff_reason:
            print(f"     Hook: {cliff_reason}")
    else:
        print(f"     ERROR: concat failed  ({_fmt(clip_elapsed)} elapsed)")
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

    # Subtitle eraser: delogo backend (pure FFmpeg, no per-frame OCR)
    from src.creative.subtitle_eraser import SubtitleEraser
    eraser = SubtitleEraser(roi=roi, backend="delogo")

    errors: list[str] = []
    t_total = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        for plan in plans:
            ok = _process_zh_clip(
                plan, raw_dir, en_srt_dir, out_dir, tmp, eraser
            )
            if not ok:
                errors.append(f"clip {plan.get('clip_id', '?')}: processing failed")

    total_elapsed = time.perf_counter() - t_total
    ok_count = len(plans) - len(errors)
    print(f"Done: {ok_count}/{len(plans)} clips  |  total {_fmt(total_elapsed)}")
    print(f"Output: {out_dir}")
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
