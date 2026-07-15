"""
Process marketing clips — extract, concatenate, and generate companion SRT.

For Chinese subtitle dramas (source_language="zh"):
  Each segment is extracted (frame-accurate re-encode) then concatenated.
  A companion English SRT is written alongside the video, with timestamps
  adjusted to the assembled clip timeline.

For English subtitle dramas (source_language != "zh"):
  Delegates directly to assemble_clips.main().

Usage:
    python scripts/process_creatives.py <drama_name> [clip_ids...]

Examples:
    python scripts/process_creatives.py "my drama"        # all clips
    python scripts/process_creatives.py "my drama" 1 3    # clips 1 and 3

Reads:
    config/settings.yaml                         pipeline.source_language
    data/output/<drama>/clip_plans.json
    data/raw/<drama>/<episode_id>.mp4
    data/output/<drama>/translations/en/<id>_en.srt

Output:
    data/output/<drama>/creatives/ext_<N>_<title>.mp4   concatenated clip
    data/output/<drama>/creatives/ext_<N>_<title>.srt   companion EN subtitles
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

# GPU encoder — probed once at startup, used for the extract step.
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
#  Shared helpers
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


def _renumber_srt(blocks: list[str]) -> str:
    """Merge SRT block strings from multiple segments into one, re-numbered from 1."""
    entries: list[str] = []
    n = 1
    for block in blocks:
        for entry in block.strip().split("\n\n"):
            entry = entry.strip()
            if not entry:
                continue
            lines = entry.splitlines()
            if len(lines) >= 2:
                lines[0] = str(n)
                entries.append("\n".join(lines))
                n += 1
    return "\n\n".join(entries) + "\n" if entries else ""


# ══════════════════════════════════════════════════════════════════════════════
#  Per-clip processing (Chinese drama path)
# ══════════════════════════════════════════════════════════════════════════════


def _process_zh_clip(
    plan: dict,
    raw_dir: Path,
    en_srt_dir: Path,
    out_dir: Path,
    tmp: Path,
) -> bool:
    from src.creative.srt_clipper import clip_srt

    cid          = plan.get("clip_id", "?")
    title        = str(plan.get("clip_title", f"clip_{cid}"))
    segments     = plan.get("segments", [])
    est_sec      = plan.get("estimated_total_sec", "?")
    cliff        = plan.get("cliffhanger_scene_id", "?")
    cliff_reason = plan.get("cliffhanger_reason", "")

    cid_str   = str(cid).zfill(2)
    base_name = f"ext_{cid_str}_{_safe_name(title)}"
    out_path  = out_dir / f"{base_name}.mp4"
    srt_path  = out_dir / f"{base_name}.srt"

    if out_path.exists():
        print(f"[{cid_str}] {title}  — already exists, skipping")
        return True

    t_clip_start = time.perf_counter()
    print(f"[{cid_str}] {title}")
    print(f"     {len(segments)} segments  |  ~{est_sec}s  |  cliffhanger → {cliff}")

    raw_segs: list[Path] = []
    srt_blocks: list[str] = []
    cumulative_offset = 0.0

    # Read freeze-tail config (imported lazily to avoid circular import)
    from scripts.assemble_clips import _apply_tail_effect, _sec_to_ffmpeg

    cfg_path = _ROOT / "config" / "settings.yaml"
    tail_freeze_sec = 2.5
    tail_audio_fade_sec = 1.0
    if cfg_path.exists():
        try:
            creatives_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            creatives_cfg = creatives_cfg.get("intelligence", {}).get("creatives", {})
            tail_freeze_sec     = float(creatives_cfg.get("tail_freeze_sec",     tail_freeze_sec))
            tail_audio_fade_sec = float(creatives_cfg.get("tail_audio_fade_sec", tail_audio_fade_sec))
        except Exception:
            pass

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

        is_last = (j == len(segments))
        has_cliffhanger_cut = seg.get("_cliffhanger_cut", False)

        # ── Extract segment (frame-accurate re-encode) ────────────────────
        raw_seg = tmp / f"c{cid_str}_s{j:02d}.mp4"
        t0 = time.perf_counter()
        _CLIFF_SPEECH_TAIL = 0.5  # seconds past subtitle end to let actor finish speaking

        if is_last and has_cliffhanger_cut:
            # Cliffhanger: short tail so actor finishes the line; freeze+darken after.
            extended_end = _sec_to_ffmpeg(_ts_to_sec(end) + _CLIFF_SPEECH_TAIL)
            print(f"     seg {j}/{len(segments)}: ep{ep_id}  {start}→{end}  [{dur:.0f}s]{note_str}  +{_CLIFF_SPEECH_TAIL:.1f}s speech tail")
            if not _extract_segment(video, _srt_to_ffmpeg(start), extended_end, raw_seg):
                print(f"     seg {j}: extraction failed — skipped")
                continue
        elif is_last and not has_cliffhanger_cut and tail_freeze_sec > 0:
            # Non-cliffhanger: extend to capture ambient audio for the fade tail.
            extended_end = _sec_to_ffmpeg(_ts_to_sec(end) + tail_freeze_sec)
            print(f"     seg {j}/{len(segments)}: ep{ep_id}  {start}→{end}  [{dur:.0f}s]{note_str}  +{tail_freeze_sec:.1f}s ambient")
            if not _extract_segment(video, _srt_to_ffmpeg(start), extended_end, raw_seg):
                print(f"     seg {j}: extraction failed — skipped")
                continue
        else:
            print(f"     seg {j}/{len(segments)}: ep{ep_id}  {start}→{end}  [{dur:.0f}s]{note_str}")
            if not _extract_segment(video, _srt_to_ffmpeg(start), _srt_to_ffmpeg(end), raw_seg):
                print(f"     seg {j}: extraction failed — skipped")
                continue
        print(f"          extract    {_fmt(time.perf_counter() - t0)}")

        # ── Clip EN SRT for this segment, offset to clip timeline ─────────
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

        if srt_src is not None:
            block = clip_srt(srt_src, start_sec, end_sec, cumulative_offset)
            if block:
                srt_blocks.append(block)
                n_entries = block.count("\n\n") + 1
                print(f"          srt        {n_entries} entries")
            else:
                print(f"          srt        no entries in [{start}→{end}]")
        else:
            print(f"          srt        EN SRT not found for ep{ep_id}")

        raw_segs.append(raw_seg)
        cumulative_offset += dur

    if not raw_segs:
        print("     ERROR: no segments processed\n")
        return False

    # ── Concatenate video segments ────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"     Concatenating {len(raw_segs)} segments → {out_path.name}")
    if len(raw_segs) == 1:
        shutil.copy2(raw_segs[0], out_path)
        ok = True
    else:
        ok = _concat(raw_segs, out_path)

    clip_elapsed = time.perf_counter() - t_clip_start
    is_cliffhanger = segments[-1].get("_cliffhanger_cut", False)
    if ok:
        if tail_freeze_sec > 0:
            kind = "cliffhanger" if is_cliffhanger else "ambient-fade"
            print(f"     Applying tail effect ({tail_freeze_sec:.1f}s, {kind}) …")
            _apply_tail_effect(out_path, is_cliffhanger, tail_freeze_sec, tail_audio_fade_sec, _venc_args())
        size_mb = out_path.stat().st_size / (1024 * 1024)
        concat_elapsed = time.perf_counter() - t0
        tail_note = f" + {tail_freeze_sec:.1f}s tail" if tail_freeze_sec > 0 else ""
        print(f"     OK  {out_path.name}  ({size_mb:.1f} MB  |  ~{cumulative_offset:.0f}s{tail_note}  |  concat {_fmt(concat_elapsed)})")

        # ── Write companion SRT ───────────────────────────────────────────
        merged = _renumber_srt(srt_blocks)
        if merged:
            srt_path.write_text(merged, encoding="utf-8")
            n_total = merged.count("\n\n") + 1
            print(f"     SRT {srt_path.name}  ({n_total} entries)")
        else:
            print(f"     SRT skipped (no EN subtitle entries found)")

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

    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        source_language = cfg.get("pipeline", {}).get("source_language", "zh")

    # ── English dramas: delegate to assemble_clips ────────────────────────
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

    errors: list[str] = []
    t_total = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        for plan in plans:
            ok = _process_zh_clip(plan, raw_dir, en_srt_dir, out_dir, tmp)
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
