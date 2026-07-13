"""
Assemble extended marketing clips from clip_plans.json using ffmpeg.

Each clip plan is assembled in two passes:
  1. Each segment is extracted with re-encoding (frame-accurate cuts).
  2. All segments for a clip are concatenated via stream-copy (fast, lossless join).

Usage:
    python scripts/assemble_clips.py <drama_name> [clip_ids...]

Examples:
    python scripts/assemble_clips.py "dollar baby"           # all clips
    python scripts/assemble_clips.py "dollar baby" 1 3 5     # clips 1, 3, 5 only

Reads:   data/output/<drama>/clip_plans.json
Videos:  data/raw/<drama>/<episode_id>.mp4
Output:  data/output/<drama>/creatives/ext_<N>_<title>.mp4
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


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
    for candidate in (raw_dir / f"{ep_id}.mp4", raw_dir / f"{int(ep_id):02d}.mp4"):
        if candidate.exists():
            return candidate
    return None


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in s).strip()[:35]


def _run(cmd: list[str]) -> tuple[bool, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
        print(f"         ffmpeg error: {tail[-1] if tail else '(empty)'}")
    return ok


def _concat(segment_paths: list[Path], out: Path) -> bool:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for p in segment_paths:
            f.write(f"file '{p.as_posix()}'\n")
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


def _assemble_one(plan: dict, raw_dir: Path, out_dir: Path, tmp: Path) -> bool:
    cid      = plan.get("clip_id", "?")
    title    = str(plan.get("clip_title", f"clip_{cid}"))
    segments = plan.get("segments", [])
    est_sec  = plan.get("estimated_total_sec", "?")
    cliff    = plan.get("cliffhanger_scene_id", "?")
    cliff_reason = plan.get("cliffhanger_reason", "")

    cid_str  = str(cid).zfill(2)
    out_name = f"ext_{cid_str}_{_safe_name(title)}.mp4"
    out_path = out_dir / out_name

    print(f"[{cid_str}] {title}")
    print(f"     {len(segments)} segments  |  ~{est_sec}s  |  cliffhanger → {cliff}")

    seg_paths: list[Path] = []
    total_sec = 0.0

    for j, seg in enumerate(segments, 1):
        ep_id = seg.get("episode_id", "")
        start = seg.get("start_time", "")
        end   = seg.get("end_time", "")
        note  = seg.get("note", "")
        dur   = _ts_to_sec(end) - _ts_to_sec(start)
        total_sec += max(dur, 0.0)

        video = _find_video(raw_dir, ep_id)
        if video is None:
            print(f"     seg {j}/{len(segments)}: ep{ep_id} — video not found, skipped")
            continue

        seg_out = tmp / f"c{cid_str}_s{j:02d}.mp4"
        note_str = f"  ({note})" if note else ""
        print(f"     seg {j}/{len(segments)}: ep{ep_id}  {start}→{end}  [{dur:.0f}s]{note_str}")

        if _extract_segment(video, start, end, seg_out):
            seg_paths.append(seg_out)
        else:
            print(f"     seg {j}: extraction failed — skipped")

    if not seg_paths:
        print("     ERROR: no segments extracted\n")
        return False

    print(f"     Concatenating {len(seg_paths)} segments → {out_name}")
    if len(seg_paths) == 1:
        shutil.copy2(seg_paths[0], out_path)
        ok = True
    else:
        ok = _concat(seg_paths, out_path)

    if ok:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"     OK  {out_name}  ({size_mb:.1f} MB  |  actual ~{total_sec:.0f}s)")
        print(f"     Hook: {cliff_reason}")
    else:
        print("     ERROR: concat failed")
    print()
    return ok


def main(drama_name: str, filter_ids: set[int] | None) -> int:
    plans_path = _ROOT / "data" / "output" / drama_name / "clip_plans.json"
    raw_dir    = _ROOT / "data" / "raw" / drama_name
    out_dir    = _ROOT / "data" / "output" / drama_name / "creatives"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not plans_path.exists():
        print(f"ERROR: clip_plans.json not found.\nRun: python scripts/plan_clips.py \"{drama_name}\" first.")
        return 1

    with plans_path.open(encoding="utf-8") as f:
        data = json.load(f)

    all_plans = data.get("clip_plans", [])
    plans = [p for p in all_plans if filter_ids is None or p.get("clip_id") in filter_ids]

    print(f"Drama : {drama_name}")
    print(f"Plans : {len(plans)}/{len(all_plans)} selected")
    print(f"Output: {out_dir}")
    print()

    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        for plan in plans:
            ok = _assemble_one(plan, raw_dir, out_dir, tmp)
            if not ok:
                errors.append(f"clip {plan.get('clip_id', '?')}: assembly failed")

    ok_count = len(plans) - len(errors)
    print(f"Done: {ok_count}/{len(plans)} clips assembled in {out_dir}")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  • {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/assemble_clips.py <drama_name> [clip_ids...]")
        sys.exit(1)
    drama = sys.argv[1]
    ids = {int(x) for x in sys.argv[2:]} if len(sys.argv) > 2 else None
    sys.exit(main(drama, ids))
