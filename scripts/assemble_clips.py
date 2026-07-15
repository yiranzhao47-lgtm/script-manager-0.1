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


def _sec_to_ffmpeg(s: float) -> str:
    """Convert seconds to HH:MM:SS.mmm for ffmpeg -to / -ss."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def _get_duration(path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _apply_tail_effect(
    clip_path: Path,
    is_cliffhanger: bool,
    freeze_sec: float,
    fade_sec: float,
    venc_args: list[str] | None = None,
) -> bool:
    """
    Post-process an assembled clip to add a polished tail, in-place.

    The last segment was already extracted with freeze_sec of extra source video,
    so the assembled clip is freeze_sec longer than the planned content.

    Cliffhanger: natural playback (actor still speaking/moving) fills those extra
    seconds.  We append a brief 0.3s video freeze and fade audio to silence.

    Non-cliffhanger: the planned content ends at (total_dur - freeze_sec).  We
    trim video there, clone-freeze the last frame for freeze_sec, gradually darken
    it to black, and fade the original ambient audio to silence.
    """
    if freeze_sec <= 0:
        return True
    total_dur = _get_duration(clip_path)
    if total_dur <= 0:
        return True

    enc = venc_args or ["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"]
    tmp = clip_path.with_stem(clip_path.stem + "_notail")
    clip_path.rename(tmp)

    if is_cliffhanger:
        brief = 0.3
        fade_start = max(total_dur + brief - fade_sec, 0.0)
        fc = (
            f"[0:v]tpad=stop_mode=clone:stop_duration={brief:.3f}[vout];"
            f"[0:a]apad=pad_dur={brief:.3f},"
            f"afade=t=out:st={fade_start:.3f}:d={fade_sec:.3f}[aout]"
        )
    else:
        plan_dur = total_dur - freeze_sec
        fade_start = max(total_dur - fade_sec, 0.0)
        fc = (
            f"[0:v]trim=end={plan_dur:.3f},setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop_duration={freeze_sec:.3f},"
            f"fade=t=out:st={plan_dur:.3f}:d={freeze_sec:.3f}[vout];"
            f"[0:a]afade=t=out:st={fade_start:.3f}:d={fade_sec:.3f}[aout]"
        )

    ok, stderr = _run([
        "ffmpeg", "-i", str(tmp),
        "-filter_complex", fc,
        "-map", "[vout]", "-map", "[aout]",
        *enc,
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-y", str(clip_path),
    ])

    if ok:
        tmp.unlink()
    else:
        tail_lines = stderr.strip().splitlines()
        print(f"         tail-effect error: {tail_lines[-1] if tail_lines else '(empty)'}")
        tmp.rename(clip_path)
    return ok


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
            # ffmpeg concat protocol: escape single quotes inside the path
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


def _assemble_one(
    plan: dict,
    raw_dir: Path,
    out_dir: Path,
    tmp: Path,
    tail_freeze_sec: float = 0.0,
    tail_audio_fade_sec: float = 1.0,
) -> bool:
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
        is_last = (j == len(segments))

        if is_last and tail_freeze_sec > 0:
            extended_end = _sec_to_ffmpeg(_ts_to_sec(end) + tail_freeze_sec)
            print(f"     seg {j}/{len(segments)}: ep{ep_id}  {start}→{end}  [{dur:.0f}s]{note_str}  +{tail_freeze_sec:.1f}s")
            ok = _extract_segment(video, start, extended_end, seg_out)
        else:
            print(f"     seg {j}/{len(segments)}: ep{ep_id}  {start}→{end}  [{dur:.0f}s]{note_str}")
            ok = _extract_segment(video, start, end, seg_out)

        if ok:
            seg_paths.append(seg_out)
        else:
            print(f"     seg {j}: extraction failed — skipped")

    if not seg_paths:
        print("     ERROR: no segments extracted\n")
        return False

    last_seg = segments[-1]
    is_cliffhanger = last_seg.get("_cliffhanger_cut", False)

    print(f"     Concatenating {len(seg_paths)} segments → {out_name}")
    if len(seg_paths) == 1:
        shutil.copy2(seg_paths[0], out_path)
        ok = True
    else:
        ok = _concat(seg_paths, out_path)

    if ok:
        if tail_freeze_sec > 0:
            kind = "cliffhanger" if is_cliffhanger else "ambient-fade"
            print(f"     Applying tail effect ({tail_freeze_sec:.1f}s, {kind}) …")
            _apply_tail_effect(out_path, is_cliffhanger, tail_freeze_sec, tail_audio_fade_sec)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        tail_note = f" + {tail_freeze_sec:.1f}s tail" if tail_freeze_sec > 0 else ""
        print(f"     OK  {out_name}  ({size_mb:.1f} MB  |  actual ~{total_sec:.0f}s{tail_note})")
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

    # Read freeze-tail config from settings.yaml
    tail_freeze_sec = 2.5
    tail_audio_fade_sec = 1.0
    cfg_path = _ROOT / "config" / "settings.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with cfg_path.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            creatives_cfg = cfg.get("intelligence", {}).get("creatives", {})
            tail_freeze_sec     = float(creatives_cfg.get("tail_freeze_sec",     tail_freeze_sec))
            tail_audio_fade_sec = float(creatives_cfg.get("tail_audio_fade_sec", tail_audio_fade_sec))
        except Exception:
            pass

    with plans_path.open(encoding="utf-8") as f:
        data = json.load(f)

    all_plans = data.get("clip_plans", [])
    plans = [p for p in all_plans if filter_ids is None or p.get("clip_id") in filter_ids]

    print(f"Drama : {drama_name}")
    print(f"Plans : {len(plans)}/{len(all_plans)} selected")
    print(f"Output: {out_dir}")
    print(f"Tail  : freeze={tail_freeze_sec:.1f}s  fade={tail_audio_fade_sec:.1f}s")
    print()

    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        for plan in plans:
            ok = _assemble_one(plan, raw_dir, out_dir, tmp,
                               tail_freeze_sec=tail_freeze_sec,
                               tail_audio_fade_sec=tail_audio_fade_sec)
            if not ok:
                errors.append(f"clip {plan.get('clip_id', '?')}: assembly failed")

    ok_count = len(plans) - len(errors)
    print(f"Done: {ok_count}/{len(plans)} clips assembled in {out_dir}")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/assemble_clips.py <drama_name> [clip_ids...]")
        sys.exit(1)
    drama = sys.argv[1]
    ids = {int(x) for x in sys.argv[2:]} if len(sys.argv) > 2 else None
    sys.exit(main(drama, ids))
