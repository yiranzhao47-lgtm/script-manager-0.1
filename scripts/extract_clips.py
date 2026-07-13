"""
Extract marketing clips from drama_structure_graph.json using ffmpeg.

Usage:
    python scripts/extract_clips.py <drama_name>

Example:
    python scripts/extract_clips.py "dollar baby"

Reads:  data/output/<drama_name>/drama_structure_graph.json
Videos: data/raw/<drama_name>/<episode_id>.mp4
Output: data/output/<drama_name>/creatives/clip_<N>_ep<ep>_<strategy>.mp4
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _srt_to_ffmpeg(ts: str) -> str:
    """Convert SRT timestamp HH:MM:SS,mmm to ffmpeg format HH:MM:SS.mmm."""
    return ts.replace(",", ".")


def _find_video(raw_dir: Path, ep_id: str) -> Path | None:
    for candidate in (
        raw_dir / f"{ep_id}.mp4",
        raw_dir / f"{int(ep_id):02d}.mp4",
    ):
        if candidate.exists():
            return candidate
    return None


def main(drama_name: str) -> int:
    graph_path = _PROJECT_ROOT / "data" / "output" / drama_name / "drama_structure_graph.json"
    raw_dir    = _PROJECT_ROOT / "data" / "raw" / drama_name
    out_dir    = _PROJECT_ROOT / "data" / "output" / drama_name / "creatives"

    if not graph_path.exists():
        print(f"ERROR: drama_structure_graph.json not found at {graph_path}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    with graph_path.open(encoding="utf-8") as f:
        graph = json.load(f)

    clips = graph.get("macro_blueprint", {}).get("marketing_clips", [])
    if not clips:
        print("No marketing_clips found in drama_structure_graph.json")
        return 1

    print(f"Drama : {drama_name}")
    print(f"Clips : {len(clips)}")
    print(f"Output: {out_dir}")
    print()

    errors: list[str] = []
    for i, clip in enumerate(clips, 1):
        ep_id    = clip.get("episode_id", "")
        start    = clip.get("clip_start_time", "")
        end      = clip.get("clip_end_time", "")
        strategy = clip.get("mix_strategy", "clip")

        if not ep_id or not start or not end:
            msg = f"clip {i:02d}: missing episode_id or timecodes — skipped"
            print(f"[{i:02d}] SKIP  {msg}")
            errors.append(msg)
            continue

        video = _find_video(raw_dir, ep_id)
        if video is None:
            msg = f"clip {i:02d}: video not found for ep {ep_id} in {raw_dir}"
            print(f"[{i:02d}] SKIP  {msg}")
            errors.append(msg)
            continue

        out_name = f"clip_{i:02d}_ep{ep_id}_{strategy}.mp4"
        out_path = out_dir / out_name

        cmd = [
            "ffmpeg",
            "-i", str(video),
            "-ss", _srt_to_ffmpeg(start),
            "-to", _srt_to_ffmpeg(end),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-y",
            str(out_path),
        ]

        print(f"[{i:02d}] ep{ep_id}  {start} → {end}  ({strategy})")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            tail = result.stderr.strip().splitlines()
            print(f"       ERROR: {tail[-1] if tail else '(no output)'}")
            errors.append(f"clip {i:02d}: ffmpeg exit {result.returncode}")
        else:
            size_kb = out_path.stat().st_size // 1024
            print(f"       → {out_name}  ({size_kb} KB)")

    print()
    ok = len(clips) - len(errors)
    print(f"Done: {ok}/{len(clips)} clips extracted to {out_dir}")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  • {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_clips.py <drama_name>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
