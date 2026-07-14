"""
SRT clip extraction with timestamp windowing and offset.

Given a source SRT file and a time window [start_sec, end_sec], extracts all
subtitle entries whose START time falls within the window and adjusts every
timestamp by (time_offset_sec - start_sec).

This is used to produce a per-segment SRT for burning into a concatenated
marketing clip.  Segments are processed left-to-right; callers accumulate
*time_offset_sec* by summing segment durations.

Example
───────
Source SRT ep01_en.srt has a line:
    00:05:32,100 --> 00:05:34,800  "You can't do this."

Segment: start=00:05:30,000  end=00:05:50,000  time_offset=0.0
  → new timestamps: 00:00:02,100 --> 00:00:04,800

Second segment: start=00:06:00,000  end=00:06:20,000  time_offset=20.0
  → lines in that window are offset to start at 20 s in the output clip.
"""
from __future__ import annotations

import re
from pathlib import Path

# Matches a single SRT block:  index\nTS --> TS\ntext(s)
_BLOCK_RE = re.compile(
    r"\d+\r?\n"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\r?\n"
    r"((?:.+\r?\n?)+)",
    re.MULTILINE,
)


def _ts_to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, rest = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _sec_to_ts(sec: float) -> str:
    sec = max(0.0, sec)
    h   = int(sec // 3600)
    m   = int((sec % 3600) // 60)
    s   = int(sec % 60)
    ms  = round((sec - int(sec)) * 1000)
    if ms >= 1000:
        ms -= 1000
        s  += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clip_srt(
    srt_path: Path,
    start_sec: float,
    end_sec: float,
    time_offset_sec: float = 0.0,
) -> str:
    """
    Extract subtitle entries from *srt_path* whose start time is within
    [start_sec, end_sec) and shift all timestamps by
    ``time_offset_sec - start_sec``.

    Entries that extend past *end_sec* are clamped at the window boundary.

    Parameters
    ----------
    srt_path:
        Path to the source SRT file (may not exist — returns "" if absent).
    start_sec:
        Window start in seconds (absolute, from the source episode timeline).
    end_sec:
        Window end in seconds (exclusive).
    time_offset_sec:
        Where this segment starts in the assembled output clip.  The shift
        applied to each timestamp is ``time_offset_sec - start_sec``.

    Returns
    -------
    str
        A valid SRT-formatted string, or "" when there are no entries in range.
    """
    if not srt_path.exists():
        return ""

    content = srt_path.read_text(encoding="utf-8")
    shift   = time_offset_sec - start_sec
    clip_duration = end_sec - start_sec
    entries: list[tuple[float, float, str]] = []

    for m in _BLOCK_RE.finditer(content):
        seg_start = _ts_to_sec(m.group(1))
        seg_end   = _ts_to_sec(m.group(2))
        text      = m.group(3).strip()

        # Include entries that START within the window
        if seg_start < end_sec and seg_start >= start_sec:
            new_start = seg_start + shift
            new_end   = min(seg_end + shift, time_offset_sec + clip_duration)
            if new_end > new_start:
                entries.append((new_start, new_end, text))

    if not entries:
        return ""

    blocks: list[str] = []
    for idx, (s, e, text) in enumerate(entries, 1):
        blocks.append(f"{idx}\n{_sec_to_ts(s)} --> {_sec_to_ts(e)}\n{text}")
    return "\n\n".join(blocks) + "\n"
