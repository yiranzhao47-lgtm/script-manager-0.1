"""
SRT output validator.

Checks every LLM-produced SRT string for four categories of defect:
  1.  Sequential numbering — indices must start at 1 and increment by exactly 1
  2.  Timestamp format    — every timing line must be HH:MM:SS,mmm --> HH:MM:SS,mmm
  3.  Timestamp ordering  — end time must be strictly after start time
  4.  Block separators    — each subtitle block must be preceded by at least one
                            blank line; extra blank lines are tolerated

Stuck-block detection
─────────────────────
When two blocks are written without a blank line between them, the block-splitter
treats them as one combined block.  The validator detects this by scanning the
text-line portion of each parsed block for a bare timestamp pattern.  A line that
is *only* a timestamp string cannot be legitimate subtitle text, so its presence
signals a missing separator.

Design notes
────────────
•  The validator is intentionally pure — no I/O, no logging, no side effects.
•  Windows CRLF and stray \\r are normalised before parsing.
•  The validator returns on the *first* violation found; it does not accumulate
   multiple errors (fail-fast strategy matches how the refiner uses it).
"""
from __future__ import annotations

import re

# Matches a complete SRT timestamp line (full-line anchored with ^ and $)
_TS_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\s*$"
)
_TS_SPLIT_RE = re.compile(r"\s*-->\s*")


def _parse_seconds(ts_str: str) -> float:
    """Parse 'HH:MM:SS,mmm' into a float of total seconds."""
    h, m, rest = ts_str.strip().split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


class SRTValidator:
    """
    Stateless SRT format validator.

    Usage::

        ok, reason = SRTValidator().validate_srt_string(raw_srt)
        if not ok:
            logger.warning("SRT invalid: %s", reason)
    """

    def validate_srt_string(self, srt_content: str) -> tuple[bool, str]:
        """
        Validate a complete SRT string.

        Returns:
            ``(True,  "OK")``                  — all checks passed
            ``(False, "<error description>")`` — first violation found
        """
        if not srt_content or not srt_content.strip():
            return False, "SRT content is empty"

        # ── Normalise line endings ─────────────────────────────────────────
        normalised = srt_content.replace("\r\n", "\n").replace("\r", "\n")

        # ── Split into blocks on two-or-more consecutive blank lines ───────
        # Stripping first ensures leading/trailing blank lines don't produce
        # empty phantom blocks.
        raw_blocks = re.split(r"\n{2,}", normalised.strip())
        blocks = [b.strip() for b in raw_blocks if b.strip()]

        if not blocks:
            return False, "No subtitle blocks found after blank-line splitting"

        expected_index = 1

        for block_pos, block in enumerate(blocks, start=1):
            lines = block.splitlines()

            # ── Line 0: sequence number ───────────────────────────────────
            index_str = lines[0].strip()
            if not index_str.isdigit():
                return False, (
                    f"Block {block_pos}: expected a sequence number on the first "
                    f"line, got {index_str!r}"
                )
            index = int(index_str)
            if index != expected_index:
                return False, (
                    f"Block {block_pos}: sequence number out of order — "
                    f"expected {expected_index}, got {index}"
                )
            expected_index += 1

            # ── Line 1: timestamp ─────────────────────────────────────────
            if len(lines) < 2:
                return False, (
                    f"Block {block_pos} (#{index}): missing timestamp line"
                )
            ts_line = lines[1].strip()
            if not _TS_RE.match(ts_line):
                return False, (
                    f"Block {block_pos} (#{index}): malformed timestamp — "
                    f"expected HH:MM:SS,mmm --> HH:MM:SS,mmm, got {ts_line!r}"
                )

            # ── Timestamp ordering: end > start ───────────────────────────
            ts_parts = _TS_SPLIT_RE.split(ts_line)
            if len(ts_parts) == 2:
                try:
                    t_start = _parse_seconds(ts_parts[0])
                    t_end = _parse_seconds(ts_parts[1])
                    if t_end <= t_start:
                        return False, (
                            f"Block {block_pos} (#{index}): end time is not "
                            f"after start time ({ts_parts[0]} --> {ts_parts[1]})"
                        )
                except (ValueError, IndexError):
                    # Regex already validated the format; this branch is
                    # defensive only and should never be reached.
                    pass

            # ── Lines 2+: subtitle text ───────────────────────────────────
            if len(lines) < 3:
                return False, (
                    f"Block {block_pos} (#{index}): subtitle text is empty"
                )

            # Stuck-block detection: a bare timestamp in the text portion
            # means a missing blank-line separator between two adjacent blocks.
            for rel, line in enumerate(lines[2:], start=3):
                if _TS_RE.match(line.strip()):
                    return False, (
                        f"Block {block_pos} (#{index}): a timestamp pattern was "
                        f"found at text line {rel} — two blocks are likely stuck "
                        f"together (missing blank-line separator): {line!r}"
                    )

        return True, "OK"
