"""
Per-episode LLM refinement.

Assembles an aligned-segment JSON + meta.json into a mode-specific Jinja2
prompt, calls the LLM, validates the SRT output, and writes the final subtitle
file.  All operations are atomic; the output SRT is only written after passing
`SRTValidator`.

Execution flow per episode
──────────────────────────
1.  Load aligned JSON from data/cache/aligned/{episode_id}_aligned.json
2.  Format segments as a compact JSON array for the prompt
3.  Estimate input tokens; warn if approaching model context limits
4.  Render mode-specific Jinja2 prompt (refine_same_lang.j2 | refine_cross_lang.j2)
5.  LLM call #1  →  SRTValidator
        OK  →  write SRT  →  done
        FAIL  →  build correction prompt (includes error + snippet of bad output)
6.  LLM call #2 (correction)  →  SRTValidator
        OK  →  write SRT  →  done
        FAIL  →  fallback path
7.  Fallback: assemble SRT directly from raw master_text (no LLM),
    record episode in data/output/validation_report.json for human review

Checkpoint
──────────
If data/output/{episode_id}.srt already exists the episode is skipped.
Delete the SRT file to force re-refinement.

Token budget
────────────
Estimation uses tiktoken when available; falls back to len(text) // 3.
Thresholds: WARN at 50 k tokens, ERROR-log (but proceed) at 100 k tokens.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.execution.srt_validator import SRTValidator
from src.utils.llm_client import LLMClient, LLMCallError

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"

_TOKEN_WARN  = 50_000
_TOKEN_LIMIT = 100_000


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _estimate_tokens(text: str) -> int:
    """Best-effort token count: tiktoken when available, else char ÷ 3."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 3)


def _fmt_ts(sec: float) -> str:
    """Convert float seconds to SRT timestamp string HH:MM:SS,mmm."""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s_int = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    # Guard millisecond overflow after rounding
    if ms >= 1000:
        ms -= 1000
        s_int += 1
    return f"{h:02d}:{m:02d}:{s_int:02d},{ms:03d}"


def _build_jinja_env():
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as exc:
        raise ImportError("jinja2 required: pip install jinja2") from exc
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )


def _atomic_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


# ══════════════════════════════════════════════════════════════════════════════
#  EpisodeRefiner
# ══════════════════════════════════════════════════════════════════════════════


class EpisodeRefiner:
    """
    Refines a single episode's aligned subtitle JSON into a polished SRT file.

    Parameters
    ----------
    llm_client:
        Pre-configured `LLMClient` (already has tenacity retry for HTTP errors).
    cfg:
        Full pipeline config dict.
    meta:
        Parsed ``meta.json`` dict — the ``characters`` key is injected into
        every LLM prompt so the model can canonicalize character names.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cfg: dict,
        meta: dict,
    ) -> None:
        self._llm = llm_client
        self._cfg = cfg
        self._meta = meta
        self._mode: str = cfg["pipeline"]["mode"]
        self._validator = SRTValidator()
        self._jinja = _build_jinja_env()

        # Configurable watermark keyword blacklist (platform intros/outros)
        asr_cfg = cfg.get(self._mode, {}).get("asr", {})
        self._watermark_patterns: list[str] = asr_cfg.get("watermark_patterns", [])

        out_dir: str = cfg.get("paths", {}).get("output_dir", "data/output")
        self._output_dir = Path(out_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._report_path = self._output_dir / "validation_report.json"

        prompts_cfg = cfg.get("execution", {}).get("prompts", {})
        if self._mode == "same_lang":
            self._prompt_name = prompts_cfg.get("same_lang", "refine_same_lang.j2")
        else:
            self._prompt_name = prompts_cfg.get("cross_lang", "refine_cross_lang.j2")

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def refine_episode(self, episode_json_path: str) -> str:
        """
        Refine one episode.

        Parameters
        ----------
        episode_json_path:
            Absolute or CWD-relative path to the aligned JSON file produced
            by ``OverlapAligner`` (e.g. ``data/cache/aligned/ep01_aligned.json``).

        Returns
        -------
        str
            Absolute path to the written SRT file.
        """
        src_path = Path(episode_json_path)
        episode_id = src_path.stem.replace("_aligned", "")

        out_path = self._output_dir / f"{episode_id}.srt"
        if out_path.exists():
            logger.info("EpisodeRefiner: %s cache hit — skipped", episode_id)
            return str(out_path)

        # ── Load aligned JSON ─────────────────────────────────────────────
        with src_path.open(encoding="utf-8") as f:
            data = json.load(f)

        segments: list[dict] = data.get("segments", [])
        if not segments:
            logger.warning("EpisodeRefiner: no segments in %s — writing empty SRT", src_path.name)
            _atomic_write(out_path, "")
            return str(out_path)

        # ── Watermark filter ──────────────────────────────────────────────
        if self._watermark_patterns:
            segments = self._filter_watermarks(segments, episode_id)

        # ── Build prompt ──────────────────────────────────────────────────
        segments_json = self._format_segments(segments)
        user_prompt   = self._render_prompt(episode_id, segments_json)
        system_prompt = self._system_prompt()

        self._log_token_budget(episode_id, user_prompt)

        # ── LLM call #1 ───────────────────────────────────────────────────
        raw = self._safe_llm_call(episode_id, system_prompt, user_prompt, attempt=1)
        if raw is None:
            return self._emit_fallback(episode_id, segments, "LLM call #1 failed (LLMCallError)")

        valid, reason = self._validator.validate_srt_string(raw)

        if valid:
            logger.info("EpisodeRefiner: %s — SRT valid on first attempt", episode_id)
            _atomic_write(out_path, _clip_overlapping_ends(_merge_short_fragments(_merge_artifact_fragments(raw))))
            return str(out_path)

        # ── LLM call #2: correction retry ─────────────────────────────────
        logger.warning(
            "EpisodeRefiner: %s — SRT invalid on attempt #1 (%s) — retrying",
            episode_id, reason,
        )
        correction_prompt = self._build_correction_prompt(
            original_prompt=user_prompt,
            bad_output=raw,
            error=reason,
            episode_id=episode_id,
        )
        raw2 = self._safe_llm_call(episode_id, system_prompt, correction_prompt, attempt=2)
        if raw2 is None:
            return self._emit_fallback(episode_id, segments, "LLM call #2 failed (LLMCallError)")

        valid2, reason2 = self._validator.validate_srt_string(raw2)

        if valid2:
            logger.info("EpisodeRefiner: %s — SRT valid after correction retry", episode_id)
            _atomic_write(out_path, _clip_overlapping_ends(_merge_short_fragments(_merge_artifact_fragments(raw2))))
            return str(out_path)

        # ── Both attempts failed: fallback ────────────────────────────────
        logger.error(
            "EpisodeRefiner: %s — SRT still invalid after correction (%s) — using fallback",
            episode_id, reason2,
        )
        return self._emit_fallback(
            episode_id, segments,
            f"SRT invalid after two LLM attempts. First error: {reason!r}. "
            f"Second error: {reason2!r}",
        )

    # ------------------------------------------------------------------ #
    #  Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    def _filter_watermarks(self, segments: list[dict], episode_id: str) -> list[dict]:
        """Remove segments whose master_text matches a known platform watermark pattern."""
        filtered = [
            s for s in segments
            if not any(p in s.get("master_text", "") for p in self._watermark_patterns)
        ]
        removed = len(segments) - len(filtered)
        if removed:
            logger.warning(
                "EpisodeRefiner: [%s] %d watermark segment(s) removed — %s",
                episode_id,
                removed,
                [s.get("master_text", "")[:40] for s in segments
                 if any(p in s.get("master_text", "") for p in self._watermark_patterns)],
            )
        return filtered

    def _format_segments(self, segments: list[dict]) -> str:
        """
        Compact the aligned segments into a JSON array for LLM input.

        Only ``context_text`` is included when the segment has an aligned
        secondary track (``context_available == True``).  This keeps the
        prompt as short as possible for episodes with low cross-track coverage.

        Two pre-processing guards applied before the LLM sees the data:

        Stale OCR guard — suppresses context_text when it matches a recently
        seen master or context text (subtitle lingering on screen from the
        previous dialogue line; alignment captures it at the wrong segment).

        Hallucination guard — when a long ASR phrase (≥8 chars) has already
        appeared 3+ times and OCR shows different content, substitute the OCR
        text as master_text (Whisper repeating earlier audio = hallucination).
        """
        phrase_counts: dict[str, int] = {}
        recent_seen: list[tuple[str, str]] = []  # (master, ctx) last 5 segs

        compact: list[dict] = []
        for i, seg in enumerate(segments, start=1):
            master = seg.get("master_text", "")
            ctx    = seg.get("context_text", "")
            ctx_avail = seg.get("context_available", False)

            # Stale OCR guard
            if ctx_avail and ctx and ctx != master:
                recent_masters = [p[0] for p in recent_seen[-5:]]
                recent_ctxs    = [p[1] for p in recent_seen[-5:] if p[1]]
                if ctx in recent_masters or ctx in recent_ctxs:
                    ctx_avail = False

            # Hallucination guard
            effective_master = master
            if (ctx_avail and ctx and ctx != master
                    and len(master) >= 8
                    and phrase_counts.get(master, 0) >= 3):
                effective_master = ctx

            phrase_counts[master] = phrase_counts.get(master, 0) + 1
            recent_seen = (recent_seen + [(master, ctx if ctx_avail else "")])[-5:]

            entry: dict = {
                "id": i,
                "start": _fmt_ts(float(seg.get("start", seg.get("start_sec", 0.0)))),
                "end":   _fmt_ts(float(seg.get("end",   seg.get("end_sec",   0.0)))),
                "master_text": effective_master,
            }
            if ctx_avail:
                entry["context_text"] = ctx
            compact.append(entry)
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _render_prompt(self, episode_id: str, segments_json: str) -> str:
        meta_json_str = json.dumps(
            {"characters": self._meta.get("characters", {})},
            ensure_ascii=False,
            indent=2,
        )
        template = self._jinja.get_template(self._prompt_name)
        return template.render(
            episode_number=episode_id,
            meta_json=meta_json_str,
            episode_json=segments_json,
        )

    def _system_prompt(self) -> str:
        if self._mode == "same_lang":
            return (
                "You are a professional Chinese subtitle editor. "
                "Output ONLY a valid SRT file — no prose, no markdown code fences, "
                "no explanations before or after the SRT."
            )
        return (
            "You are a professional English subtitle editor for Chinese drama. "
            "Output ONLY a valid SRT file — no prose, no markdown code fences, "
            "no explanations before or after the SRT."
        )

    def _build_correction_prompt(
        self,
        original_prompt: str,
        bad_output: str,
        error: str,
        episode_id: str,
    ) -> str:
        """
        Build the second-attempt prompt.

        Embeds the original task, the error description, and the first 800
        characters of the bad output so the model understands the specific
        failure without re-reading the full segment list.
        """
        snippet = bad_output[:800].replace("\n", "\\n")
        return (
            f"{original_prompt}\n\n"
            "━━ CORRECTION REQUEST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Your previous response for episode {episode_id} failed SRT validation.\n"
            f"Validation error: {error}\n\n"
            f"First 800 chars of the rejected output:\n{snippet}\n\n"
            "Please regenerate the COMPLETE SRT for this episode, strictly following:\n"
            "1. Sequence numbers must start at 1 and increment by exactly 1.\n"
            "2. Every timestamp line must match exactly: HH:MM:SS,mmm --> HH:MM:SS,mmm\n"
            "3. End time must be strictly after start time.\n"
            "4. ONE blank line between every subtitle block — no blank lines inside a block.\n"
            "5. No markdown fences, no explanatory text — SRT content only.\n"
            "Return ONLY the corrected SRT."
        )

    # ------------------------------------------------------------------ #
    #  LLM call (wraps LLMClient to isolate exception handling)           #
    # ------------------------------------------------------------------ #

    def _safe_llm_call(
        self,
        episode_id: str,
        system: str,
        user: str,
        attempt: int,
    ) -> Optional[str]:
        """
        Call the LLM; return the response text or None on permanent failure.

        ``LLMClient.complete`` already applies tenacity retry for 429/503 HTTP
        errors.  This method converts the final ``LLMCallError`` into ``None``
        so the caller can apply the fallback path without exception propagation.
        """
        try:
            response = self._llm.complete(
                system=system,
                user=user,
                module_name="Subtitle_Refine",
            )
            # Strip accidental markdown fences that slip through despite instructions
            response = _strip_markdown_fence(response)
            return response
        except LLMCallError as exc:
            logger.error(
                "EpisodeRefiner: %s — LLM call #%d failed: %s",
                episode_id, attempt, exc,
            )
            return None

    # ------------------------------------------------------------------ #
    #  Fallback SRT assembly                                               #
    # ------------------------------------------------------------------ #

    def _emit_fallback(
        self,
        episode_id: str,
        segments: list[dict],
        reason: str,
    ) -> str:
        """
        Assemble an emergency SRT from raw master_text (no LLM involvement).

        Writes the SRT, records the episode in validation_report.json, and
        returns the output path.  The fallback SRT is always valid SRT format
        even if the content is unrefined.
        """
        fallback_srt = self._assemble_fallback_srt(segments)
        out_path = self._output_dir / f"{episode_id}.srt"
        _atomic_write(out_path, _clip_overlapping_ends(_merge_short_fragments(_merge_artifact_fragments(fallback_srt))))

        self._append_validation_report(episode_id, reason)
        logger.warning(
            "EpisodeRefiner: %s — fallback SRT written (%d block(s)), "
            "episode logged in validation_report.json",
            episode_id,
            fallback_srt.count("\n\n"),
        )
        return str(out_path)

    def _assemble_fallback_srt(self, segments: list[dict]) -> str:
        """
        Build a plain SRT from raw master_text without any LLM refinement.

        Skips segments with empty master_text.  Uses the aligned timestamps
        directly (already in seconds from the OverlapAligner).
        """
        lines: list[str] = []
        idx = 1
        for seg in segments:
            text = seg.get("master_text", "").strip()
            if not text:
                continue
            start = _fmt_ts(float(seg.get("start", seg.get("start_sec", 0.0))))
            end   = _fmt_ts(float(seg.get("end",   seg.get("end_sec",   0.0))))
            lines.append(str(idx))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")   # blank-line separator
            idx += 1
        # Remove the trailing blank line added after the last block
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Validation report                                                   #
    # ------------------------------------------------------------------ #

    def _append_validation_report(self, episode_id: str, reason: str) -> None:
        """
        Append a failure record to ``data/output/validation_report.json``.

        The report file is read-modify-written atomically.  If the file is
        corrupt or missing it is recreated.
        """
        if self._report_path.exists():
            try:
                with self._report_path.open(encoding="utf-8") as f:
                    report = json.load(f)
            except (json.JSONDecodeError, OSError):
                report = {"failures": []}
        else:
            report = {"failures": []}

        report["failures"].append({
            "episode_id": episode_id,
            "mode": self._mode,
            "reason": reason,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "action": "Manual review required — SRT is unrefined fallback",
        })

        _atomic_write(
            self._report_path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )

    # ------------------------------------------------------------------ #
    #  Token budget                                                        #
    # ------------------------------------------------------------------ #

    def _log_token_budget(self, episode_id: str, prompt: str) -> None:
        est = _estimate_tokens(prompt)
        if est > _TOKEN_LIMIT:
            logger.error(
                "EpisodeRefiner: %s — estimated %d input tokens exceeds hard "
                "limit %d.  Consider splitting the episode or using a larger "
                "context model.",
                episode_id, est, _TOKEN_LIMIT,
            )
        elif est > _TOKEN_WARN:
            logger.warning(
                "EpisodeRefiner: %s — estimated %d input tokens (warn threshold %d)",
                episode_id, est, _TOKEN_WARN,
            )
        else:
            logger.debug("EpisodeRefiner: %s — estimated %d input tokens", episode_id, est)


# ══════════════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════════════


def _merge_artifact_fragments(srt_text: str) -> str:
    """Merge consecutive identical subtitle blocks that are artifact duplicates.

    Two conditions trigger a merge:
    - Either block is under 200 ms (micro-duplicate burst from Whisper), OR
    - The gap between the blocks is ≤ 3 seconds (same line repeated with pause,
      e.g. Whisper emitting "继续讨论" twice at 52.8 s and 54.5 s for one utterance).

    Keeps the earliest start and latest end timestamp, then renumbers.
    """
    blocks: list[list[str]] = []
    for raw in re.split(r'\n\n+', srt_text.strip()):
        parts = raw.strip().split('\n', 2)
        if len(parts) < 3:
            continue
        m = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', parts[1])
        if not m:
            continue
        blocks.append([m.group(1), m.group(2), parts[2].strip()])

    def _ms(ts: str) -> int:
        h, rest = ts.split(':', 1)
        mi, rest2 = rest.split(':', 1)
        s, ms = rest2.split(',')
        return int(h) * 3_600_000 + int(mi) * 60_000 + int(s) * 1_000 + int(ms)

    merged: list[tuple[str, str, str]] = []
    i = 0
    while i < len(blocks):
        start, end, text = blocks[i]
        dur = _ms(end) - _ms(start)
        j = i + 1
        while j < len(blocks):
            n_start, n_end, n_text = blocks[j]
            n_dur = _ms(n_end) - _ms(n_start)
            gap_ms = _ms(n_start) - _ms(end)
            if n_text == text and (dur < 200 or n_dur < 200 or gap_ms <= 3000):
                end = n_end
                dur = _ms(end) - _ms(start)
                j += 1
            else:
                break
        merged.append((start, end, text))
        i = j

    lines: list[str] = []
    for idx, (start, end, text) in enumerate(merged, 1):
        lines.extend([str(idx), f"{start} --> {end}", text, ''])
    if lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)


def _merge_short_fragments(srt_text: str) -> str:
    """Forward-merge any sub-300ms subtitle block into its following neighbor.

    Handles word-by-word ASR fragmentation where distinct short blocks must be
    concatenated to form a readable subtitle.  Repeated passes run until the
    output is stable (no more merges possible).

    Examples:
      "你"(60ms) "说"(120ms) "什么"(240ms) → "你说什么"(420ms) merged with
      next block "话"(1720ms) → "你说什么话"(2140ms) over two passes.
    """
    def _parse(t: str) -> list[list[str]]:
        blocks = []
        for raw in re.split(r'\n\n+', t.strip()):
            parts = raw.strip().split('\n', 2)
            if len(parts) < 3:
                continue
            m = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', parts[1])
            if m:
                blocks.append([m.group(1), m.group(2), parts[2].strip()])
        return blocks

    def _ms(ts: str) -> int:
        h, rest = ts.split(':', 1)
        mi, rest2 = rest.split(':', 1)
        s, ms = rest2.split(',')
        return int(h) * 3_600_000 + int(mi) * 60_000 + int(s) * 1_000 + int(ms)

    blocks = _parse(srt_text)
    changed = True
    while changed:
        changed = False
        new_blocks: list[list[str]] = []
        i = 0
        while i < len(blocks):
            start, end, text = blocks[i]
            dur = _ms(end) - _ms(start)
            if dur < 300 and i + 1 < len(blocks):
                n_start, n_end, n_text = blocks[i + 1]
                merged_text = text if n_text == text else text + n_text
                new_blocks.append([start, n_end, merged_text])
                i += 2
                changed = True
            else:
                new_blocks.append([start, end, text])
                i += 1
        blocks = new_blocks

    lines: list[str] = []
    for idx, (start, end, text) in enumerate(
            ((b[0], b[1], b[2]) for b in blocks), 1):
        lines.extend([str(idx), f"{start} --> {end}", text, ''])
    if lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)


def _clip_overlapping_ends(srt_text: str) -> str:
    """Clip each block's end time to the next block's start time when they overlap.

    Handles ~20ms overlaps introduced by OCR frame-interval rounding in same_lang
    rescue mode.  The SRT validator does not check cross-block overlap, so this
    pass is the last line of defence before writing.
    """
    if not srt_text or not srt_text.strip():
        return srt_text

    def _ms(ts: str) -> int:
        h, rest = ts.split(":", 1)
        mi, rest2 = rest.split(":", 1)
        s, ms = rest2.split(",")
        return int(h) * 3_600_000 + int(mi) * 60_000 + int(s) * 1_000 + int(ms)

    def _fmt(total_ms: int) -> str:
        ms = total_ms % 1_000
        total_s = total_ms // 1_000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks: list[list[str]] = []
    for raw in re.split(r"\n\n+", srt_text.strip()):
        parts = raw.strip().split("\n", 2)
        if len(parts) < 3:
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", parts[1])
        if m:
            blocks.append([parts[0].strip(), m.group(1), m.group(2), parts[2].strip()])

    clipped = False
    for i in range(len(blocks) - 1):
        end_ms   = _ms(blocks[i][2])
        next_ms  = _ms(blocks[i + 1][1])
        if end_ms > next_ms:
            blocks[i][2] = _fmt(next_ms)
            clipped = True

    if not clipped:
        return srt_text

    lines: list[str] = []
    for blk in blocks:
        lines.extend([blk[0], f"{blk[1]} --> {blk[2]}", blk[3], ""])
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _strip_markdown_fence(text: str) -> str:
    """
    Remove accidental markdown code fences from an SRT response.

    Some LLMs wrap the SRT in ```srt ... ``` or ``` ... ``` despite
    instructions.  This helper strips the outermost fence pair so the
    validator receives clean SRT text.
    """
    stripped = text.strip()
    fence_match = re.match(
        r"^```(?:srt|SRT)?\s*\n([\s\S]*?)\n```\s*$",
        stripped,
        re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1)
    return stripped
