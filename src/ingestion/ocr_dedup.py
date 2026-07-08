"""
OCR frame deduplication — cross_lang mode only.

Reads the raw per-frame OCR JSON produced by OCRRunner and merges consecutive
frames whose normalized text is similar (SequenceMatcher ratio >= threshold)
into compact subtitle segments with accurate start/end timestamps.

Input  : data/cache/ocr/{episode_id}_ocr.json          (mode must == "cross_lang")
Output : data/cache/ocr/ocr_timeline/{episode_id}_timeline.json

Output JSON schema:
    {
      "episode": "ep01",
      "mode": "cross_lang",
      "segment_count": 312,
      "segments": [
        {
          "id": 0,
          "start": 12.200,
          "end":   13.800,
          "raw_text":        "I w1ll k1ll you",   ← best OCR text for this run
          "normalized_text": "i will kill you",    ← after normalization
          "avg_confidence":  0.82,
          "frame_count":     8                     ← includes transition frames
        }, ...
      ]
    }

Algorithm
─────────
For each frame in arrival order:

  1. Classify the frame:
       • empty      — no text detected (OCR returned nothing above threshold)
       • transition — text detected but avg_confidence < FADE_CONF_FLOOR (≈ fade
                      in/out artifact); text is unreliable, timestamp is valid
       • normal     — text + confidence both usable

  2. Decision table:
       empty      → flush active segment (subtitle disappeared)
       transition → extend active segment end time only; do NOT update stored text
                    or trigger a new segment (avoids false cuts during fades)
       normal, no active segment    → open new segment
       normal, similarity >= thresh → extend active segment; update running avg conf
       normal, similarity <  thresh → flush active segment; open new one

  3. After iterating: flush any remaining active segment.

  4. Post-filter: discard segments shorter than min_segment_duration_sec
     (captures residual single-frame noise from rapid scene cuts).

Normalization applied BEFORE similarity comparison (never modifies stored text):
  • Strip HTML/formatting tags
  • Remove music / special symbols
  • Repair line-end hyphens  (word-\n → word)
  • Collapse whitespace
  • Lowercase
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Frames below this confidence are treated as "transition" (fade in/out)
_FADE_CONF_FLOOR = 0.55

# ── Normalization patterns ────────────────────────────────────────────────────
_RE_HTML = re.compile(r"<[^>]+>")
_RE_SYMBOLS = re.compile(r"[♪♫♬♩♭♮♯★☆■□●○▲▼◆◇►▻»«◄◅♦♪-♯]")
_RE_LINE_HYPHEN = re.compile(r"-\s*\n\s*")   # word-\n continuation
_RE_WHITESPACE = re.compile(r"\s+")


# ══════════════════════════════════════════════════════════════════════════════
#  Text helpers
# ══════════════════════════════════════════════════════════════════════════════


def _normalize(text: str) -> str:
    """
    Normalize OCR text for similarity comparison.
    This function is pure — it never modifies the value stored in the segment.
    """
    t = _RE_HTML.sub("", text)
    t = _RE_SYMBOLS.sub("", t)
    t = _RE_LINE_HYPHEN.sub("", t)
    t = _RE_WHITESPACE.sub(" ", t)
    return t.strip().lower()


def _similarity(a: str, b: str) -> float:
    """
    Character-level SequenceMatcher ratio between two normalized strings.
    Returns 1.0 for two empty strings (same empty → same subtitle).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OCRSegment:
    """One subtitle segment in the OCR-master timeline."""
    id: int
    start: float
    end: float
    raw_text: str           # best OCR text from normal frames in this run
    normalized_text: str    # normalized form of raw_text
    avg_confidence: float   # averaged over normal frames only
    frame_count: int        # total frames (normal + transition) in this run


@dataclass
class _Active:
    """
    Mutable accumulator for the segment currently being built.
    Private — not exposed outside this module.
    """
    start: float
    end: float
    raw_text: str
    normalized_text: str
    conf_sum: float     # sum of confidence values from normal frames
    normal_count: int   # number of normal (non-transition) frames
    total_count: int    # normal + transition frames

    def extend_normal(self, frame_time: float, interval: float, raw: str,
                      norm: str, conf: float) -> None:
        """Update with a normal frame that matches this segment."""
        self.end = frame_time + interval
        self.conf_sum += conf
        self.normal_count += 1
        self.total_count += 1
        # Keep whichever text has higher confidence as the canonical raw_text
        if conf > (self.conf_sum / max(self.normal_count, 1)):
            self.raw_text = raw
            self.normalized_text = norm

    def extend_transition(self, frame_time: float, interval: float) -> None:
        """Extend end time through a fade frame without updating text."""
        self.end = frame_time + interval
        self.total_count += 1

    def to_segment(self, seg_id: int) -> OCRSegment:
        avg_conf = self.conf_sum / max(self.normal_count, 1)
        return OCRSegment(
            id=seg_id,
            start=round(self.start, 3),
            end=round(self.end, 3),
            raw_text=self.raw_text,
            normalized_text=self.normalized_text,
            avg_confidence=round(avg_conf, 4),
            frame_count=self.total_count,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _atomic_json_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ══════════════════════════════════════════════════════════════════════════════
#  OCRDedup
# ══════════════════════════════════════════════════════════════════════════════


class OCRDedup:
    """
    Merges raw per-frame OCR data into a compact subtitle timeline.

    Designed exclusively for cross_lang mode where OCR is the master track.
    Raises ValueError if called on a same_lang OCR file.
    """

    def __init__(self, cfg: dict) -> None:
        dedup_cfg = cfg.get("cross_lang", {}).get("dedup", {})
        self._threshold: float = float(dedup_cfg.get("similarity_threshold", 0.85))
        self._min_duration: float = float(dedup_cfg.get("min_segment_duration_sec", 0.4))

        cache_root = Path(cfg["paths"]["cache_dir"])
        self._timeline_dir = cache_root / "ocr" / "ocr_timeline"
        self._timeline_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run_episode(self, raw_ocr_path: Path, episode_id: str) -> Path:
        """
        Deduplicate raw OCR frames from OCRRunner output.
        Returns path to the timeline JSON.  No-op if cache exists.

        Raises:
            FileNotFoundError : raw_ocr_path does not exist.
            ValueError        : input file is not cross_lang mode.
        """
        out_path = self._timeline_dir / f"{episode_id}_timeline.json"
        if out_path.exists():
            logger.info("OCR timeline cache hit — [%s] skipped", episode_id)
            return out_path

        if not raw_ocr_path.exists():
            raise FileNotFoundError(f"Raw OCR file not found: {raw_ocr_path}")

        with raw_ocr_path.open(encoding="utf-8") as f:
            raw_data = json.load(f)

        if raw_data.get("mode") != "cross_lang":
            raise ValueError(
                f"OCRDedup requires a cross_lang OCR file; "
                f"got mode='{raw_data.get('mode')}' in {raw_ocr_path.name}.  "
                "same_lang OCR blocks are handled directly by the overlap aligner."
            )

        fps_sampled = float(raw_data.get("fps_sampled", 5.0))
        frames: list[dict] = raw_data.get("frames", [])

        logger.info(
            "OCR dedup start — [%s]  frames=%d  threshold=%.2f  min_duration=%.2fs",
            episode_id, len(frames), self._threshold, self._min_duration,
        )

        segments = self._merge(frames, fps_sampled)
        segments = self._post_filter(segments, episode_id)

        # Re-index after filter
        for i, seg in enumerate(segments):
            seg.id = i

        payload = self._build_payload(episode_id, segments)
        _atomic_json_write(out_path, payload)

        logger.info(
            "OCR dedup done — [%s]  segments=%d  → %s",
            episode_id, len(segments), out_path.name,
        )
        return out_path

    # ------------------------------------------------------------------ #
    #  Core merge algorithm                                                #
    # ------------------------------------------------------------------ #

    def _merge(self, frames: list[dict], fps_sampled: float) -> list[OCRSegment]:
        interval = 1.0 / fps_sampled
        segments: list[OCRSegment] = []
        active: Optional[_Active] = None

        for frame in frames:
            frame_time = float(frame.get("frame_time", 0.0))
            raw_text = frame.get("combined_text", "")
            avg_conf = float(frame.get("avg_confidence", 0.0))

            is_empty = not raw_text.strip()

            # ── Empty frame: subtitle has disappeared ──────────────────
            if is_empty:
                if active is not None:
                    segments.append(active.to_segment(len(segments)))
                    active = None
                continue

            # ── Transition frame: fade in/out artifact ─────────────────
            if avg_conf < _FADE_CONF_FLOOR:
                if active is not None:
                    active.extend_transition(frame_time, interval)
                # A transition frame alone (no active segment) does not
                # start a new segment — it's noise without a text anchor.
                continue

            # ── Normal frame ───────────────────────────────────────────
            norm = _normalize(raw_text)

            if active is None:
                active = _Active(
                    start=frame_time,
                    end=frame_time + interval,
                    raw_text=raw_text,
                    normalized_text=norm,
                    conf_sum=avg_conf,
                    normal_count=1,
                    total_count=1,
                )
            else:
                sim = _similarity(norm, active.normalized_text)
                if sim >= self._threshold:
                    active.extend_normal(frame_time, interval, raw_text, norm, avg_conf)
                else:
                    # Text changed — flush current, open new
                    segments.append(active.to_segment(len(segments)))
                    active = _Active(
                        start=frame_time,
                        end=frame_time + interval,
                        raw_text=raw_text,
                        normalized_text=norm,
                        conf_sum=avg_conf,
                        normal_count=1,
                        total_count=1,
                    )

        if active is not None:
            segments.append(active.to_segment(len(segments)))

        return segments

    # ------------------------------------------------------------------ #
    #  Post-processing                                                     #
    # ------------------------------------------------------------------ #

    def _post_filter(
        self,
        segments: list[OCRSegment],
        episode_id: str,
    ) -> list[OCRSegment]:
        """Remove micro-segments shorter than min_duration (fade/cut artifacts)."""
        kept = [s for s in segments if (s.end - s.start) >= self._min_duration]
        removed = len(segments) - len(kept)
        if removed:
            logger.debug(
                "[%s] Post-filter removed %d micro-segment(s) "
                "(duration < %.2fs)",
                episode_id, removed, self._min_duration,
            )
        return kept

    # ------------------------------------------------------------------ #
    #  Serialization                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_payload(episode_id: str, segments: list[OCRSegment]) -> dict:
        return {
            "episode": episode_id,
            "mode": "cross_lang",
            "segment_count": len(segments),
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "raw_text": s.raw_text,
                    "normalized_text": s.normalized_text,
                    "avg_confidence": s.avg_confidence,
                    "frame_count": s.frame_count,
                }
                for s in segments
            ],
        }
