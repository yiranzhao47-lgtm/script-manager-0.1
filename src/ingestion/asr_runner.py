"""
ASR runner — stable-whisper (stable-ts) wrapper.

Produces word-level timestamped segments cached to:
    data/cache/asr/{episode_id}_asr.json

Roles:
  same_lang  : "master"           — ASR drives the aligned timeline
  cross_lang : "semantic_anchor"  — ASR provides Chinese semantic context for
                                    English OCR correction; stored as context_text
               "disabled"         — module is never called; GPUManager enforces this

Output JSON schema (per segment):
    {
      "id": 0,
      "start": 12.450,
      "end":   13.820,
      "text":  "你给我等着",
      "avg_probability": 0.96,
      "hallucination_risk": false,
      "words": [
        {"word": "你", "start": 12.450, "end": 12.650,
         "probability": 0.98, "low_confidence": false},
        ...
      ]
    }
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Per-word confidence below which the word is flagged (not filtered out)
_WORD_LOW_CONF = 0.40
# ── Segment average below which the whole segment is marked as hallucination risk
_SEG_HALLUCINATION_THRESHOLD = 0.50
# ── Segments longer than this are almost certainly Whisper hallucinations during
#    silence / non-speech regions.  Real dialogue segments rarely exceed 4–5 s;
#    hallucinated "请点赞订阅" fills span 10–20 s and always exceed this limit.
_SEG_MAX_DURATION_SEC = 8.0


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ASRWord:
    word: str
    start: float
    end: float
    probability: float
    low_confidence: bool


@dataclass
class ASRSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[ASRWord]
    avg_probability: float
    hallucination_risk: bool


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _atomic_json_write(path: Path, data: dict) -> None:
    """Write JSON via tmp-file + atomic rename (safe on crash mid-write)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    """
    Load stable-ts model.  Prefers faster-whisper backend (lower VRAM,
    quicker inference); falls back to standard Whisper if ctranslate2 is
    not installed.
    """
    try:
        import stable_whisper
    except ImportError as exc:
        raise ImportError(
            "stable-ts is required for ASR.  Install with:\n"
            "  pip install stable-ts"
        ) from exc

    # Attempt faster-whisper backend first
    try:
        model = stable_whisper.load_faster_whisper(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        logger.debug(
            "Whisper backend: faster-whisper  model=%s  device=%s  compute_type=%s",
            model_name, device, compute_type,
        )
        return model
    except Exception as exc:
        logger.warning(
            "faster-whisper unavailable (%s) — falling back to standard Whisper "
            "(slower, more VRAM).  Consider: pip install faster-whisper",
            exc,
        )

    model = stable_whisper.load_model(model_name, device=device)
    logger.debug("Whisper backend: standard  model=%s  device=%s", model_name, device)
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  ASRRunner
# ══════════════════════════════════════════════════════════════════════════════


class ASRRunner:
    """
    Transcribes episode video audio with Whisper and writes word-level
    timestamped segments to data/cache/asr/{episode_id}_asr.json.

    Cache hit: if the output file already exists the run is skipped entirely —
    no model load, no VRAM use, returns the cached path immediately.

    VRAM lifecycle is managed through gpu_manager.scope("whisper").  The
    caller must ensure GPUManager.configure(cfg) has been called first.
    """

    def __init__(self, cfg: dict) -> None:
        mode: str = cfg["pipeline"]["mode"]
        asr_cfg: dict = cfg.get(mode, {}).get("asr", {})

        self._model_name: str = asr_cfg.get("model", "large-v3")
        self._device: str = asr_cfg.get("device", "cuda")
        self._compute_type: str = asr_cfg.get("compute_type", "float16")

        # In same_lang the role is implicitly "master"; cross_lang reads from config.
        self._role: str = (
            "master" if mode == "same_lang"
            else asr_cfg.get("role", "semantic_anchor")
        )

        cache_root = Path(cfg["paths"]["cache_dir"])
        self._cache_dir = cache_root / "asr"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run_episode(self, video_path: Path, episode_id: str) -> Path:
        """
        Transcribe one episode.  Returns path to output JSON.
        Skips silently if the cache file already exists.

        Raises FileNotFoundError if video_path does not exist.
        """
        out_path = self._cache_dir / f"{episode_id}_asr.json"
        if out_path.exists():
            logger.info("ASR cache hit — [%s] skipped", episode_id)
            return out_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(
            "ASR start — [%s]  model=%s  role=%s  device=%s",
            episode_id, self._model_name, self._role, self._device,
        )

        from src.utils.gpu_manager import gpu_manager

        with gpu_manager.scope("whisper") as scope:
            model = _load_whisper_model(
                self._model_name, self._device, self._compute_type
            )
            scope.register(model)
            segments = self._transcribe(model, video_path, episode_id)
        # model freed here

        payload = self._build_payload(episode_id, video_path, segments)
        _atomic_json_write(out_path, payload)
        logger.info(
            "ASR done — [%s]  segments=%d  → %s",
            episode_id, len(segments), out_path.name,
        )
        return out_path

    # ------------------------------------------------------------------ #
    #  Transcription                                                       #
    # ------------------------------------------------------------------ #

    def _transcribe(
        self,
        model: Any,
        video_path: Path,
        episode_id: str,
    ) -> list[ASRSegment]:
        logger.info("Transcribing [%s] ...", video_path.name)

        # stable-ts accepts video paths directly (ffmpeg handles audio extraction).
        # language="zh" avoids auto-detect overhead; valid for both same_lang
        # (Chinese audio) and cross_lang semantic_anchor (also Chinese audio).
        result = model.transcribe(
            str(video_path),
            language="zh",
            word_timestamps=True,
            verbose=False,
        )

        segments: list[ASRSegment] = []
        for idx, raw_seg in enumerate(result.segments):
            words = self._parse_words(raw_seg)
            avg_prob = (
                sum(w.probability for w in words) / len(words) if words else 0.0
            )
            segments.append(
                ASRSegment(
                    id=idx,
                    start=round(float(raw_seg.start), 3),
                    end=round(float(raw_seg.end), 3),
                    text=str(raw_seg.text).strip(),
                    words=words,
                    avg_probability=round(avg_prob, 4),
                    hallucination_risk=avg_prob < _SEG_HALLUCINATION_THRESHOLD,
                )
            )

        n_risk = sum(1 for s in segments if s.hallucination_risk)
        if n_risk:
            logger.warning(
                "[%s] %d / %d segments flagged hallucination_risk "
                "(avg_probability < %.2f) — often silence or background music",
                episode_id, n_risk, len(segments), _SEG_HALLUCINATION_THRESHOLD,
            )

        # ── Duration-based hallucination filter ───────────────────────────
        # Whisper silently fills non-speech regions (silence, BGM) with
        # plausible-sounding text at high confidence.  The tell is segment
        # duration: real dialogue is < 4–5 s; hallucinated fills span 10–20 s.
        long_segs = [s for s in segments if (s.end - s.start) > _SEG_MAX_DURATION_SEC]
        if long_segs:
            segments = [s for s in segments if (s.end - s.start) <= _SEG_MAX_DURATION_SEC]
            logger.warning(
                "[%s] %d over-long segment(s) removed (duration > %.0fs) — "
                "likely Whisper hallucination during silence/BGM regions.  "
                "Removed: %s",
                episode_id,
                len(long_segs),
                _SEG_MAX_DURATION_SEC,
                [(round(s.end - s.start, 1), repr(s.text[:30])) for s in long_segs],
            )

        return segments

    @staticmethod
    def _parse_words(raw_seg: Any) -> list[ASRWord]:
        raw_words = getattr(raw_seg, "words", None)
        if not raw_words:
            return []
        words: list[ASRWord] = []
        for w in raw_words:
            prob = float(getattr(w, "probability", 1.0))
            words.append(
                ASRWord(
                    word=str(w.word),
                    start=round(float(w.start), 3),
                    end=round(float(w.end), 3),
                    probability=round(prob, 4),
                    low_confidence=prob < _WORD_LOW_CONF,
                )
            )
        return words

    # ------------------------------------------------------------------ #
    #  Serialization                                                       #
    # ------------------------------------------------------------------ #

    def _build_payload(
        self,
        episode_id: str,
        video_path: Path,
        segments: list[ASRSegment],
    ) -> dict:
        return {
            "episode": episode_id,
            "source": str(video_path),
            "model": self._model_name,
            "role": self._role,
            "segment_count": len(segments),
            "segments": [self._seg_to_dict(s) for s in segments],
        }

    @staticmethod
    def _seg_to_dict(seg: ASRSegment) -> dict:
        return {
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "avg_probability": seg.avg_probability,
            "hallucination_risk": seg.hallucination_risk,
            "words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                    "low_confidence": w.low_confidence,
                }
                for w in seg.words
            ],
        }
