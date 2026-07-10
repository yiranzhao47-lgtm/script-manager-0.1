"""
Alignment layer — time-axis overlap aligner.

Merges ASR and OCR data into a unified AlignedSegment schema consumed by the
Execution layer.  Both pipeline modes emit identical JSON field names so the
LLM prompt templates and downstream code never branch on mode.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  same_lang  — ASR is Master, OCR is Context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  For every ASR segment:

    1.  Guard: if ASR_duration < min_asr_duration_sec  →  skip OCR search
        (very short segments — single phonemes, breath marks — produce
        unreliable OCR matches at sub-frame granularity)

    2.  Scan every OCR block for positive time overlap:
            abs_overlap = min(asr_end, ocr_end) − max(asr_start, ocr_start)

    3.  Qualify the OCR block if EITHER condition holds:
            a.  abs_overlap  >  min_overlap_abs_sec   (0.5 s default)
            b.  abs_overlap / ASR_duration  >  min_overlap_ratio  (0.40 default)

        Condition (a) handles long ASR segments with short OCR blocks.
        Condition (b) handles short ASR segments fully inside a long OCR block.

    4.  Compute a bidirectional score for ranking:
            score = abs_overlap / min(ASR_duration, OCR_duration)
        This symmetrically rewards tight temporal matches regardless of which
        track has the shorter duration.

    5.  Store ALL qualifying blocks as ocr_candidates[] (sorted by score desc).
        context_text = candidates[0].text  (highest score = best context).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cross_lang — OCR is Master, ASR is Context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  For every OCR timeline segment:

    1.  Collect all ASR segments with abs_overlap > 0.
    2.  Sort by start time; concatenate texts (Chinese, no space separator).
    3.  Store as context_text.  Empty string when asr_role == "disabled".

Input files
───────────
  same_lang  :  asr/  {id}_asr.json     (segments[] with id/start/end/text/…)
                ocr/  {id}_ocr.json     (blocks[]   with start/end/combined_text/…)

  cross_lang :  asr/  {id}_asr.json     (segments[] — only when role=semantic_anchor)
                ocr/ocr_timeline/  {id}_timeline.json  (segments[] with raw_text/…)

Output
──────
  cache/aligned/{episode_id}_aligned.json

  Per-segment fields (both modes):
    segment_id        str   "ep01_0000"
    start / end       float seconds
    master_text       str   what was SPOKEN (ASR) or DISPLAYED (OCR)
    master_source     str   "asr" | "ocr"
    master_confidence float 0–1
    hallucination_risk bool  from ASR; always False when OCR is master
    context_text      str   best single context string ("" if none)
    context_source    str   "ocr" | "asr"
    context_available bool
    ocr_candidates    list  same_lang: all qualifying OCR blocks; cross_lang: []
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  Overlap arithmetic
# ══════════════════════════════════════════════════════════════════════════════


def _raw_overlap(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    """Signed temporal overlap in seconds.  Negative → no overlap."""
    return min(a_e, b_e) - max(a_s, b_s)


def _bidir_score(abs_ov: float, dur_a: float, dur_b: float) -> float:
    """
    Bidirectional overlap score: abs_overlap / min(dur_a, dur_b).
    Ranges [0, 1].  Rewards tight temporal matches symmetrically;
    does not penalise valid matches caused by asymmetric track durations.
    """
    min_dur = min(dur_a, dur_b)
    return abs_ov / min_dur if min_dur > 1e-6 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OCRCandidate:
    """One OCR block qualifying as context for a same_lang ASR segment."""
    text: str
    avg_confidence: float
    start: float
    end: float
    abs_overlap: float
    overlap_score: float    # abs_overlap / min(asr_dur, ocr_dur)


@dataclass
class AlignedSegment:
    segment_id: str
    start: float
    end: float
    master_text: str
    master_source: str          # "asr" | "ocr"
    master_confidence: float
    hallucination_risk: bool    # from ASR; False when OCR is master
    context_text: str           # "" when no context available
    context_source: str         # "ocr" | "asr"
    context_available: bool
    ocr_candidates: list        # list[OCRCandidate-dict]; [] for cross_lang


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _atomic_json_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  OverlapAligner
# ══════════════════════════════════════════════════════════════════════════════


class OverlapAligner:
    """
    Time-axis aligner for ASR and OCR ingestion outputs.

    Cache: data/cache/aligned/{episode_id}_aligned.json
    Cache hit → returns cached path, no computation.

    Instantiate once per pipeline run (reads config once), call run_episode()
    for each episode.
    """

    def __init__(self, cfg: dict) -> None:
        self._mode: str = cfg["pipeline"]["mode"]
        self._asr_role: str = (
            "master"
            if self._mode == "same_lang"
            else cfg.get("cross_lang", {}).get("asr", {}).get("role", "semantic_anchor")
        )

        # same_lang alignment thresholds
        al = cfg.get("same_lang", {}).get("alignment", {})
        self._min_abs: float = float(al.get("min_overlap_abs_sec", 0.5))
        self._min_ratio: float = float(al.get("min_overlap_ratio", 0.40))
        self._min_asr_dur: float = float(al.get("min_asr_duration_sec", 0.2))
        self._asr_sparse_rescue: bool = bool(al.get("asr_sparse_rescue", True))
        self._asr_min_segs: int = int(al.get("asr_min_segments", 10))
        self._rescue_dedup_sim: float = float(al.get("rescue_dedup_sim_threshold", 0.80))
        self._rescue_dedup_gap: float = float(al.get("rescue_dedup_gap_sec", 0.6))

        cache = Path(cfg["paths"]["cache_dir"])
        self._asr_dir = cache / "asr"
        self._ocr_dir = cache / "ocr"
        self._aligned_dir = cache / "aligned"
        self._aligned_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run_episode(
        self,
        episode_id: str,
        asr_path: Optional[Path] = None,
        ocr_path: Optional[Path] = None,
    ) -> Path:
        """
        Align one episode.  Returns path to aligned JSON.

        Paths are inferred from episode_id and pipeline mode when not supplied.
        Raises FileNotFoundError for the OCR master file if absent (ASR absence
        is non-fatal — produces segments with context_available=False).
        """
        out_path = self._aligned_dir / f"{episode_id}_aligned.json"
        if out_path.exists():
            logger.info("Alignment cache hit — [%s] skipped", episode_id)
            return out_path

        asr_path, ocr_path = self._resolve_paths(episode_id, asr_path, ocr_path)

        logger.info(
            "Alignment start — [%s]  mode=%s  asr_role=%s",
            episode_id, self._mode, self._asr_role,
        )

        if self._mode == "same_lang":
            segments = self._align_same_lang(episode_id, asr_path, ocr_path)
        else:
            segments = self._align_cross_lang(episode_id, asr_path, ocr_path)

        payload = self._build_payload(episode_id, segments)
        _atomic_json_write(out_path, payload)

        n_ctx = sum(1 for s in segments if s.context_available)
        logger.info(
            "Alignment done — [%s]  segments=%d  context=%d/%.1f%%  → %s",
            episode_id,
            len(segments),
            n_ctx,
            100.0 * n_ctx / max(len(segments), 1),
            out_path.name,
        )
        return out_path

    # ------------------------------------------------------------------ #
    #  Path resolution                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_paths(
        self,
        episode_id: str,
        asr_path: Optional[Path],
        ocr_path: Optional[Path],
    ) -> tuple[Optional[Path], Path]:
        if asr_path is None and self._asr_role != "disabled":
            asr_path = self._asr_dir / f"{episode_id}_asr.json"

        if ocr_path is None:
            if self._mode == "same_lang":
                ocr_path = self._ocr_dir / f"{episode_id}_ocr.json"
            else:
                ocr_path = self._ocr_dir / "ocr_timeline" / f"{episode_id}_timeline.json"

        return asr_path, ocr_path

    # ------------------------------------------------------------------ #
    #  same_lang: ASR master ↔ OCR context                                #
    # ------------------------------------------------------------------ #

    def _align_same_lang(
        self,
        episode_id: str,
        asr_path: Optional[Path],
        ocr_path: Path,
    ) -> list[AlignedSegment]:
        asr_segs = self._read_asr(asr_path, episode_id)
        ocr_blocks = self._read_ocr_blocks(ocr_path, episode_id)

        if not ocr_blocks:
            logger.warning(
                "[%s] No OCR blocks — context_available will be False for all %d segments",
                episode_id, len(asr_segs),
            )

        # ── Sparse-ASR rescue: flip to OCR-as-master when ASR barely transcribed ──
        # NOTE: checked before the empty-ASR guard so that episodes where ALL ASR
        # segments were hallucinations (asr_segs == []) still get OCR rescue when
        # OCR has sufficient blocks.
        if (
            self._asr_sparse_rescue
            and len(asr_segs) < self._asr_min_segs
            and len(ocr_blocks) >= 5
        ):
            return self._align_ocr_rescue(episode_id, asr_segs, ocr_blocks)

        if not asr_segs:
            logger.warning("[%s] No ASR segments — aligned output is empty", episode_id)
            return []

        # Leading OCR rescue: promote OCR blocks that precede the first ASR segment.
        # Handles episodes where Whisper misses speech in the opening seconds (BGM,
        # cold open with no clear voice) while OCR captured the on-screen subtitles.
        leading_segs: list[AlignedSegment] = []
        if asr_segs and ocr_blocks:
            first_asr_start = asr_segs[0]["start"]
            cutoff = first_asr_start - 1.0  # 1s safety margin before first ASR
            leading = [b for b in ocr_blocks if b["end"] <= cutoff]
            if leading:
                leading = self._dedup_rescue_blocks(leading, episode_id)
                logger.info(
                    "[%s] Leading OCR rescue: %d block(s) before first ASR at %.2fs",
                    episode_id, len(leading), first_asr_start,
                )
                for idx, blk in enumerate(leading):
                    leading_segs.append(
                        AlignedSegment(
                            segment_id=f"{episode_id}_lead_{idx:04d}",
                            start=blk["start"],
                            end=blk["end"],
                            master_text=blk["combined_text"],
                            master_source="ocr_leading",
                            master_confidence=blk["avg_confidence"],
                            hallucination_risk=False,
                            context_text="",
                            context_source="ocr",
                            context_available=False,
                            ocr_candidates=[],
                        )
                    )

        results: list[AlignedSegment] = []
        for seg in asr_segs:
            asr_dur = seg["end"] - seg["start"]
            candidates = self._find_ocr_candidates(seg, asr_dur, ocr_blocks)

            best = candidates[0] if candidates else None
            results.append(
                AlignedSegment(
                    segment_id=f"{episode_id}_{seg['id']:04d}",
                    start=seg["start"],
                    end=seg["end"],
                    master_text=seg["text"],
                    master_source="asr",
                    master_confidence=seg["avg_probability"],
                    hallucination_risk=seg.get("hallucination_risk", False),
                    context_text=best.text if best else "",
                    context_source="ocr",
                    context_available=best is not None,
                    ocr_candidates=[
                        {
                            "text": c.text,
                            "avg_confidence": c.avg_confidence,
                            "start": c.start,
                            "end": c.end,
                            "abs_overlap": c.abs_overlap,
                            "overlap_score": c.overlap_score,
                        }
                        for c in candidates
                    ],
                )
            )

        self._log_coverage_same_lang(episode_id, results, ocr_blocks)
        return leading_segs + results

    def _find_ocr_candidates(
        self,
        asr_seg: dict,
        asr_dur: float,
        ocr_blocks: list[dict],
    ) -> list[OCRCandidate]:
        """
        Return all OCR blocks qualifying as context for one ASR segment,
        sorted by bidirectional overlap score descending.
        """
        if asr_dur < self._min_asr_dur:
            return []

        candidates: list[OCRCandidate] = []
        for blk in ocr_blocks:
            abs_ov = _raw_overlap(
                asr_seg["start"], asr_seg["end"],
                blk["start"], blk["end"],
            )
            if abs_ov <= 0:
                continue

            ocr_dur = blk["end"] - blk["start"]
            qualifies = (
                abs_ov > self._min_abs
                or (asr_dur > 0 and abs_ov / asr_dur > self._min_ratio)
            )
            if not qualifies:
                continue

            score = _bidir_score(abs_ov, asr_dur, ocr_dur)
            candidates.append(
                OCRCandidate(
                    text=blk["combined_text"],
                    avg_confidence=blk["avg_confidence"],
                    start=blk["start"],
                    end=blk["end"],
                    abs_overlap=round(abs_ov, 3),
                    overlap_score=round(score, 4),
                )
            )

        candidates.sort(key=lambda c: c.overlap_score, reverse=True)
        return candidates

    # ------------------------------------------------------------------ #
    #  same_lang sparse-ASR rescue: OCR as master                         #
    # ------------------------------------------------------------------ #

    def _dedup_rescue_blocks(self, blocks: list[dict], episode_id: str) -> list[dict]:
        """
        Merge adjacent same_lang OCR blocks with near-identical text.

        In same_lang mode OCR blocks are never deduped (OCRDedup is cross_lang only).
        When rescue promotes them to master, boundary-frame OCR misreads (e.g. 干→千)
        create separate blocks for what is physically one subtitle on screen.
        This pass merges consecutive blocks that are temporally adjacent
        (gap ≤ 0.6 s) AND textually similar (SequenceMatcher ≥ 0.80).
        """
        if not blocks:
            return blocks

        _SIM_THRESH = self._rescue_dedup_sim
        _GAP_THRESH = self._rescue_dedup_gap

        sorted_blocks = sorted(blocks, key=lambda b: b["start"])
        merged: list[dict] = []
        active: dict | None = None

        for blk in sorted_blocks:
            if active is None:
                active = dict(blk)
                continue

            gap = blk["start"] - active["end"]
            if gap <= _GAP_THRESH:
                sim = SequenceMatcher(
                    None, active["combined_text"], blk["combined_text"]
                ).ratio()
                if sim >= _SIM_THRESH:
                    # Extend time range; keep higher-confidence text
                    active["end"] = blk["end"]
                    if blk["avg_confidence"] > active["avg_confidence"]:
                        active["combined_text"] = blk["combined_text"]
                        active["avg_confidence"] = blk["avg_confidence"]
                    continue

            merged.append(active)
            active = dict(blk)

        if active is not None:
            merged.append(active)

        n_removed = len(sorted_blocks) - len(merged)
        if n_removed:
            logger.info(
                "[%s] Rescue dedup merged %d duplicate OCR block(s) "
                "(%d → %d blocks)",
                episode_id, n_removed, len(sorted_blocks), len(merged),
            )
        return merged

    def _align_ocr_rescue(
        self,
        episode_id: str,
        asr_segs: list[dict],
        ocr_blocks: list[dict],
    ) -> list[AlignedSegment]:
        """
        Fallback for same_lang episodes where Whisper produced too few segments
        (heavy BGM suppressing speech detection).  Uses OCR blocks as master;
        any surviving ASR segments become context for downstream LLM refinement.
        """
        logger.warning(
            "[%s] ASR sparse (%d segment(s) < threshold %d) — "
            "rescue: %d OCR block(s) promoted to master",
            episode_id, len(asr_segs), self._asr_min_segs, len(ocr_blocks),
        )
        ocr_blocks = self._dedup_rescue_blocks(ocr_blocks, episode_id)
        results: list[AlignedSegment] = []
        for idx, blk in enumerate(ocr_blocks):
            context_text, context_ok = self._build_asr_context(blk, asr_segs)
            results.append(
                AlignedSegment(
                    segment_id=f"{episode_id}_{idx:04d}",
                    start=blk["start"],
                    end=blk["end"],
                    master_text=blk["combined_text"],
                    master_source="ocr",
                    master_confidence=blk["avg_confidence"],
                    hallucination_risk=False,
                    context_text=context_text,
                    context_source="asr",
                    context_available=context_ok,
                    ocr_candidates=[],
                )
            )
        return results

    # ------------------------------------------------------------------ #
    #  cross_lang: OCR master ↔ ASR context                               #
    # ------------------------------------------------------------------ #

    def _align_cross_lang(
        self,
        episode_id: str,
        asr_path: Optional[Path],
        ocr_path: Path,
    ) -> list[AlignedSegment]:
        ocr_segs = self._read_ocr_timeline(ocr_path, episode_id)
        asr_segs: list[dict] = []

        if self._asr_role == "semantic_anchor" and asr_path is not None:
            asr_segs = self._read_asr(asr_path, episode_id)
            if not asr_segs:
                logger.warning(
                    "[%s] asr_role=semantic_anchor but ASR file has no segments — "
                    "all OCR segments will have context_available=False",
                    episode_id,
                )

        if not ocr_segs:
            logger.warning("[%s] No OCR timeline segments — aligned output is empty", episode_id)
            return []

        results: list[AlignedSegment] = []
        for seg in ocr_segs:
            context_text, context_ok = self._build_asr_context(seg, asr_segs)
            results.append(
                AlignedSegment(
                    segment_id=f"{episode_id}_{seg['id']:04d}",
                    start=seg["start"],
                    end=seg["end"],
                    master_text=seg["raw_text"],
                    master_source="ocr",
                    master_confidence=seg["avg_confidence"],
                    hallucination_risk=False,  # not a concept when OCR is master
                    context_text=context_text,
                    context_source="asr",
                    context_available=context_ok,
                    ocr_candidates=[],  # N/A when OCR is master
                )
            )

        return results

    @staticmethod
    def _build_asr_context(
        ocr_seg: dict,
        asr_segs: list[dict],
    ) -> tuple[str, bool]:
        """
        Concatenate text of all ASR segments overlapping this OCR segment.

        Returns (context_text, context_available).
        Segments are sorted by start time before joining so the resulting
        Chinese text reads in chronological speech order.
        """
        if not asr_segs:
            return "", False

        overlapping = [
            s for s in asr_segs
            if _raw_overlap(
                ocr_seg["start"], ocr_seg["end"],
                s["start"], s["end"],
            ) > 0
        ]
        if not overlapping:
            return "", False

        overlapping.sort(key=lambda s: s["start"])
        # Chinese text: join without spaces (punctuation handles boundaries)
        text = "".join(s["text"] for s in overlapping).strip()
        return text, bool(text)

    # ------------------------------------------------------------------ #
    #  Data loading                                                        #
    # ------------------------------------------------------------------ #

    def _read_asr(self, path: Optional[Path], episode_id: str) -> list[dict]:
        if path is None:
            return []
        if not path.exists():
            logger.warning("[%s] ASR file not found: %s", episode_id, path.name)
            return []
        return _load_json(path).get("segments", [])

    def _read_ocr_blocks(self, path: Path, episode_id: str) -> list[dict]:
        """Load same_lang OCR blocks.  Raises ValueError on mode mismatch."""
        if not path.exists():
            logger.warning("[%s] OCR file not found: %s", episode_id, path.name)
            return []
        data = _load_json(path)
        if data.get("mode") != "same_lang":
            raise ValueError(
                f"[{episode_id}] Expected same_lang OCR (blocks[]), "
                f"got mode='{data.get('mode')}' in {path.name}.  "
                "Check that you are not passing a cross_lang timeline to a same_lang run."
            )
        return data.get("blocks", [])

    def _read_ocr_timeline(self, path: Path, episode_id: str) -> list[dict]:
        """Load cross_lang OCR dedup timeline.  Raises ValueError on mode mismatch."""
        if not path.exists():
            raise FileNotFoundError(
                f"[{episode_id}] OCR timeline not found: {path}\n"
                "Run OCRRunner → OCRDedup before OverlapAligner in cross_lang mode."
            )
        data = _load_json(path)
        if data.get("mode") != "cross_lang":
            raise ValueError(
                f"[{episode_id}] Expected cross_lang OCR timeline (segments[]), "
                f"got mode='{data.get('mode')}' in {path.name}."
            )
        return data.get("segments", [])

    # ------------------------------------------------------------------ #
    #  Serialization                                                       #
    # ------------------------------------------------------------------ #

    def _build_payload(
        self,
        episode_id: str,
        segments: list[AlignedSegment],
    ) -> dict:
        n = len(segments)
        n_ctx = sum(1 for s in segments if s.context_available)
        n_risk = sum(1 for s in segments if s.hallucination_risk)
        n_multi = sum(1 for s in segments if len(s.ocr_candidates) > 1)

        return {
            "episode": episode_id,
            "mode": self._mode,
            "asr_role": self._asr_role,
            "segment_count": n,
            "stats": {
                "with_context": n_ctx,
                "without_context": n - n_ctx,
                "context_coverage_pct": round(100.0 * n_ctx / max(n, 1), 1),
                "hallucination_risk_count": n_risk,
                "multi_ocr_candidate_count": n_multi,  # same_lang only
            },
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "start": s.start,
                    "end": s.end,
                    "master_text": s.master_text,
                    "master_source": s.master_source,
                    "master_confidence": s.master_confidence,
                    "hallucination_risk": s.hallucination_risk,
                    "context_text": s.context_text,
                    "context_source": s.context_source,
                    "context_available": s.context_available,
                    "ocr_candidates": s.ocr_candidates,
                }
                for s in segments
            ],
        }

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def _log_coverage_same_lang(
        self,
        episode_id: str,
        results: list[AlignedSegment],
        ocr_blocks: list[dict],
    ) -> None:
        """
        Warn if context coverage is suspiciously low — may indicate ROI
        misconfiguration or a video with no embedded subtitles.
        """
        if not ocr_blocks:
            return  # already warned upstream

        n = len(results)
        n_ctx = sum(1 for r in results if r.context_available)
        coverage = n_ctx / max(n, 1)

        if coverage < 0.50:
            logger.warning(
                "[%s] Low OCR context coverage: %.1f%% (%d/%d segments have OCR match).  "
                "Check roi settings or subtitle ROI placement.",
                episode_id, coverage * 100, n_ctx, n,
            )

        # Warn about ASR segments whose short duration forced skipping OCR search
        n_skipped = sum(
            1 for r in results
            if not r.context_available
            and (r.end - r.start) < self._min_asr_dur
        )
        if n_skipped:
            logger.debug(
                "[%s] %d ASR segment(s) skipped OCR search "
                "(duration < %.2fs min_asr_duration)",
                episode_id, n_skipped, self._min_asr_dur,
            )
