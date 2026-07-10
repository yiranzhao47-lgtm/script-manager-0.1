"""
Unit tests for src.alignment.overlap_aligner.

Coverage
────────
  Group 1  Module-level arithmetic  (_raw_overlap, _bidir_score)
  Group 2  OCR candidate selection  (_find_ocr_candidates)
             · abs threshold boundary (> not >=)
             · ratio threshold fallback for short ASR
             · short-ASR skip guard (dur < min_asr_duration_sec)
             · no temporal overlap → rejected
             · multi-candidate sorted by score descending
  Group 3  ASR context builder      (_build_asr_context, static)
             · single / multiple overlapping segments
             · chronological join order
             · no overlap / empty ASR list / adjacent-not-overlapping
  Group 4  Rescue dedup             (_dedup_rescue_blocks)
             · similar + adjacent → merged; end time extended
             · dissimilar texts → kept separate
             · large gap → kept separate
             · higher-confidence text wins on merge
             · empty input / single block
  Group 5  same_lang integration    (run_episode round-trip)
             · happy path: context_available=True, correct field values
             · no OCR match: context_available=False
             · cache hit: returns existing file untouched
             · hallucination_risk propagated from ASR
             · multiple OCR candidates: best score → context_text
             · sparse-ASR rescue: OCR promoted to master
  Group 6  cross_lang integration
             · multiple ASR segments concatenated in time order
             · no ASR overlap → context_available=False
             · ocr_candidates always [] in cross_lang
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.alignment.overlap_aligner import (
    OverlapAligner,
    _bidir_score,
    _raw_overlap,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(tmp: Path, mode: str = "same_lang") -> dict:
    return {
        "pipeline": {"mode": mode},
        "same_lang": {"alignment": {}},
        "cross_lang": {"asr": {"role": "semantic_anchor"}},
        "paths": {"cache_dir": str(tmp)},
    }


def _asr_seg(id: int, start: float, end: float, text: str,
             prob: float = 0.9, hallucination: bool = False) -> dict:
    return {
        "id": id, "start": start, "end": end,
        "text": text, "avg_probability": prob,
        "hallucination_risk": hallucination,
    }


def _ocr_block(start: float, end: float, text: str, conf: float = 0.9) -> dict:
    return {"start": start, "end": end, "combined_text": text, "avg_confidence": conf}


def _ocr_tl_seg(id: int, start: float, end: float, text: str, conf: float = 0.9) -> dict:
    return {"id": id, "start": start, "end": end,
            "raw_text": text, "avg_confidence": conf}


def _write_asr(path: Path, segs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"segments": segs}), encoding="utf-8")


def _write_ocr_same(path: Path, blocks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "same_lang", "blocks": blocks}), encoding="utf-8")


def _write_ocr_timeline(path: Path, segs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "cross_lang", "segments": segs}), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Group 1 — module-level arithmetic
# ─────────────────────────────────────────────────────────────────────────────

class TestRawOverlap:
    def test_full_containment(self):
        # ASR [0,2] contains OCR [0.5,1.5] → overlap = 1.0
        assert _raw_overlap(0.0, 2.0, 0.5, 1.5) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # [0,2] ∩ [1,3] = 1.0
        assert _raw_overlap(0.0, 2.0, 1.0, 3.0) == pytest.approx(1.0)

    def test_no_overlap_returns_negative(self):
        assert _raw_overlap(0.0, 1.0, 2.0, 3.0) < 0

    def test_adjacent_segments_zero_overlap(self):
        # Touching but not overlapping
        assert _raw_overlap(0.0, 1.0, 1.0, 2.0) == pytest.approx(0.0)

    def test_symmetric(self):
        assert _raw_overlap(0.0, 2.0, 0.5, 1.5) == pytest.approx(
            _raw_overlap(0.5, 1.5, 0.0, 2.0)
        )


class TestBidirScore:
    def test_perfect_match_equal_durations(self):
        assert _bidir_score(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_short_ocr_fully_inside_long_asr(self):
        # ASR 4s, OCR 1s fully inside → overlap 1s → score = 1/min(4,1) = 1.0
        assert _bidir_score(1.0, 4.0, 1.0) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # overlap 0.5, min(dur_a=2, dur_b=4) = 2 → 0.25
        assert _bidir_score(0.5, 2.0, 4.0) == pytest.approx(0.25)

    def test_zero_duration_returns_zero(self):
        assert _bidir_score(0.5, 0.0, 0.0) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Group 2 — _find_ocr_candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestFindOCRCandidates:
    def _aligner(self, tmp: Path) -> OverlapAligner:
        return OverlapAligner(_cfg(tmp))

    def test_qualifies_via_abs_threshold(self, tmp_path):
        """abs_overlap 0.51 > 0.5 default → candidate accepted."""
        al = self._aligner(tmp_path)
        asr = _asr_seg(0, 0.0, 2.0, "hello")
        blk = _ocr_block(0.0, 0.51, "HELLO")           # overlap = 0.51
        candidates = al._find_ocr_candidates(asr, 2.0, [blk])
        assert len(candidates) == 1
        assert candidates[0].text == "HELLO"
        assert candidates[0].abs_overlap == pytest.approx(0.51, abs=1e-3)

    def test_rejects_at_abs_boundary_exact(self, tmp_path):
        """abs_overlap exactly 0.5 does NOT qualify — condition is strictly >."""
        al = self._aligner(tmp_path)
        asr = _asr_seg(0, 0.0, 2.0, "hello")
        blk = _ocr_block(0.0, 0.5, "HELLO")            # overlap = 0.5 exactly
        # ratio = 0.5/2.0 = 0.25 < 0.40 → also fails ratio check
        assert al._find_ocr_candidates(asr, 2.0, [blk]) == []

    def test_qualifies_via_ratio_fallback(self, tmp_path):
        """Short ASR: ratio 0.41 > 0.40 qualifies even though abs < 0.5."""
        al = self._aligner(tmp_path)
        # ASR 0.8s; OCR [0.4, 0.73] → overlap = 0.33s
        # ratio = 0.33 / 0.8 = 0.4125 > 0.40 ✓
        asr = _asr_seg(0, 0.0, 0.8, "hi")
        blk = _ocr_block(0.4, 0.73, "HI")
        assert len(al._find_ocr_candidates(asr, 0.8, [blk])) == 1

    def test_rejects_below_both_thresholds(self, tmp_path):
        """Small abs AND small ratio → rejected."""
        al = self._aligner(tmp_path)
        asr = _asr_seg(0, 0.0, 2.0, "hello")
        blk = _ocr_block(0.0, 0.1, "HI")              # abs=0.1; ratio=0.05
        assert al._find_ocr_candidates(asr, 2.0, [blk]) == []

    def test_short_asr_skips_ocr_search(self, tmp_path):
        """ASR duration < min_asr_duration_sec (0.2) → returns [] without checking OCR."""
        al = self._aligner(tmp_path)
        asr = _asr_seg(0, 0.0, 0.19, "uh")
        blk = _ocr_block(0.0, 0.19, "UH")
        assert al._find_ocr_candidates(asr, 0.19, [blk]) == []

    def test_no_temporal_overlap_rejected(self, tmp_path):
        """OCR block ends before ASR starts → no candidates."""
        al = self._aligner(tmp_path)
        asr = _asr_seg(0, 2.0, 4.0, "hello")
        blk = _ocr_block(0.0, 1.5, "before")
        assert al._find_ocr_candidates(asr, 2.0, [blk]) == []

    def test_multi_candidate_sorted_by_score_descending(self, tmp_path):
        """Highest-scoring OCR block is first; context_text should use it."""
        al = self._aligner(tmp_path)
        asr = _asr_seg(0, 0.0, 2.0, "hello")
        # Block A: [0.4,3.4] OCR_dur=3.0; abs_ov=min(2,3.4)-max(0,0.4)=2-0.4=1.6
        #   score = 1.6 / min(2.0, 3.0) = 0.8
        blk_a = _ocr_block(0.4, 3.4, "LOWER_SCORE")
        # Block B: [0.5,1.5] OCR_dur=1.0; abs_ov=min(2,1.5)-max(0,0.5)=1.5-0.5=1.0
        #   score = 1.0 / min(2.0, 1.0) = 1.0
        blk_b = _ocr_block(0.5, 1.5, "BEST_SCORE")
        candidates = al._find_ocr_candidates(asr, 2.0, [blk_a, blk_b])
        assert len(candidates) == 2
        assert candidates[0].text == "BEST_SCORE"
        assert candidates[0].overlap_score > candidates[1].overlap_score


# ─────────────────────────────────────────────────────────────────────────────
#  Group 3 — _build_asr_context (static method)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildASRContext:
    @staticmethod
    def _asr(start, end, text):
        return {"start": start, "end": end, "text": text}

    @staticmethod
    def _ocr(start, end):
        return {"start": start, "end": end}

    def test_single_overlapping_asr(self):
        text, ok = OverlapAligner._build_asr_context(
            self._ocr(0.0, 2.0),
            [self._asr(0.5, 1.5, "你好")],
        )
        assert ok is True
        assert text == "你好"

    def test_multiple_overlapping_joined_chronologically(self):
        """Two overlapping ASR segments are sorted by start time before joining."""
        text, ok = OverlapAligner._build_asr_context(
            self._ocr(0.0, 4.0),
            [
                self._asr(2.0, 3.0, "世界"),   # later — but passed first
                self._asr(0.5, 1.5, "你好"),   # earlier
            ],
        )
        assert ok is True
        assert text == "你好世界"            # chronological order, no space

    def test_no_overlapping_asr_returns_empty(self):
        text, ok = OverlapAligner._build_asr_context(
            self._ocr(5.0, 6.0),
            [self._asr(0.0, 1.0, "不重叠")],
        )
        assert ok is False
        assert text == ""

    def test_empty_asr_list_returns_empty(self):
        text, ok = OverlapAligner._build_asr_context(self._ocr(0.0, 2.0), [])
        assert ok is False
        assert text == ""

    def test_adjacent_asr_not_included(self):
        """ASR ends exactly where OCR starts → overlap = 0 → not included."""
        text, ok = OverlapAligner._build_asr_context(
            self._ocr(1.0, 2.0),
            [self._asr(0.0, 1.0, "刚好接触")],
        )
        assert ok is False
        assert text == ""


# ─────────────────────────────────────────────────────────────────────────────
#  Group 4 — _dedup_rescue_blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestDedupRescueBlocks:
    def _aligner(self, tmp: Path) -> OverlapAligner:
        return OverlapAligner(_cfg(tmp))

    def _blk(self, start, end, text, conf=0.9):
        return {"start": start, "end": end, "combined_text": text, "avg_confidence": conf}

    def test_similar_adjacent_merged_end_extended(self, tmp_path):
        """Gap 0.1 s ≤ 0.6, similar text → merged; end time updated."""
        al = self._aligner(tmp_path)
        b1 = self._blk(0.0, 1.0, "你好世界")
        b2 = self._blk(1.1, 2.0, "你好世界!")  # sim ≈ 0.89 ≥ 0.80
        result = al._dedup_rescue_blocks([b1, b2], "ep01")
        assert len(result) == 1
        assert result[0]["end"] == pytest.approx(2.0)

    def test_dissimilar_texts_not_merged(self, tmp_path):
        al = self._aligner(tmp_path)
        b1 = self._blk(0.0, 1.0, "你好世界")
        b2 = self._blk(1.1, 2.0, "完全不同的台词内容")   # sim ≪ 0.80
        result = al._dedup_rescue_blocks([b1, b2], "ep01")
        assert len(result) == 2

    def test_large_gap_not_merged(self, tmp_path):
        """Gap 2.0 s > 0.6 s threshold → kept separate."""
        al = self._aligner(tmp_path)
        b1 = self._blk(0.0, 1.0, "你好世界")
        b2 = self._blk(3.0, 4.0, "你好世界")   # identical text but huge gap
        result = al._dedup_rescue_blocks([b1, b2], "ep01")
        assert len(result) == 2

    def test_higher_confidence_text_wins_on_merge(self, tmp_path):
        """When merging, the block with higher avg_confidence supplies the text."""
        al = self._aligner(tmp_path)
        # Gap 0.1 ≤ 0.6; texts differ slightly (sim ≈ 0.94) → will merge
        b1 = self._blk(0.0, 1.0, "我爱你China",  conf=0.6)
        b2 = self._blk(1.1, 2.0, "我爱你China!", conf=0.95)
        result = al._dedup_rescue_blocks([b1, b2], "ep01")
        assert len(result) == 1
        assert result[0]["avg_confidence"] == pytest.approx(0.95)
        assert result[0]["combined_text"] == "我爱你China!"

    def test_empty_input_returns_empty(self, tmp_path):
        al = self._aligner(tmp_path)
        assert al._dedup_rescue_blocks([], "ep01") == []

    def test_single_block_returned_unchanged(self, tmp_path):
        al = self._aligner(tmp_path)
        b = self._blk(0.0, 1.0, "只有一个")
        result = al._dedup_rescue_blocks([b], "ep01")
        assert len(result) == 1
        assert result[0]["combined_text"] == "只有一个"


# ─────────────────────────────────────────────────────────────────────────────
#  Group 5 — same_lang integration (run_episode round-trip)
# ─────────────────────────────────────────────────────────────────────────────

class TestSameLangIntegration:
    def _aligner(self, tmp: Path, *, rescue: bool = False) -> OverlapAligner:
        cfg = _cfg(tmp)
        cfg["same_lang"]["alignment"]["asr_sparse_rescue"] = rescue
        return OverlapAligner(cfg)

    def test_happy_path_context_found(self, tmp_path):
        """ASR and OCR overlap → context_available=True, all fields correct."""
        al = self._aligner(tmp_path)
        ep = "ep01"
        asr_p = tmp_path / "asr" / f"{ep}_asr.json"
        ocr_p = tmp_path / "ocr" / f"{ep}_ocr.json"
        _write_asr(asr_p, [_asr_seg(0, 0.0, 2.0, "你好世界", prob=0.95)])
        _write_ocr_same(ocr_p, [_ocr_block(0.5, 1.5, "Hello World", conf=0.88)])

        out = al.run_episode(ep, asr_path=asr_p, ocr_path=ocr_p)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))

        assert data["mode"] == "same_lang"
        assert data["segment_count"] == 1
        s = data["segments"][0]
        assert s["segment_id"] == "ep01_0000"
        assert s["master_text"] == "你好世界"
        assert s["master_source"] == "asr"
        assert s["master_confidence"] == pytest.approx(0.95)
        assert s["context_available"] is True
        assert s["context_text"] == "Hello World"

    def test_no_ocr_temporal_overlap_context_unavailable(self, tmp_path):
        """OCR block outside ASR time window → context_available=False."""
        al = self._aligner(tmp_path)
        ep = "ep02"
        asr_p = tmp_path / "asr" / f"{ep}_asr.json"
        ocr_p = tmp_path / "ocr" / f"{ep}_ocr.json"
        _write_asr(asr_p, [_asr_seg(0, 0.0, 2.0, "你好")])
        _write_ocr_same(ocr_p, [_ocr_block(5.0, 7.0, "后面字幕")])  # no overlap

        out = al.run_episode(ep, asr_path=asr_p, ocr_path=ocr_p)
        s = json.loads(out.read_text(encoding="utf-8"))["segments"][0]
        assert s["context_available"] is False
        assert s["context_text"] == ""

    def test_cache_hit_returns_existing_file_unchanged(self, tmp_path):
        """Second call returns cached path; no recomputation."""
        al = self._aligner(tmp_path)
        ep = "ep03"
        aligned_dir = tmp_path / "aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        cached = aligned_dir / f"{ep}_aligned.json"
        cached.write_text('{"sentinel": true}', encoding="utf-8")

        out = al.run_episode(ep)   # no asr/ocr paths needed — hits cache
        assert out == cached
        assert json.loads(out.read_text())["sentinel"] is True

    def test_hallucination_risk_propagated(self, tmp_path):
        """ASR hallucination_risk=True is preserved in aligned output."""
        al = self._aligner(tmp_path)
        ep = "ep04"
        asr_p = tmp_path / "asr" / f"{ep}_asr.json"
        ocr_p = tmp_path / "ocr" / f"{ep}_ocr.json"
        _write_asr(asr_p, [_asr_seg(0, 0.0, 2.0, "幻觉", hallucination=True)])
        _write_ocr_same(ocr_p, [_ocr_block(0.5, 1.5, "Real")])

        out = al.run_episode(ep, asr_path=asr_p, ocr_path=ocr_p)
        s = json.loads(out.read_text(encoding="utf-8"))["segments"][0]
        assert s["hallucination_risk"] is True

    def test_best_scoring_candidate_is_context_text(self, tmp_path):
        """When multiple OCR candidates qualify, highest-score one sets context_text."""
        al = self._aligner(tmp_path)
        ep = "ep05"
        asr_p = tmp_path / "asr" / f"{ep}_asr.json"
        ocr_p = tmp_path / "ocr" / f"{ep}_ocr.json"
        _write_asr(asr_p, [_asr_seg(0, 0.0, 4.0, "长段落")])
        # Block A [1,5]: abs_ov = min(4,5)-max(0,1) = 4-1 = 3; score = 3/min(4,4) = 0.75
        # Block B [0.5,1.5]: abs_ov = min(4,1.5)-max(0,0.5) = 1.5-0.5 = 1; score = 1/min(4,1) = 1.0
        _write_ocr_same(ocr_p, [
            _ocr_block(1.0, 5.0, "低分候选"),
            _ocr_block(0.5, 1.5, "高分候选"),
        ])

        out = al.run_episode(ep, asr_path=asr_p, ocr_path=ocr_p)
        s = json.loads(out.read_text(encoding="utf-8"))["segments"][0]
        assert len(s["ocr_candidates"]) == 2
        assert s["context_text"] == "高分候选"

    def test_sparse_asr_rescue_promotes_ocr_to_master(self, tmp_path):
        """Fewer than asr_min_segments (10) ASR segs + ≥5 OCR blocks → OCR becomes master."""
        al = self._aligner(tmp_path, rescue=True)
        ep = "ep06"
        asr_p = tmp_path / "asr" / f"{ep}_asr.json"
        ocr_p = tmp_path / "ocr" / f"{ep}_ocr.json"
        # Only 3 ASR segments — below default threshold of 10
        _write_asr(asr_p, [
            _asr_seg(i, float(i * 5), float(i * 5 + 2), f"asr{i}") for i in range(3)
        ])
        # 5 OCR blocks → triggers rescue
        _write_ocr_same(ocr_p, [
            _ocr_block(float(i * 5), float(i * 5 + 2), f"ocr字幕{i}") for i in range(5)
        ])

        out = al.run_episode(ep, asr_path=asr_p, ocr_path=ocr_p)
        data = json.loads(out.read_text(encoding="utf-8"))
        segs = data["segments"]
        assert len(segs) == 5
        assert all(s["master_source"] == "ocr" for s in segs)
        assert all(s["hallucination_risk"] is False for s in segs)


# ─────────────────────────────────────────────────────────────────────────────
#  Group 6 — cross_lang integration
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossLangIntegration:
    def _aligner(self, tmp: Path) -> OverlapAligner:
        return OverlapAligner(_cfg(tmp, mode="cross_lang"))

    def test_multiple_asr_segments_concatenated(self, tmp_path):
        """Two ASR segs overlapping one OCR seg → joined in chronological order."""
        al = self._aligner(tmp_path)
        ep = "ep01"
        asr_p = tmp_path / "asr" / f"{ep}_asr.json"
        tl_p  = tmp_path / "ocr" / "ocr_timeline" / f"{ep}_timeline.json"
        _write_asr(asr_p, [
            _asr_seg(0, 0.5, 1.5, "你好"),
            _asr_seg(1, 1.5, 2.5, "世界"),
        ])
        _write_ocr_timeline(tl_p, [_ocr_tl_seg(0, 0.0, 3.0, "Hello World")])

        out = al.run_episode(ep, asr_path=asr_p, ocr_path=tl_p)
        s = json.loads(out.read_text(encoding="utf-8"))["segments"][0]
        assert s["master_text"] == "Hello World"
        assert s["master_source"] == "ocr"
        assert s["context_available"] is True
        assert s["context_text"] == "你好世界"
        assert s["hallucination_risk"] is False

    def test_no_asr_overlap_context_unavailable(self, tmp_path):
        """ASR completely outside OCR window → context_available=False."""
        al = self._aligner(tmp_path)
        ep = "ep02"
        asr_p = tmp_path / "asr" / f"{ep}_asr.json"
        tl_p  = tmp_path / "ocr" / "ocr_timeline" / f"{ep}_timeline.json"
        _write_asr(asr_p, [_asr_seg(0, 5.0, 6.0, "不重叠")])
        _write_ocr_timeline(tl_p, [_ocr_tl_seg(0, 0.0, 2.0, "No match")])

        out = al.run_episode(ep, asr_path=asr_p, ocr_path=tl_p)
        s = json.loads(out.read_text(encoding="utf-8"))["segments"][0]
        assert s["context_available"] is False
        assert s["context_text"] == ""

    def test_ocr_candidates_always_empty_in_cross_lang(self, tmp_path):
        """ocr_candidates list is N/A in cross_lang and must always be []."""
        al = self._aligner(tmp_path)
        ep = "ep03"
        asr_p = tmp_path / "asr" / f"{ep}_asr.json"
        tl_p  = tmp_path / "ocr" / "ocr_timeline" / f"{ep}_timeline.json"
        _write_asr(asr_p, [_asr_seg(0, 0.0, 2.0, "你好")])
        _write_ocr_timeline(tl_p, [_ocr_tl_seg(0, 0.0, 2.0, "Hello")])

        out = al.run_episode(ep, asr_path=asr_p, ocr_path=tl_p)
        s = json.loads(out.read_text(encoding="utf-8"))["segments"][0]
        assert s["ocr_candidates"] == []
