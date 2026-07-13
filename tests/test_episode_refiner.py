"""
Unit tests for src.execution.episode_refiner.

Mocking strategy
────────────────
  LLMClient.complete()           → MagicMock with controlled side_effect
  EpisodeRefiner._render_prompt  → patched via patch.object to return a plain
                                   string, keeping tests independent of template
                                   files under config/prompts/
  All file I/O uses pytest tmp_path.

Coverage
────────
  Group 1  _fmt_ts                   — timestamp formatting + overflow guard
  Group 2  _strip_markdown_fence     — fence stripping variants
  Group 3  _merge_artifact_fragments — micro-duration / gap merge + renumber
  Group 4  _merge_short_fragments    — sub-300ms forward-merge, multi-pass
  Group 5  _clip_overlapping_ends    — overlap clipping, early-return identity
  Group 6  EpisodeRefiner._format_segments
             · id sequencing
             · context_available gate
             · stale OCR guard (ctx seen in recent window → suppressed)
             · hallucination guard (master repeated ≥3 times → ctx promoted)
  Group 7  EpisodeRefiner._assemble_fallback_srt
             · valid SRT output
             · empty master_text segments skipped
             · timestamps in SRT format
  Group 8  EpisodeRefiner._build_correction_prompt
             · CORRECTION REQUEST header present
             · error message embedded
             · bad-output snippet capped at 800 chars
  Group 9  EpisodeRefiner.refine_episode  (mocked LLM)
             · cache hit → no LLM call
             · LLM #1 valid → SRT written, 1 call
             · LLM #1 invalid, LLM #2 valid → SRT written, 2 calls
             · both invalid → fallback SRT written
             · both invalid → validation_report.json updated
             · LLMCallError on #1 → fallback (1 attempted call)
             · empty segments → empty SRT, no LLM call
  Group 10 EpisodeRefiner._filter_watermarks
             · matching segment removed
             · no-match: all kept
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.execution.episode_refiner import (
    EpisodeRefiner,
    _clip_overlapping_ends,
    _fmt_ts,
    _merge_artifact_fragments,
    _merge_short_fragments,
    _strip_markdown_fence,
)
from src.utils.llm_client import LLMCallError


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SRT = (
    "1\n"
    "00:00:01,000 --> 00:00:03,000\n"
    "你好世界\n"
    "\n"
    "2\n"
    "00:00:04,000 --> 00:00:06,000\n"
    "再见"
)

_INVALID_SRT = "not a valid srt string at all"


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(tmp: Path) -> dict:
    return {
        "pipeline": {"mode": "same_lang", "source_language": "zh"},
        "same_lang": {"asr": {}},
        "paths": {"output_dir": str(tmp / "output")},
        "execution": {"prompts": {}},
    }


def _make_refiner(tmp: Path) -> EpisodeRefiner:
    """Create a minimal EpisodeRefiner with a no-op mock LLM."""
    return EpisodeRefiner(MagicMock(), _cfg(tmp), {"characters": {}})


def _write_aligned(tmp: Path, ep_id: str, segs: list[dict]) -> Path:
    p = tmp / f"{ep_id}_aligned.json"
    p.write_text(json.dumps({"segments": segs}), encoding="utf-8")
    return p


def _seg(start: float, end: float, text: str,
         ctx: str = "", ctx_avail: bool = False) -> dict:
    return {
        "start": start, "end": end,
        "master_text": text,
        "context_text": ctx,
        "context_available": ctx_avail,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Group 1 — _fmt_ts
# ─────────────────────────────────────────────────────────────────────────────

class TestFmtTs:
    def test_zero(self):
        assert _fmt_ts(0.0) == "00:00:00,000"

    def test_hh_mm_ss_ms(self):
        # 1 h + 2 min + 3 s + 500 ms
        assert _fmt_ts(3723.5) == "01:02:03,500"

    def test_negative_clamped_to_zero(self):
        assert _fmt_ts(-10.0) == "00:00:00,000"

    def test_millisecond_overflow_guard(self):
        # 0.9995 → ms rounds to 1000 → guard bumps s by 1, ms → 0
        assert _fmt_ts(0.9995) == "00:00:01,000"

    def test_sub_minute_boundary(self):
        assert _fmt_ts(59.999) == "00:00:59,999"


# ─────────────────────────────────────────────────────────────────────────────
#  Group 2 — _strip_markdown_fence
# ─────────────────────────────────────────────────────────────────────────────

class TestStripMarkdownFence:
    _INNER = "1\n00:00:01,000 --> 00:00:02,000\nHello"

    def test_srt_label_fence_stripped(self):
        assert _strip_markdown_fence(f"```srt\n{self._INNER}\n```") == self._INNER

    def test_plain_fence_stripped(self):
        assert _strip_markdown_fence(f"```\n{self._INNER}\n```") == self._INNER

    def test_uppercase_srt_label_stripped(self):
        assert _strip_markdown_fence(f"```SRT\n{self._INNER}\n```") == self._INNER

    def test_no_fence_unchanged(self):
        assert _strip_markdown_fence(self._INNER) == self._INNER


# ─────────────────────────────────────────────────────────────────────────────
#  Group 3 — _merge_artifact_fragments
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeArtifactFragments:
    def test_micro_duration_block_merged(self):
        """First block < 200 ms with same text → both blocks collapse into one."""
        srt = (
            "1\n00:00:01,000 --> 00:00:01,100\n你好\n\n"   # 100 ms < 200
            "2\n00:00:01,200 --> 00:00:03,000\n你好"
        )
        result = _merge_artifact_fragments(srt)
        assert "00:00:01,000 --> 00:00:03,000" in result
        assert result.count("\n\n") == 0       # single block

    def test_gap_leq_3s_identical_text_merged(self):
        """Both blocks ≥ 200 ms but gap ≤ 3 s and same text → merged."""
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
            "2\n00:00:04,500 --> 00:00:05,500\n你好"       # gap = 2500 ms
        )
        result = _merge_artifact_fragments(srt)
        assert "00:00:01,000 --> 00:00:05,500" in result
        assert result.count("\n\n") == 0

    def test_different_texts_not_merged(self):
        """Different text → two blocks remain."""
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
            "2\n00:00:02,100 --> 00:00:03,000\n世界"
        )
        result = _merge_artifact_fragments(srt)
        assert "你好" in result
        assert "世界" in result
        assert result.count("\n\n") == 1

    def test_renumbered_after_merge(self):
        """After blocks 1+2 merge, remaining block gets number 2."""
        srt = (
            "1\n00:00:01,000 --> 00:00:01,100\n你好\n\n"   # <200ms
            "2\n00:00:01,200 --> 00:00:02,000\n你好\n\n"   # merges with #1
            "3\n00:00:03,000 --> 00:00:04,000\n世界"
        )
        result = _merge_artifact_fragments(srt)
        blocks = [b.strip() for b in result.split("\n\n") if b.strip()]
        assert len(blocks) == 2
        assert blocks[0].startswith("1\n")
        assert blocks[1].startswith("2\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Group 4 — _merge_short_fragments
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeShortFragments:
    def test_short_block_merged_with_next(self):
        """Block < 300 ms → merged with next; texts concatenated."""
        srt = (
            "1\n00:00:01,000 --> 00:00:01,200\n你\n\n"     # 200 ms < 300
            "2\n00:00:01,200 --> 00:00:03,000\n好世界"
        )
        result = _merge_short_fragments(srt)
        assert "你好世界" in result
        assert "00:00:01,000 --> 00:00:03,000" in result
        assert result.count("\n\n") == 0

    def test_normal_block_unchanged(self):
        """Block ≥ 300 ms → not merged."""
        srt = "1\n00:00:01,000 --> 00:00:01,500\n你好"     # 500 ms
        result = _merge_short_fragments(srt)
        assert "00:00:01,000 --> 00:00:01,500" in result

    def test_two_consecutive_short_blocks_multi_pass(self):
        """A(100ms) + B(100ms) + C(1.8s): two passes merge all into ABC."""
        srt = (
            "1\n00:00:01,000 --> 00:00:01,100\nA\n\n"
            "2\n00:00:01,100 --> 00:00:01,200\nB\n\n"
            "3\n00:00:01,200 --> 00:00:03,000\nC"
        )
        result = _merge_short_fragments(srt)
        assert "ABC" in result
        assert "00:00:01,000 --> 00:00:03,000" in result
        assert result.count("\n\n") == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Group 5 — _clip_overlapping_ends
# ─────────────────────────────────────────────────────────────────────────────

class TestClipOverlappingEnds:
    def test_overlapping_end_clipped_to_next_start(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:03,000\n你好\n\n"
            "2\n00:00:02,500 --> 00:00:04,000\n世界"
        )
        result = _clip_overlapping_ends(srt)
        assert "00:00:01,000 --> 00:00:02,500" in result   # clipped
        assert "00:00:02,500 --> 00:00:04,000" in result   # unchanged

    def test_no_overlap_returns_same_object(self):
        """When no clipping occurs the original string object is returned (early exit)."""
        result = _clip_overlapping_ends(_VALID_SRT)
        assert result is _VALID_SRT

    def test_empty_string_returned_unchanged(self):
        assert _clip_overlapping_ends("") == ""


# ─────────────────────────────────────────────────────────────────────────────
#  Group 6 — EpisodeRefiner._format_segments
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatSegments:
    def test_ids_start_at_one_and_increment(self, tmp_path):
        r = _make_refiner(tmp_path)
        data = json.loads(r._format_segments([_seg(0.0, 1.0, "A"), _seg(1.0, 2.0, "B")]))
        assert data[0]["id"] == 1
        assert data[1]["id"] == 2

    def test_no_context_key_when_unavailable(self, tmp_path):
        r = _make_refiner(tmp_path)
        data = json.loads(r._format_segments([_seg(0.0, 1.0, "你好", ctx_avail=False)]))
        assert "context_text" not in data[0]

    def test_context_key_included_when_available(self, tmp_path):
        r = _make_refiner(tmp_path)
        data = json.loads(r._format_segments([
            _seg(0.0, 1.0, "你好", ctx="Hello", ctx_avail=True)
        ]))
        assert data[0]["context_text"] == "Hello"

    def test_stale_ocr_guard_suppresses_repeated_context(self, tmp_path):
        """ctx seen as context in the previous segment → suppressed for this segment."""
        r = _make_refiner(tmp_path)
        segs = [
            _seg(0.0, 1.0, "台词一", ctx="字幕A", ctx_avail=True),
            _seg(1.0, 2.0, "台词二", ctx="字幕A", ctx_avail=True),  # stale
        ]
        data = json.loads(r._format_segments(segs))
        assert "context_text" in data[0]        # first: fresh, allowed
        assert "context_text" not in data[1]    # second: stale, suppressed

    def test_hallucination_guard_promotes_ctx_to_master(self, tmp_path):
        """After master phrase appears 3 times, 4th occurrence uses ctx as master_text."""
        r = _make_refiner(tmp_path)
        repeated = "反复出现的台词!"    # len=9 ≥ 8

        # Use different ctx values so the stale OCR guard does not interfere
        segs = [
            _seg(float(i), float(i + 1), repeated, ctx=f"ctx{i}", ctx_avail=True)
            for i in range(3)
        ] + [
            _seg(3.0, 4.0, repeated, ctx="正确字幕", ctx_avail=True)  # 4th → guard fires
        ]
        data = json.loads(r._format_segments(segs))
        assert data[3]["master_text"] == "正确字幕"

    def test_hallucination_risk_flag_promotes_ctx_on_first_occurrence(self, tmp_path):
        """hallucination_risk=True overrides master even before phrase repeats 3 times."""
        r = _make_refiner(tmp_path)
        seg = {
            "start": 1.0, "end": 2.0,
            "master_text": "低置信度的识别",
            "context_text": "真实字幕内容",
            "context_available": True,
            "hallucination_risk": True,
        }
        data = json.loads(r._format_segments([seg]))
        assert data[0]["master_text"] == "真实字幕内容"

    def test_hallucination_risk_false_keeps_master(self, tmp_path):
        """hallucination_risk=False does not trigger substitution (phrase only seen once)."""
        r = _make_refiner(tmp_path)
        seg = {
            "start": 1.0, "end": 2.0,
            "master_text": "正常识别结果",
            "context_text": "不同的字幕",
            "context_available": True,
            "hallucination_risk": False,
        }
        data = json.loads(r._format_segments([seg]))
        assert data[0]["master_text"] == "正常识别结果"

    def test_hallucination_risk_without_ctx_keeps_master(self, tmp_path):
        """hallucination_risk=True but no valid OCR → master unchanged (no OCR to fall back to)."""
        r = _make_refiner(tmp_path)
        seg = {
            "start": 1.0, "end": 2.0,
            "master_text": "低置信度但无字幕",
            "context_text": "",
            "context_available": False,
            "hallucination_risk": True,
        }
        data = json.loads(r._format_segments([seg]))
        assert data[0]["master_text"] == "低置信度但无字幕"


# ─────────────────────────────────────────────────────────────────────────────
#  Group 7 — EpisodeRefiner._assemble_fallback_srt
# ─────────────────────────────────────────────────────────────────────────────

class TestAssembleFallbackSrt:
    def test_produces_valid_srt(self, tmp_path):
        from src.execution.srt_validator import SRTValidator
        r = _make_refiner(tmp_path)
        srt = r._assemble_fallback_srt([_seg(1.0, 3.0, "你好"), _seg(4.0, 6.0, "世界")])
        ok, reason = SRTValidator().validate_srt_string(srt)
        assert ok, f"Fallback SRT should be valid, got: {reason}"

    def test_empty_master_text_segments_skipped(self, tmp_path):
        r = _make_refiner(tmp_path)
        segs = [
            _seg(0.0, 1.0, ""),         # empty → skip
            _seg(1.0, 3.0, "你好"),
        ]
        srt = r._assemble_fallback_srt(segs)
        blocks = [b.strip() for b in srt.split("\n\n") if b.strip()]
        assert len(blocks) == 1
        assert blocks[0].startswith("1\n")   # numbering starts at 1, no gap
        assert "你好" in srt

    def test_timestamps_converted_to_srt_format(self, tmp_path):
        r = _make_refiner(tmp_path)
        srt = r._assemble_fallback_srt([_seg(3723.5, 3725.0, "测试")])
        assert "01:02:03,500 --> 01:02:05,000" in srt


# ─────────────────────────────────────────────────────────────────────────────
#  Group 8 — EpisodeRefiner._build_correction_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCorrectionPrompt:
    def test_contains_correction_header(self, tmp_path):
        r = _make_refiner(tmp_path)
        prompt = r._build_correction_prompt("orig", "bad", "some error", "ep01")
        assert "CORRECTION REQUEST" in prompt

    def test_contains_error_message(self, tmp_path):
        r = _make_refiner(tmp_path)
        error = "sequence number out of order — expected 2, got 5"
        prompt = r._build_correction_prompt("orig", "bad", error, "ep01")
        assert error in prompt

    def test_bad_output_snippet_capped_at_800_chars(self, tmp_path):
        r = _make_refiner(tmp_path)
        long_bad = "X" * 2000
        prompt = r._build_correction_prompt("orig", long_bad, "error", "ep01")
        assert "X" * 800 in prompt
        assert "X" * 801 not in prompt


# ─────────────────────────────────────────────────────────────────────────────
#  Group 9 — EpisodeRefiner.refine_episode  (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestRefineEpisode:
    _SEGS = [_seg(1.0, 3.0, "你好世界")]

    def _run(
        self,
        tmp: Path,
        llm_responses: list,
        segs: list[dict] | None = None,
    ) -> tuple[str, MagicMock]:
        """Write aligned JSON, build refiner with controlled LLM, run refine_episode."""
        segs = segs if segs is not None else self._SEGS

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = llm_responses
        refiner = EpisodeRefiner(mock_llm, _cfg(tmp), {"characters": {}})

        aligned = _write_aligned(tmp, "ep01", segs)
        with patch.object(refiner, "_render_prompt", return_value="dummy prompt"):
            result = refiner.refine_episode(str(aligned))
        return result, mock_llm

    def test_cache_hit_no_llm_call(self, tmp_path):
        """If SRT already exists, return it immediately without calling LLM."""
        mock_llm = MagicMock()
        refiner = EpisodeRefiner(mock_llm, _cfg(tmp_path), {"characters": {}})

        cn_dir = tmp_path / "output" / "cn"
        cn_dir.mkdir(parents=True, exist_ok=True)
        cached = cn_dir / "ep01.srt"
        cached.write_text(_VALID_SRT, encoding="utf-8")

        aligned = _write_aligned(tmp_path, "ep01", self._SEGS)
        result = refiner.refine_episode(str(aligned))

        assert result == str(cached)
        mock_llm.complete.assert_not_called()

    def test_first_attempt_valid_writes_srt(self, tmp_path):
        """LLM #1 returns valid SRT → file written after exactly one LLM call."""
        result, mock_llm = self._run(tmp_path, [_VALID_SRT])

        assert Path(result).exists()
        mock_llm.complete.assert_called_once()

    def test_first_invalid_retry_valid_writes_srt(self, tmp_path):
        """LLM #1 invalid → correction retry → LLM #2 valid → file written, 2 calls."""
        result, mock_llm = self._run(tmp_path, [_INVALID_SRT, _VALID_SRT])

        assert Path(result).exists()
        assert mock_llm.complete.call_count == 2

    def test_both_invalid_writes_fallback_srt(self, tmp_path):
        """Both LLM calls invalid → fallback SRT from raw master_text is written."""
        result, mock_llm = self._run(tmp_path, [_INVALID_SRT, _INVALID_SRT])

        assert Path(result).exists()
        assert mock_llm.complete.call_count == 2
        content = Path(result).read_text(encoding="utf-8")
        assert "你好世界" in content    # segment master_text is present

    def test_both_invalid_records_validation_report(self, tmp_path):
        """Fallback path writes episode ID to validation_report.json."""
        self._run(tmp_path, [_INVALID_SRT, _INVALID_SRT])

        report_path = tmp_path / "output" / "validation_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert any(f["episode_id"] == "ep01" for f in report["failures"])

    def test_llm_call_error_triggers_fallback(self, tmp_path):
        """LLMCallError on first call → fallback written, one attempted LLM call."""
        result, mock_llm = self._run(tmp_path, [LLMCallError("API down")])

        assert Path(result).exists()
        mock_llm.complete.assert_called_once()

    def test_empty_segments_writes_empty_srt_without_llm(self, tmp_path):
        """No segments in aligned JSON → empty SRT written immediately, no LLM call."""
        result, mock_llm = self._run(tmp_path, [], segs=[])

        assert Path(result).exists()
        assert Path(result).read_text(encoding="utf-8") == ""
        mock_llm.complete.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Group 10 — EpisodeRefiner._filter_watermarks
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterWatermarks:
    def _refiner_with_patterns(self, tmp: Path, patterns: list[str]) -> EpisodeRefiner:
        cfg = {
            "pipeline": {"mode": "same_lang", "source_language": "zh"},
            "same_lang": {"asr": {"watermark_patterns": patterns}},
            "paths": {"output_dir": str(tmp / "output")},
            "execution": {"prompts": {}},
        }
        return EpisodeRefiner(MagicMock(), cfg, {"characters": {}})

    def test_matching_segment_removed(self, tmp_path):
        r = self._refiner_with_patterns(tmp_path, ["抖音"])
        segs = [
            {"master_text": "抖音水印内容", "start": 0.0, "end": 1.0},
            {"master_text": "正常台词",    "start": 1.0, "end": 2.0},
        ]
        result = r._filter_watermarks(segs, "ep01")
        assert len(result) == 1
        assert result[0]["master_text"] == "正常台词"

    def test_no_match_all_segments_kept(self, tmp_path):
        r = self._refiner_with_patterns(tmp_path, ["抖音"])
        segs = [
            {"master_text": "正常台词一", "start": 0.0, "end": 1.0},
            {"master_text": "正常台词二", "start": 1.0, "end": 2.0},
        ]
        assert len(r._filter_watermarks(segs, "ep01")) == 2
