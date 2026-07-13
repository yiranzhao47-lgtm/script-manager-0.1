"""
Unit tests for RhythmAnalyzer (Layer 5 Intelligence).

Coverage:
  • Init: config reading, drama_map directory creation
  • Cache: hit returns cached data, miss triggers LLM
  • SRT loading: file-not-found, truncation to srt_char_limit
  • Conflict chain building: structured_dialogues excluded, natural sort
  • Blueprint reduction: LLM success + failure fallback
  • Assembly: output structure, episode_conflicts ordering
  • Integration run: full run with mocked LLM
  • Thread safety: parallel map calls don't corrupt the ledger

All tests are self-contained and use temporary directories; no live API calls.
"""
from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Shared minimal config
# ---------------------------------------------------------------------------
_CFG = {
    "pipeline": {"mode": "same_lang", "source_language": "zh"},
    "paths": {
        "raw_video_dir": "data/raw",
        "cache_dir":     "data/cache",
        "meta_dir":      "data/meta",
        "output_dir":    "data/output",
    },
    "execution": {
        "llm": {
            "model":       "deepseek-chat",
            "base_url":    "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "max_tokens":  8000,
        },
        "retry": {
            "max_attempts":   1,
            "wait_min_sec":   0,
            "wait_max_sec":   0,
            "no_retry_codes": [400, 401],
        },
    },
    "intelligence": {
        "drama_analysis": {
            "enabled":          True,
            "map_workers":      2,
            "map_max_tokens":   3000,
            "reduce_max_tokens":4000,
            "srt_char_limit":   500,
        }
    },
    "pricing": {
        "deepseek-chat": {
            "input_cost_per_m":  1.0,
            "output_cost_per_m": 2.0,
        }
    },
}

_META = {
    "characters": {
        "林晓": {"canonical_en": "Lin Xiao", "aliases": ["小林"]},
        "陈默": {"canonical_en": "Chen Mo",  "aliases": []},
    }
}


def _make_llm_client():
    from src.utils.llm_client import LLMClient
    return LLMClient(_CFG)


def _scene(scene_id: str, **kwargs) -> dict:
    return {
        "scene_id": scene_id,
        "location": kwargs.get("location", "office"),
        "time": kwargs.get("time", "morning"),
        "scene_start_time": kwargs.get("scene_start_time", "00:01:00,000"),
        "scene_end_time":   kwargs.get("scene_end_time",   "00:03:00,000"),
        "scene_actions": kwargs.get("scene_actions", ["A does something"]),
        "unresolved_debt": kwargs.get("unresolved_debt", "secret remains"),
        "pivot_signals": kwargs.get("pivot_signals", []),
        "structured_dialogues": kwargs.get("structured_dialogues", [
            {"speaker": "林晓", "line": "I know what you did."}
        ]),
    }


def _ep_result(ep_id: str, n_scenes: int = 2) -> dict:
    return {
        "episode_id": ep_id,
        "scenes": [_scene(f"{ep_id}_sc_{i+1:02d}") for i in range(n_scenes)],
    }


_UNSET = object()


def _make_analyzer(tmp: Path, llm_client=None, meta=_UNSET):
    from src.intelligence.rhythm_analyzer import RhythmAnalyzer
    client = llm_client or _make_llm_client()
    actual_meta = _META if meta is _UNSET else meta
    return RhythmAnalyzer(
        cfg=_CFG,
        llm_client=client,
        output_dir=tmp / "output",
        cache_dir=tmp / "cache",
        meta=actual_meta,
    )


# ===========================================================================
# Test: __init__ — config reading and directory creation
# ===========================================================================

class TestInit(unittest.TestCase):

    def test_drama_map_dir_created(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            self.assertTrue((tmp / "cache" / "drama_map").is_dir())

    def test_config_values_loaded(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            self.assertEqual(analyzer._map_workers,       2)
            self.assertEqual(analyzer._map_max_tokens,    3000)
            self.assertEqual(analyzer._reduce_max_tokens, 4000)
            self.assertEqual(analyzer._srt_char_limit,    500)

    def test_characters_loaded_from_meta(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp, meta=_META)
            self.assertIn("林晓", analyzer._characters)
            self.assertIn("陈默", analyzer._characters)

    def test_empty_meta_gives_empty_characters(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp, meta={})
            self.assertEqual(analyzer._characters, {})

    def test_none_meta_gives_empty_characters(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp, meta=None)
            self.assertEqual(analyzer._characters, {})


# ===========================================================================
# Test: SRT loading
# ===========================================================================

class TestSRTLoading(unittest.TestCase):

    def _write_srt(self, srt_dir: Path, ep_id: str, content: str) -> Path:
        srt_dir.mkdir(parents=True, exist_ok=True)
        p = srt_dir / f"{ep_id}.srt"
        p.write_text(content, encoding="utf-8")
        return p

    def test_load_existing_srt(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            self._write_srt(tmp / "output" / "cn", "ep01", "1\n00:00:01,000 --> 00:00:02,000\nHello\n")
            content = analyzer._load_srt("ep01")
            self.assertIn("Hello", content)

    def test_missing_srt_raises_file_not_found(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            with self.assertRaises(FileNotFoundError):
                analyzer._load_srt("ep99")

    def test_srt_truncated_to_char_limit(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            # srt_char_limit is 500 in test config
            long_content = "A" * 1000
            self._write_srt(tmp / "output" / "cn", "ep01", long_content)
            result = analyzer._load_srt("ep01")
            self.assertEqual(len(result), 500)

    def test_short_srt_not_truncated(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            short_content = "B" * 100
            self._write_srt(tmp / "output" / "cn", "ep01", short_content)
            result = analyzer._load_srt("ep01")
            self.assertEqual(len(result), 100)


# ===========================================================================
# Test: _analyze_one_episode — cache hit / miss
# ===========================================================================

class TestAnalyzeOneEpisode(unittest.TestCase):

    def test_cache_hit_returns_cached_data(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            cached = _ep_result("ep01")
            cache_path = tmp / "cache" / "drama_map" / "ep01_conflict.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached), encoding="utf-8")

            result = analyzer._analyze_one_episode("ep01")
            self.assertEqual(result["episode_id"], "ep01")
            self.assertEqual(len(result["scenes"]), 2)

    def test_cache_hit_does_not_call_llm(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            mock_llm = MagicMock()
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)

            cached = _ep_result("ep02")
            cache_path = tmp / "cache" / "drama_map" / "ep02_conflict.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached), encoding="utf-8")

            analyzer._analyze_one_episode("ep02")
            mock_llm.complete.assert_not_called()

    def test_corrupt_cache_triggers_llm(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            # Write an SRT so _load_srt doesn't fail
            srt_dir = tmp / "output" / "cn"
            srt_dir.mkdir(parents=True)
            (srt_dir / "ep03.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")

            mock_llm = MagicMock()
            mock_llm.complete.return_value = json.dumps(_ep_result("ep03"))
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)

            # Write corrupt cache
            cache_path = tmp / "cache" / "drama_map" / "ep03_conflict.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("not valid json", encoding="utf-8")

            result = analyzer._analyze_one_episode("ep03")
            mock_llm.complete.assert_called_once()
            self.assertEqual(result["episode_id"], "ep03")

    def test_cache_miss_writes_new_cache(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            srt_dir = tmp / "output" / "cn"
            srt_dir.mkdir(parents=True)
            (srt_dir / "ep04.srt").write_text("subtitle content", encoding="utf-8")

            mock_llm = MagicMock()
            mock_llm.complete.return_value = json.dumps(_ep_result("ep04"))
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)

            analyzer._analyze_one_episode("ep04")
            cache_path = tmp / "cache" / "drama_map" / "ep04_conflict.json"
            self.assertTrue(cache_path.exists())

    def test_llm_missing_scenes_raises_value_error(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            srt_dir = tmp / "output" / "cn"
            srt_dir.mkdir(parents=True)
            (srt_dir / "ep05.srt").write_text("content", encoding="utf-8")

            mock_llm = MagicMock()
            mock_llm.complete.return_value = json.dumps({"episode_id": "ep05"})
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)

            with self.assertRaises(ValueError):
                analyzer._analyze_one_episode("ep05")


# ===========================================================================
# Test: _build_conflict_chain — structured_dialogues excluded, natural sort
# ===========================================================================

class TestBuildConflictChain(unittest.TestCase):

    def test_structured_dialogues_excluded(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            conflict_map = {
                "ep01": _ep_result("ep01"),
                "ep02": _ep_result("ep02"),
            }
            chain = analyzer._build_conflict_chain(conflict_map)
            self.assertNotIn("structured_dialogues", chain)
            self.assertNotIn("I know what you did", chain)

    def test_scene_id_in_chain(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            conflict_map = {"ep01": _ep_result("ep01", n_scenes=1)}
            chain = analyzer._build_conflict_chain(conflict_map)
            self.assertIn("ep01_sc_01", chain)

    def test_unresolved_debt_in_chain(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            ep = {
                "episode_id": "ep01",
                "scenes": [_scene("ep01_sc_01", unresolved_debt="secret contract")],
            }
            chain = analyzer._build_conflict_chain({"ep01": ep})
            self.assertIn("secret contract", chain)

    def test_natural_sort_order(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            # ep2 must appear before ep10 in natural sort
            conflict_map = {
                "ep10": _ep_result("ep10"),
                "ep2":  _ep_result("ep2"),
                "ep1":  _ep_result("ep1"),
            }
            chain = analyzer._build_conflict_chain(conflict_map)
            pos_ep1  = chain.index("=== ep1 ===")
            pos_ep2  = chain.index("=== ep2 ===")
            pos_ep10 = chain.index("=== ep10 ===")
            self.assertLess(pos_ep1,  pos_ep2)
            self.assertLess(pos_ep2,  pos_ep10)

    def test_pivot_signals_in_chain(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            ep = {
                "episode_id": "ep01",
                "scenes": [_scene("ep01_sc_01", pivot_signals=["evidence found"])],
            }
            chain = analyzer._build_conflict_chain({"ep01": ep})
            self.assertIn("evidence found", chain)

    def test_null_debt_omitted_from_chain(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            ep = {
                "episode_id": "ep01",
                "scenes": [_scene("ep01_sc_01", unresolved_debt=None)],
            }
            chain = analyzer._build_conflict_chain({"ep01": ep})
            self.assertNotIn("DEBT:", chain)


# ===========================================================================
# Test: _reduce_to_blueprint — LLM success and failure fallback
# ===========================================================================

class TestReduceToBlueprint(unittest.TestCase):

    def test_returns_llm_dict_on_success(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            expected = {"debt_chain_narrative": "The secret builds.", "first_pinch": {}}
            mock_llm = MagicMock()
            mock_llm.complete.return_value = json.dumps(expected)
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)

            conflict_map = {"ep01": _ep_result("ep01")}
            result = analyzer._reduce_to_blueprint(conflict_map)
            self.assertEqual(result["debt_chain_narrative"], "The secret builds.")

    def test_returns_empty_dict_on_llm_failure(self):
        from src.utils.llm_client import LLMCallError
        with TemporaryDirectory() as td:
            tmp = Path(td)
            mock_llm = MagicMock()
            mock_llm.complete.side_effect = LLMCallError("timeout")
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)

            result = analyzer._reduce_to_blueprint({"ep01": _ep_result("ep01")})
            self.assertEqual(result, {})

    def test_returns_empty_dict_on_non_json_response(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            mock_llm = MagicMock()
            mock_llm.complete.return_value = "Sorry, I cannot assist with that."
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)

            result = analyzer._reduce_to_blueprint({"ep01": _ep_result("ep01")})
            self.assertEqual(result, {})


# ===========================================================================
# Test: _assemble_and_write — output structure and file persistence
# ===========================================================================

class TestAssembleAndWrite(unittest.TestCase):

    def test_output_file_created(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            conflict_map = {"ep01": _ep_result("ep01"), "ep02": _ep_result("ep02")}
            analyzer._assemble_and_write(conflict_map, {"first_pinch": {}}, 5)
            self.assertTrue((tmp / "output" / "drama_structure_graph.json").exists())

    def test_output_structure_keys(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            conflict_map = {"ep01": _ep_result("ep01")}
            result = analyzer._assemble_and_write(conflict_map, {}, 3)
            for key in ("total_episodes_analysed", "total_episodes_in_series",
                        "macro_blueprint", "episode_conflicts"):
                self.assertIn(key, result)

    def test_total_episodes_counts(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            conflict_map = {"ep01": _ep_result("ep01"), "ep02": _ep_result("ep02")}
            result = analyzer._assemble_and_write(conflict_map, {}, 10)
            self.assertEqual(result["total_episodes_analysed"],    2)
            self.assertEqual(result["total_episodes_in_series"],  10)

    def test_episode_conflicts_naturally_sorted(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            conflict_map = {
                "ep10": _ep_result("ep10"),
                "ep2":  _ep_result("ep2"),
                "ep1":  _ep_result("ep1"),
            }
            result = analyzer._assemble_and_write(conflict_map, {}, 10)
            keys = list(result["episode_conflicts"].keys())
            self.assertEqual(keys, ["ep1", "ep2", "ep10"])

    def test_structured_dialogues_preserved_in_output(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            ep = {
                "episode_id": "ep01",
                "scenes": [_scene("ep01_sc_01", structured_dialogues=[
                    {"speaker": "林晓", "line": "I know."}
                ])],
            }
            result = analyzer._assemble_and_write({"ep01": ep}, {}, 1)
            dialogues = result["episode_conflicts"]["ep01"]["scenes"][0]["structured_dialogues"]
            self.assertEqual(dialogues[0]["speaker"], "林晓")

    def test_output_json_is_valid_utf8(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            analyzer._assemble_and_write({"ep01": _ep_result("ep01")}, {}, 1)
            raw = (tmp / "output" / "drama_structure_graph.json").read_bytes()
            raw.decode("utf-8")  # must not raise


# ===========================================================================
# Test: run() — integration with mocked LLM
# ===========================================================================

class TestRunIntegration(unittest.TestCase):

    def _setup_srt(self, output_dir: Path, ep_ids: list[str]) -> None:
        # source_language="zh" → SRTs live in cn/ subdir
        srt_dir = output_dir / "cn"
        srt_dir.mkdir(parents=True, exist_ok=True)
        for ep_id in ep_ids:
            (srt_dir / f"{ep_id}.srt").write_text(
                f"1\n00:00:01,000 --> 00:00:02,000\nContent for {ep_id}\n",
                encoding="utf-8",
            )

    def test_run_produces_graph_json(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            ep_ids = ["ep01", "ep02"]
            self._setup_srt(tmp / "output", ep_ids)

            blueprint = {"debt_chain_narrative": "builds", "first_pinch": {}, "second_pinch": {}}

            mock_llm = MagicMock()
            def _llm_side_effect(system, user, max_tokens=None,
                                 json_mode=False, module_name="default"):
                if module_name == "Drama_Analysis":
                    ep_id = "ep01" if "ep01" in user else "ep02"
                    return json.dumps(_ep_result(ep_id))
                return json.dumps(blueprint)

            mock_llm.complete.side_effect = _llm_side_effect
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)
            result = analyzer.run(ep_ids)
            self.assertIn("episode_conflicts", result)
            self.assertIn("macro_blueprint", result)
            self.assertTrue((tmp / "output" / "drama_structure_graph.json").exists())

    def test_run_empty_episode_ids_raises(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            with self.assertRaises(ValueError):
                analyzer.run([])

    def test_run_all_episodes_fail_raises_runtime_error(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            mock_llm = MagicMock()
            mock_llm.complete.side_effect = FileNotFoundError("no srt")
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)
            with self.assertRaises(RuntimeError):
                analyzer.run(["ep01", "ep02"])


# ===========================================================================
# Test: _cache_has_timecodes — cache staleness detection
# ===========================================================================

class TestCacheHasTimecodes(unittest.TestCase):

    def test_scene_with_timecodes_returns_true(self):
        from src.intelligence.rhythm_analyzer import _cache_has_timecodes
        cached = {"scenes": [_scene("ep01_sc_01")]}
        self.assertTrue(_cache_has_timecodes(cached))

    def test_scene_without_timecodes_returns_false(self):
        from src.intelligence.rhythm_analyzer import _cache_has_timecodes
        cached = {
            "scenes": [{
                "scene_id": "ep01_sc_01",
                "location": "office",
                "scene_actions": [],
            }]
        }
        self.assertFalse(_cache_has_timecodes(cached))

    def test_empty_scenes_returns_false(self):
        from src.intelligence.rhythm_analyzer import _cache_has_timecodes
        self.assertFalse(_cache_has_timecodes({"scenes": []}))

    def test_no_scenes_key_returns_false(self):
        from src.intelligence.rhythm_analyzer import _cache_has_timecodes
        self.assertFalse(_cache_has_timecodes({}))

    def test_stale_cache_triggers_reanalysis(self):
        """Cache without timecodes should NOT be returned; LLM should be called."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            stale = {
                "episode_id": "ep01",
                "scenes": [{"scene_id": "ep01_sc_01", "location": "office",
                             "scene_actions": [], "unresolved_debt": None,
                             "pivot_signals": [], "structured_dialogues": []}],
            }
            cache_path = tmp / "cache" / "drama_map" / "ep01_conflict.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(stale), encoding="utf-8")

            srt_dir = tmp / "output" / "cn"
            srt_dir.mkdir(parents=True)
            (srt_dir / "ep01.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8"
            )

            mock_llm = MagicMock()
            fresh = _ep_result("ep01")
            mock_llm.complete.return_value = json.dumps(fresh)
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)
            analyzer._analyze_one_episode("ep01")
            mock_llm.complete.assert_called_once()


# ===========================================================================
# Test: _enrich_clips_with_timecodes — timecode injection into marketing_clips
# ===========================================================================

class TestEnrichClipsWithTimecodes(unittest.TestCase):

    def _make_conflict_map(self):
        return {
            "ep01": {
                "episode_id": "01",
                "scenes": [
                    {**_scene("ep01_sc_01",
                               scene_start_time="00:01:10,000",
                               scene_end_time="00:03:45,500")},
                    {**_scene("ep01_sc_02",
                               scene_start_time="00:03:50,000",
                               scene_end_time="00:06:20,000")},
                ],
            },
            "ep04": {
                "episode_id": "04",
                "scenes": [
                    {**_scene("ep04_sc_02",
                               scene_start_time="00:08:00,000",
                               scene_end_time="00:10:30,000")},
                ],
            },
        }

    def test_timecodes_injected_from_scene_index(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            blueprint = {
                "marketing_clips": [
                    {"scene_id": "ep01_sc_01", "mix_strategy": "cold_open",
                     "action_start_focus": "start", "action_end_focus": "end",
                     "contrast_rationale": "contrast"},
                ]
            }
            analyzer._enrich_clips_with_timecodes(blueprint, self._make_conflict_map())
            clip = blueprint["marketing_clips"][0]
            self.assertEqual(clip["episode_id"],      "01")
            self.assertEqual(clip["clip_start_time"], "00:01:10,000")
            self.assertEqual(clip["clip_end_time"],   "00:03:45,500")

    def test_multiple_clips_from_different_episodes(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            blueprint = {
                "marketing_clips": [
                    {"scene_id": "ep01_sc_02", "mix_strategy": "single_scene",
                     "action_start_focus": "s", "action_end_focus": "e",
                     "contrast_rationale": "c"},
                    {"scene_id": "ep04_sc_02", "mix_strategy": "confrontation_cut",
                     "action_start_focus": "s", "action_end_focus": "e",
                     "contrast_rationale": "c"},
                ]
            }
            analyzer._enrich_clips_with_timecodes(blueprint, self._make_conflict_map())
            clips = blueprint["marketing_clips"]
            self.assertEqual(clips[0]["clip_start_time"], "00:03:50,000")
            self.assertEqual(clips[1]["clip_start_time"], "00:08:00,000")
            self.assertEqual(clips[1]["episode_id"],      "04")

    def test_unknown_scene_id_gets_empty_timecodes(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            blueprint = {
                "marketing_clips": [
                    {"scene_id": "ep99_sc_01", "mix_strategy": "montage",
                     "action_start_focus": "s", "action_end_focus": "e",
                     "contrast_rationale": "c"},
                ]
            }
            analyzer._enrich_clips_with_timecodes(blueprint, self._make_conflict_map())
            clip = blueprint["marketing_clips"][0]
            self.assertEqual(clip["clip_start_time"], "")
            self.assertEqual(clip["clip_end_time"],   "")
            self.assertEqual(clip["episode_id"],      "99")

    def test_no_clips_is_noop(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            blueprint = {"marketing_clips": []}
            analyzer._enrich_clips_with_timecodes(blueprint, self._make_conflict_map())
            self.assertEqual(blueprint["marketing_clips"], [])

    def test_missing_marketing_clips_key_is_noop(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            blueprint = {"debt_chain_narrative": "something"}
            analyzer._enrich_clips_with_timecodes(blueprint, {})
            self.assertNotIn("marketing_clips", blueprint)

    def test_start_end_scene_id_spans_two_scenes(self):
        """start_scene_id drives clip_start_time; end_scene_id drives clip_end_time."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            blueprint = {
                "marketing_clips": [
                    {"start_scene_id": "ep01_sc_01", "end_scene_id": "ep01_sc_02",
                     "mix_strategy": "confrontation_cut",
                     "action_start_focus": "start", "action_end_focus": "end",
                     "contrast_rationale": "contrast"},
                ]
            }
            analyzer._enrich_clips_with_timecodes(blueprint, self._make_conflict_map())
            clip = blueprint["marketing_clips"][0]
            self.assertEqual(clip["episode_id"],      "01")
            self.assertEqual(clip["clip_start_time"], "00:01:10,000")
            self.assertEqual(clip["clip_end_time"],   "00:06:20,000")

    def test_start_scene_id_without_end_defaults_to_same_scene(self):
        """When end_scene_id is absent, behaves like a single-scene clip."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            blueprint = {
                "marketing_clips": [
                    {"start_scene_id": "ep01_sc_02",
                     "mix_strategy": "single_scene",
                     "action_start_focus": "s", "action_end_focus": "e",
                     "contrast_rationale": "c"},
                ]
            }
            analyzer._enrich_clips_with_timecodes(blueprint, self._make_conflict_map())
            clip = blueprint["marketing_clips"][0]
            self.assertEqual(clip["clip_start_time"], "00:03:50,000")
            self.assertEqual(clip["clip_end_time"],   "00:06:20,000")

    def test_unknown_end_scene_id_gives_empty_end_time(self):
        """Known start_scene_id + unknown end_scene_id → valid start, empty end."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            analyzer = _make_analyzer(tmp)
            blueprint = {
                "marketing_clips": [
                    {"start_scene_id": "ep01_sc_01", "end_scene_id": "ep01_sc_99",
                     "mix_strategy": "cold_open",
                     "action_start_focus": "s", "action_end_focus": "e",
                     "contrast_rationale": "c"},
                ]
            }
            analyzer._enrich_clips_with_timecodes(blueprint, self._make_conflict_map())
            clip = blueprint["marketing_clips"][0]
            self.assertEqual(clip["clip_start_time"], "00:01:10,000")
            self.assertEqual(clip["clip_end_time"],   "")


# ===========================================================================
# Test: run() partial failure — orphaned method restored to TestRunIntegration
# ===========================================================================

class TestRunIntegrationPartial(unittest.TestCase):

    def _setup_srt(self, output_dir: Path, ep_ids: list[str]) -> None:
        srt_dir = output_dir / "cn"
        srt_dir.mkdir(parents=True, exist_ok=True)
        for ep_id in ep_ids:
            (srt_dir / f"{ep_id}.srt").write_text(
                f"1\n00:00:01,000 --> 00:00:02,000\nContent for {ep_id}\n",
                encoding="utf-8",
            )

    def test_run_partial_failure_uses_successful_episodes(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            # Only ep01 has an SRT (in cn/ subdir); ep02 will fail
            self._setup_srt(tmp / "output", ["ep01"])

            blueprint = {"debt_chain_narrative": "partial"}
            mock_llm = MagicMock()
            def _llm_side_effect(system, user, max_tokens=None,
                                 json_mode=False, module_name="default"):
                if module_name == "Drama_Analysis":
                    if "ep01" in user:
                        return json.dumps(_ep_result("ep01"))
                    raise FileNotFoundError("no srt for ep02")
                return json.dumps(blueprint)

            mock_llm.complete.side_effect = _llm_side_effect
            analyzer = _make_analyzer(tmp, llm_client=mock_llm)
            result = analyzer.run(["ep01", "ep02"])
            self.assertIn("ep01", result["episode_conflicts"])
            self.assertNotIn("ep02", result["episode_conflicts"])
            self.assertEqual(result["total_episodes_analysed"], 1)


# ===========================================================================
# Test: thread safety — parallel map calls don't corrupt ledger
# ===========================================================================

class TestThreadSafety(unittest.TestCase):

    def test_parallel_record_usage_accumulates_correctly(self):
        client = _make_llm_client()
        n_threads = 20
        calls_per_thread = 50

        def worker():
            for _ in range(calls_per_thread):
                client._record_usage("Drama_Analysis", 100, 50)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        data = client.get_ledger_data()
        expected_calls  = n_threads * calls_per_thread
        expected_input  = expected_calls * 100
        expected_output = expected_calls * 50
        self.assertEqual(data["total"]["calls"],         expected_calls)
        self.assertEqual(data["total"]["input_tokens"],  expected_input)
        self.assertEqual(data["total"]["output_tokens"], expected_output)
        self.assertEqual(data["by_module"]["Drama_Analysis"]["calls"], expected_calls)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
