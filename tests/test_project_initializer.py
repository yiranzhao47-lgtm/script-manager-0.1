"""
Unit tests for src.utils.project_initializer.ProjectInitializer

All heavy I/O (cv2, PaddleOCR, LLMClient) is mocked out so tests run without
GPU hardware or a live API key.  File operations use a temporary directory
(tmp_path) so the real project files are never touched.
"""
from __future__ import annotations

import json
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal config dict used across all tests
# ---------------------------------------------------------------------------
_CFG = {
    "pipeline": {"mode": "same_lang"},
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
            "max_attempts":   2,
            "wait_min_sec":   0,
            "wait_max_sec":   0,
            "no_retry_codes": [400, 401],
        },
    },
}

# Minimal settings.yaml content (same_lang + cross_lang sections)
_SETTINGS_YAML = textwrap.dedent("""\
    pipeline:
      mode: "same_lang"

    same_lang:
      asr:
        model: "large-v3"
      ocr:
        fps: 2
        roi: [0.55, 0.85]  # subtitle band
        language: "ch"

    cross_lang:
      asr:
        model: "large-v3"
      ocr:
        fps: 5
        roi: [0.80, 0.95]
        language: "en"
    """)

_INIT_SKILL_MD = "# init_agent_skill prompt"


def _make_project(tmp: Path) -> Path:
    """Create the minimal directory/file scaffold in *tmp*."""
    (tmp / "data" / "raw").mkdir(parents=True)
    (tmp / "data" / "cache" / "asr").mkdir(parents=True)
    (tmp / "data" / "cache" / "ocr").mkdir(parents=True)
    (tmp / "data" / "cache" / "aligned").mkdir(parents=True)
    (tmp / "data" / "cache" / "map_batches").mkdir(parents=True)
    (tmp / "data" / "cache" / "drama_map").mkdir(parents=True)
    (tmp / "data" / "meta").mkdir(parents=True)
    (tmp / "data" / "output").mkdir(parents=True)
    (tmp / "config" / "prompts").mkdir(parents=True)
    (tmp / "config" / "settings.yaml").write_text(_SETTINGS_YAML, encoding="utf-8")
    (tmp / "config" / "prompts" / "init_agent_skill.md").write_text(
        _INIT_SKILL_MD, encoding="utf-8"
    )
    return tmp


def _make_initializer(tmp: Path, cfg: dict | None = None):
    from src.utils.project_initializer import ProjectInitializer
    return ProjectInitializer(cfg or _CFG, _root=tmp)


# ===========================================================================
# Test: _scan_videos
# ===========================================================================

class TestScanVideos(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)

    def tearDown(self):
        self._td.cleanup()

    def test_empty_directory(self):
        pi = _make_initializer(self.tmp)
        self.assertEqual(pi._scan_videos(), [])

    def test_finds_mp4_files(self):
        raw = self.tmp / "data" / "raw"
        (raw / "01.mp4").touch()
        (raw / "02.mp4").touch()
        pi = _make_initializer(self.tmp)
        found = pi._scan_videos()
        self.assertEqual(len(found), 2)
        self.assertTrue(all(p.suffix == ".mp4" for p in found))

    def test_ignores_non_mp4(self):
        raw = self.tmp / "data" / "raw"
        (raw / "01.mp4").touch()
        (raw / "notes.txt").touch()
        pi = _make_initializer(self.tmp)
        found = pi._scan_videos()
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "01.mp4")

    def test_recursive_discovery(self):
        raw = self.tmp / "data" / "raw"
        sub = raw / "show_a"
        sub.mkdir()
        (sub / "ep01.mp4").touch()
        (sub / "ep02.mp4").touch()
        pi = _make_initializer(self.tmp)
        found = pi._scan_videos()
        self.assertEqual(len(found), 2)

    def test_missing_raw_dir(self):
        import shutil
        shutil.rmtree(self.tmp / "data" / "raw")
        pi = _make_initializer(self.tmp)
        self.assertEqual(pi._scan_videos(), [])


# ===========================================================================
# Test: _is_resume
# ===========================================================================

class TestIsResume(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)

    def tearDown(self):
        self._td.cleanup()

    def _write_checkpoint(self, video_list: list[str]) -> None:
        ckpt = {
            "version": 1,
            "episodes": {},
            "global": {"video_list": video_list},
        }
        ckpt_path = self.tmp / "data" / "cache" / "checkpoint.json"
        ckpt_path.write_text(json.dumps(ckpt), encoding="utf-8")

    def _make_videos(self, names: list[str]) -> list[Path]:
        raw = self.tmp / "data" / "raw"
        paths = []
        for name in names:
            p = raw / name
            p.touch()
            paths.append(p)
        return paths

    def test_no_checkpoint_is_not_resume(self):
        pi = _make_initializer(self.tmp)
        videos = self._make_videos(["01.mp4"])
        self.assertFalse(pi._is_resume(videos))

    def test_same_list_is_resume(self):
        self._write_checkpoint(["01.mp4", "02.mp4"])
        videos = self._make_videos(["01.mp4", "02.mp4"])
        pi = _make_initializer(self.tmp)
        self.assertTrue(pi._is_resume(videos))

    def test_different_count_is_not_resume(self):
        self._write_checkpoint(["01.mp4", "02.mp4", "03.mp4"])
        videos = self._make_videos(["01.mp4", "02.mp4"])
        pi = _make_initializer(self.tmp)
        self.assertFalse(pi._is_resume(videos))

    def test_different_names_is_not_resume(self):
        self._write_checkpoint(["ep01.mp4", "ep02.mp4"])
        videos = self._make_videos(["01.mp4", "02.mp4"])
        pi = _make_initializer(self.tmp)
        self.assertFalse(pi._is_resume(videos))

    def test_no_video_list_in_checkpoint_is_not_resume(self):
        ckpt = {"version": 1, "episodes": {}, "global": {}}
        (self.tmp / "data" / "cache" / "checkpoint.json").write_text(
            json.dumps(ckpt), encoding="utf-8"
        )
        videos = self._make_videos(["01.mp4"])
        pi = _make_initializer(self.tmp)
        self.assertFalse(pi._is_resume(videos))

    def test_corrupt_checkpoint_is_not_resume(self):
        (self.tmp / "data" / "cache" / "checkpoint.json").write_text(
            "not json {{{", encoding="utf-8"
        )
        videos = self._make_videos(["01.mp4"])
        pi = _make_initializer(self.tmp)
        self.assertFalse(pi._is_resume(videos))

    def test_order_independent(self):
        self._write_checkpoint(["02.mp4", "01.mp4"])
        videos = self._make_videos(["01.mp4", "02.mp4"])
        pi = _make_initializer(self.tmp)
        self.assertTrue(pi._is_resume(videos))

    def test_empty_both_is_resume(self):
        self._write_checkpoint([])
        pi = _make_initializer(self.tmp)
        self.assertTrue(pi._is_resume([]))


# ===========================================================================
# Test: _purge_stale_data
# ===========================================================================

class TestPurgeStaleData(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)

    def tearDown(self):
        self._td.cleanup()

    def test_cache_subdirs_cleared(self):
        for sub in ("asr", "ocr", "aligned", "map_batches", "drama_map"):
            (self.tmp / "data" / "cache" / sub / "stale.json").write_text("{}")
        pi = _make_initializer(self.tmp)
        pi._purge_stale_data()
        for sub in ("asr", "ocr", "aligned", "map_batches", "drama_map"):
            d = self.tmp / "data" / "cache" / sub
            self.assertTrue(d.is_dir(), f"cache/{sub}/ should still exist as empty dir")
            self.assertEqual(list(d.iterdir()), [], f"cache/{sub}/ should be empty")

    def test_meta_dir_cleared(self):
        (self.tmp / "data" / "meta" / "meta.json").write_text("{}")
        pi = _make_initializer(self.tmp)
        pi._purge_stale_data()
        self.assertTrue((self.tmp / "data" / "meta").is_dir())
        self.assertEqual(list((self.tmp / "data" / "meta").iterdir()), [])

    def test_checkpoint_deleted(self):
        ckpt = self.tmp / "data" / "cache" / "checkpoint.json"
        ckpt.write_text("{}")
        pi = _make_initializer(self.tmp)
        pi._purge_stale_data()
        self.assertFalse(ckpt.exists())

    def test_output_srts_deleted(self):
        out = self.tmp / "data" / "output"
        (out / "01.srt").write_text("srt content")
        (out / "02.srt").write_text("srt content")
        (out / "cost_report.json").write_text("{}")
        pi = _make_initializer(self.tmp)
        pi._purge_stale_data()
        self.assertEqual(list(out.glob("*.srt")), [])
        self.assertFalse((out / "cost_report.json").exists())

    def test_validation_report_deleted(self):
        report = self.tmp / "data" / "output" / "validation_report.json"
        report.write_text("{}")
        pi = _make_initializer(self.tmp)
        pi._purge_stale_data()
        self.assertFalse(report.exists())

    def test_purge_idempotent_on_missing_files(self):
        pi = _make_initializer(self.tmp)
        pi._purge_stale_data()  # nothing exists — should not raise


# ===========================================================================
# Test: _patch_settings_yaml
# ===========================================================================

class TestPatchSettingsYaml(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)

    def tearDown(self):
        self._td.cleanup()

    def _read_yaml(self) -> dict:
        import yaml
        with (self.tmp / "config" / "settings.yaml").open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_patches_same_lang_roi(self):
        pi = _make_initializer(self.tmp)
        pi._patch_settings_yaml([0.79, 0.93])
        cfg = self._read_yaml()
        self.assertEqual(cfg["same_lang"]["ocr"]["roi"], [0.79, 0.93])

    def test_patches_cross_lang_roi(self):
        cfg = {**_CFG, "pipeline": {"mode": "cross_lang"}}
        pi = _make_initializer(self.tmp, cfg)
        pi._patch_settings_yaml([0.81, 0.96])
        loaded = self._read_yaml()
        self.assertEqual(loaded["cross_lang"]["ocr"]["roi"], [0.81, 0.96])

    def test_preserves_other_keys(self):
        pi = _make_initializer(self.tmp)
        pi._patch_settings_yaml([0.79, 0.93])
        cfg = self._read_yaml()
        self.assertEqual(cfg["same_lang"]["ocr"]["fps"], 2)
        self.assertEqual(cfg["same_lang"]["ocr"]["language"], "ch")
        self.assertEqual(cfg["cross_lang"]["ocr"]["roi"], [0.80, 0.95])

    def test_preserves_comment_on_roi_line(self):
        raw_before = (self.tmp / "config" / "settings.yaml").read_text(encoding="utf-8")
        self.assertIn("# subtitle band", raw_before)
        pi = _make_initializer(self.tmp)
        pi._patch_settings_yaml([0.79, 0.93])
        raw_after = (self.tmp / "config" / "settings.yaml").read_text(encoding="utf-8")
        self.assertIn("# subtitle band", raw_after)

    def test_no_crash_on_missing_section(self):
        yaml_no_mode = "pipeline:\n  mode: same_lang\n"
        (self.tmp / "config" / "settings.yaml").write_text(yaml_no_mode, encoding="utf-8")
        pi = _make_initializer(self.tmp)
        pi._patch_settings_yaml([0.79, 0.93])  # should warn but not raise


# ===========================================================================
# Test: _write_checkpoint
# ===========================================================================

class TestWriteCheckpoint(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)

    def tearDown(self):
        self._td.cleanup()

    def test_writes_valid_json(self):
        raw = self.tmp / "data" / "raw"
        (raw / "01.mp4").touch()
        (raw / "02.mp4").touch()
        pi = _make_initializer(self.tmp)
        videos = [raw / "01.mp4", raw / "02.mp4"]
        pi._write_checkpoint(videos)
        ckpt = json.loads((self.tmp / "data" / "cache" / "checkpoint.json").read_text())
        self.assertEqual(ckpt["version"], 1)
        self.assertEqual(ckpt["episodes"], {})
        self.assertIn("video_list", ckpt["global"])

    def test_video_list_relative_posix_paths(self):
        raw = self.tmp / "data" / "raw"
        sub = raw / "show"
        sub.mkdir()
        (sub / "ep01.mp4").touch()
        pi = _make_initializer(self.tmp)
        pi._write_checkpoint([sub / "ep01.mp4"])
        ckpt = json.loads((self.tmp / "data" / "cache" / "checkpoint.json").read_text())
        self.assertIn("show/ep01.mp4", ckpt["global"]["video_list"])

    def test_empty_video_list(self):
        pi = _make_initializer(self.tmp)
        pi._write_checkpoint([])
        ckpt = json.loads((self.tmp / "data" / "cache" / "checkpoint.json").read_text())
        self.assertEqual(ckpt["global"]["video_list"], [])


# ===========================================================================
# Test: auto_detect_and_heal — resume scenario (no side effects)
# ===========================================================================

class TestAutoDetectResume(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)
        # Plant a video and a matching checkpoint
        raw = self.tmp / "data" / "raw"
        (raw / "01.mp4").touch()
        ckpt = {
            "version": 1,
            "episodes": {"01": "complete"},
            "global": {"video_list": ["01.mp4"]},
        }
        (self.tmp / "data" / "cache" / "checkpoint.json").write_text(
            json.dumps(ckpt), encoding="utf-8"
        )

    def tearDown(self):
        self._td.cleanup()

    def test_resume_does_not_purge(self):
        sentinel = self.tmp / "data" / "cache" / "asr" / "01_asr.json"
        sentinel.write_text("{}")
        pi = _make_initializer(self.tmp)
        pi.auto_detect_and_heal()
        # Cache file must survive — resume must not purge
        self.assertTrue(sentinel.exists())

    def test_resume_does_not_touch_settings_yaml(self):
        before = (self.tmp / "config" / "settings.yaml").read_text()
        pi = _make_initializer(self.tmp)
        pi.auto_detect_and_heal()
        after = (self.tmp / "config" / "settings.yaml").read_text()
        self.assertEqual(before, after)

    def test_resume_checkpoint_unchanged(self):
        ckpt_before = (self.tmp / "data" / "cache" / "checkpoint.json").read_text()
        pi = _make_initializer(self.tmp)
        pi.auto_detect_and_heal()
        ckpt_after = (self.tmp / "data" / "cache" / "checkpoint.json").read_text()
        self.assertEqual(ckpt_before, ckpt_after)


# ===========================================================================
# Test: auto_detect_and_heal — new-show scenario (full workflow, mocked)
# ===========================================================================

class TestAutoDetectNewShow(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)
        # Place a stale cache artifact and an old checkpoint for a DIFFERENT show
        (self.tmp / "data" / "cache" / "asr" / "old.json").write_text("{}")
        old_ckpt = {
            "version": 1,
            "episodes": {},
            "global": {"video_list": ["old_ep01.mp4"]},
        }
        (self.tmp / "data" / "cache" / "checkpoint.json").write_text(
            json.dumps(old_ckpt), encoding="utf-8"
        )
        # New show's video file
        (self.tmp / "data" / "raw" / "ep01.mp4").write_bytes(b"")

    def tearDown(self):
        self._td.cleanup()

    def _run_with_mocks(self, roi_response: list[float] = None):
        roi_response = roi_response or [0.79, 0.93]
        llm_json = json.dumps({
            "detected_scenario": "new_show_detected",
            "reason": "Mock test — new video detected.",
            "recommended_roi": roi_response,
        })

        import numpy as np

        fake_frame = np.zeros((1080, 1920, 3), dtype="uint8")

        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        fake_cap.get.return_value = 25.0
        fake_cap.read.return_value = (True, fake_frame)

        # Correct PaddleOCR result structure:
        #   raw[0] = list of detections
        #   raw[0][i] = [bbox_4pts, (text, confidence)]
        fake_ocr_result = [
            [  # raw[0]: one detection list
                [  # raw[0][0]: single detection = [bbox, text_info]
                    [[0, 900], [200, 900], [200, 940], [0, 940]],  # bbox (4 corners)
                    ("台词文本", 0.95),                             # (text, confidence)
                ]
            ]
        ]
        fake_engine = MagicMock()
        fake_engine.ocr.return_value = fake_ocr_result

        fake_llm_instance = MagicMock()
        fake_llm_instance.complete.return_value = llm_json

        with (
            patch("cv2.VideoCapture", return_value=fake_cap),
            patch("paddleocr.PaddleOCR", return_value=fake_engine),
            patch("src.utils.llm_client.LLMClient", return_value=fake_llm_instance),
        ):
            pi = _make_initializer(self.tmp)
            pi.auto_detect_and_heal()

    def test_stale_cache_purged(self):
        self._run_with_mocks()
        self.assertFalse((self.tmp / "data" / "cache" / "asr" / "old.json").exists())

    def test_fresh_checkpoint_written(self):
        self._run_with_mocks()
        ckpt = json.loads(
            (self.tmp / "data" / "cache" / "checkpoint.json").read_text()
        )
        self.assertIn("ep01.mp4", ckpt["global"]["video_list"])

    def test_settings_yaml_roi_patched(self):
        self._run_with_mocks(roi_response=[0.79, 0.93])
        import yaml
        cfg = yaml.safe_load(
            (self.tmp / "config" / "settings.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(cfg["same_lang"]["ocr"]["roi"], [0.79, 0.93])

    def test_roi_clamped_to_unit_interval(self):
        """LLM returning out-of-range values should be silently clamped."""
        self._run_with_mocks(roi_response=[-0.5, 1.5])
        import yaml
        cfg = yaml.safe_load(
            (self.tmp / "config" / "settings.yaml").read_text(encoding="utf-8")
        )
        roi = cfg["same_lang"]["ocr"]["roi"]
        self.assertGreaterEqual(roi[0], 0.0)
        self.assertLessEqual(roi[1], 1.0)


# ===========================================================================
# Test: LLMClient.complete json_mode parameter
# ===========================================================================

class TestLLMClientJsonMode(unittest.TestCase):

    def _make_client(self):
        from src.utils.llm_client import LLMClient
        cfg = {
            "execution": {
                "llm": {
                    "model": "test-model",
                    "base_url": None,
                    "api_key_env": "OPENAI_API_KEY",
                    "max_tokens": 100,
                },
                "retry": {
                    "max_attempts": 1,
                    "wait_min_sec": 0,
                    "wait_max_sec": 0,
                    "no_retry_codes": [400, 401],
                },
            }
        }
        return LLMClient(cfg)

    def test_json_mode_false_no_response_format(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"ok": true}'
        mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_resp
        client._openai_client = mock_openai

        client.complete(system="sys", user="usr", json_mode=False)

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        self.assertNotIn("response_format", call_kwargs)

    def test_json_mode_true_sends_response_format(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"ok": true}'
        mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_resp
        client._openai_client = mock_openai

        client.complete(system="sys", user="usr", json_mode=True)

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        self.assertIn("response_format", call_kwargs)
        self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
