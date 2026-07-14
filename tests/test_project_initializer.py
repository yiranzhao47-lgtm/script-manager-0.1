"""
Unit tests for src.utils.project_initializer.ProjectInitializer

Heavy I/O (subprocess/ffprobe/ffmpeg, PaddleOCR) is mocked so tests run
without GPU hardware.  File operations use a temporary directory (tmp_path)
so the real project files are never touched.

New behaviour tested here:
  - ROI calibration (_run_calibration) runs on ALL scenarios, not just new-show.
  - Statistical detection replaces the old single-frame + LLM approach.
  - settings.yaml is always patched with the measured ROI.
  - Verification (spot-check) must confirm at least one subtitle detected.
"""
from __future__ import annotations

import json
import textwrap
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np

# ---------------------------------------------------------------------------
# Minimal config dict used across all tests
# ---------------------------------------------------------------------------
_CFG = {
    "pipeline": {"mode": "same_lang"},
    "same_lang": {"ocr": {"language": "ch", "use_gpu": False, "roi": [0.78, 0.94]}},
    "paths": {
        "raw_video_dir": "data/raw",
        "cache_dir":     "data/cache",
        "meta_dir":      "data/meta",
        "output_dir":    "data/output",
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


def _make_project(tmp: Path) -> Path:
    """Create the minimal directory/file scaffold in *tmp*."""
    (tmp / "data" / "raw").mkdir(parents=True)
    (tmp / "data" / "cache" / "asr").mkdir(parents=True)
    (tmp / "data" / "cache" / "ocr").mkdir(parents=True)
    (tmp / "data" / "cache" / "aligned").mkdir(parents=True)
    (tmp / "data" / "cache" / "map_batches").mkdir(parents=True)
    (tmp / "data" / "cache" / "drama_map").mkdir(parents=True)
    (tmp / "data" / "meta").mkdir(parents=True)
    (tmp / "data" / "output" / "cn").mkdir(parents=True)
    (tmp / "config" / "prompts").mkdir(parents=True)
    (tmp / "config" / "settings.yaml").write_text(_SETTINGS_YAML, encoding="utf-8")
    return tmp


def _make_initializer(tmp: Path, cfg: dict | None = None):
    from src.utils.project_initializer import ProjectInitializer
    return ProjectInitializer(cfg or _CFG, _root=tmp)


# ---------------------------------------------------------------------------
# Shared mock helpers for calibration
# ---------------------------------------------------------------------------

def _fake_probe_stdout(frame_w: int = 720, frame_h: int = 1280) -> str:
    return json.dumps({
        "streams": [{"codec_type": "video", "width": frame_w, "height": frame_h}]
    })


def _fake_frame_bytes(frame_w: int = 720, frame_h: int = 1280) -> bytes:
    """Return a black raw-BGR frame of the correct byte count."""
    return bytes(frame_w * frame_h * 3)


def _make_ocr_detection(
    y_top: int = 860, y_bot: int = 900,
    x_left: int = 10, x_right: int = 500,
    text: str = "台词",
    conf: float = 0.95,
) -> list:
    """One PaddleOCR detection: [bbox_4pts, (text, confidence)]."""
    return [
        [[x_left, y_top], [x_right, y_top], [x_right, y_bot], [x_left, y_bot]],
        (text, conf),
    ]


def _make_engine_mock(detections: list | None = None) -> MagicMock:
    """PaddleOCR engine mock that returns *detections* on every .ocr() call."""
    if detections is None:
        detections = [_make_ocr_detection()]
    engine = MagicMock()
    engine.ocr.return_value = [detections]   # [[det1, det2, ...]]
    return engine


@contextmanager
def _mock_calibration_io(
    frame_w: int = 720,
    frame_h: int = 1280,
    engine: MagicMock | None = None,
):
    """
    Context manager that patches the I/O boundary methods of ProjectInitializer
    and the PaddleOCR constructor — without touching global subprocess.run
    (which would break paddle's own platform detection at import time).

    Patches:
      - ProjectInitializer._probe_dimensions  → (frame_w, frame_h)
      - ProjectInitializer._extract_frame     → black numpy frame
      - paddleocr.PaddleOCR                  → fake engine with subtitle detections

    Yields the fake PaddleOCR engine so callers can inspect .ocr() calls.
    """
    if engine is None:
        engine = _make_engine_mock()

    fake_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    with (
        patch(
            "src.utils.project_initializer.ProjectInitializer._probe_dimensions",
            return_value=(frame_w, frame_h),
        ),
        patch(
            "src.utils.project_initializer.ProjectInitializer._extract_frame",
            return_value=fake_frame,
        ),
        patch("paddleocr.PaddleOCR", return_value=engine),
    ):
        yield engine


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
        cn = out / "cn"
        en = out / "en"
        cn.mkdir(parents=True, exist_ok=True)
        en.mkdir(parents=True, exist_ok=True)
        (cn / "01.srt").write_text("srt content")
        (en / "01.srt").write_text("srt content")
        (out / "cost_report.json").write_text("{}")
        pi = _make_initializer(self.tmp)
        pi._purge_stale_data()
        self.assertEqual(list(out.glob("*/*.srt")), [])
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
        """Existing inline comment must survive the patch."""
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
# Test: auto_detect_and_heal — resume scenario
# ===========================================================================

class TestAutoDetectResume(unittest.TestCase):
    """
    In the new design, resume STILL runs ROI calibration.
    The test verifies:
      - cache is NOT purged
      - checkpoint is NOT modified
      - settings.yaml IS updated (calibration ran)
      - PaddleOCR is called (calibration ran)
    """

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)
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

    def test_resume_does_not_purge_cache(self):
        """Existing cache files must survive a resume run."""
        sentinel = self.tmp / "data" / "cache" / "asr" / "01_asr.json"
        sentinel.write_text("{}")
        with _mock_calibration_io():
            pi = _make_initializer(self.tmp)
            pi.auto_detect_and_heal()
        self.assertTrue(sentinel.exists())

    def test_resume_does_not_modify_checkpoint(self):
        """Checkpoint must be left unchanged on resume."""
        ckpt_before = (self.tmp / "data" / "cache" / "checkpoint.json").read_text()
        with _mock_calibration_io():
            pi = _make_initializer(self.tmp)
            pi.auto_detect_and_heal()
        ckpt_after = (self.tmp / "data" / "cache" / "checkpoint.json").read_text()
        self.assertEqual(ckpt_before, ckpt_after)

    def test_resume_runs_calibration_and_patches_yaml(self):
        """Even on resume, ROI calibration must update settings.yaml."""
        with _mock_calibration_io() as engine:
            pi = _make_initializer(self.tmp)
            pi.auto_detect_and_heal()
        # PaddleOCR must have been called
        self.assertTrue(engine.ocr.called)
        # settings.yaml must be updated (not the same as before)
        import yaml
        cfg = yaml.safe_load(
            (self.tmp / "config" / "settings.yaml").read_text(encoding="utf-8")
        )
        # ROI must be within valid range and calibrated (not the original [0.55, 0.85])
        roi = cfg["same_lang"]["ocr"]["roi"]
        self.assertIsInstance(roi, list)
        self.assertEqual(len(roi), 2)
        self.assertGreaterEqual(roi[0], 0.0)
        self.assertLessEqual(roi[1], 1.0)


# ===========================================================================
# Test: auto_detect_and_heal — new-show scenario
# ===========================================================================

class TestAutoDetectNewShow(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)
        (self.tmp / "data" / "cache" / "asr" / "old.json").write_text("{}")
        old_ckpt = {
            "version": 1,
            "episodes": {},
            "global": {"video_list": ["old_ep01.mp4"]},
        }
        (self.tmp / "data" / "cache" / "checkpoint.json").write_text(
            json.dumps(old_ckpt), encoding="utf-8"
        )
        (self.tmp / "data" / "raw" / "ep01.mp4").write_bytes(b"")

    def tearDown(self):
        self._td.cleanup()

    def test_stale_cache_purged(self):
        with _mock_calibration_io():
            pi = _make_initializer(self.tmp)
            pi.auto_detect_and_heal()
        self.assertFalse((self.tmp / "data" / "cache" / "asr" / "old.json").exists())

    def test_fresh_checkpoint_written(self):
        with _mock_calibration_io():
            pi = _make_initializer(self.tmp)
            pi.auto_detect_and_heal()
        ckpt = json.loads(
            (self.tmp / "data" / "cache" / "checkpoint.json").read_text()
        )
        self.assertIn("ep01.mp4", ckpt["global"]["video_list"])

    def test_settings_yaml_roi_patched(self):
        """settings.yaml must be updated with the statistically measured ROI."""
        with _mock_calibration_io():
            pi = _make_initializer(self.tmp)
            pi.auto_detect_and_heal()
        import yaml
        cfg = yaml.safe_load(
            (self.tmp / "config" / "settings.yaml").read_text(encoding="utf-8")
        )
        roi = cfg["same_lang"]["ocr"]["roi"]
        self.assertIsInstance(roi, list)
        self.assertEqual(len(roi), 2)
        self.assertGreaterEqual(roi[0], 0.0)
        self.assertLessEqual(roi[1], 1.0)

    def test_roi_clamped_to_unit_interval(self):
        """Detected y-values near 0 or 1 must be clamped after margin is added."""
        # Report boxes at extreme y positions to trigger clamping
        extreme_det = _make_ocr_detection(y_top=5, y_bot=1275)  # near edges of 1280px
        engine = _make_engine_mock([extreme_det])
        with _mock_calibration_io(engine=engine):
            pi = _make_initializer(self.tmp)
            pi.auto_detect_and_heal()
        import yaml
        cfg = yaml.safe_load(
            (self.tmp / "config" / "settings.yaml").read_text(encoding="utf-8")
        )
        roi = cfg["same_lang"]["ocr"]["roi"]
        self.assertGreaterEqual(roi[0], 0.0)
        self.assertLessEqual(roi[1], 1.0)

    def test_calibration_aborts_when_no_subtitles(self):
        """If OCR finds no wide subtitle boxes, RuntimeError must be raised."""
        # Detection is narrow (watermark-like) — below MIN_WIDTH_RATIO threshold
        narrow_det = _make_ocr_detection(x_left=0, x_right=50)  # 50/720 = 0.069 < 0.25
        engine = _make_engine_mock([narrow_det])
        with _mock_calibration_io(engine=engine):
            pi = _make_initializer(self.tmp)
            with self.assertRaises(RuntimeError):
                pi.auto_detect_and_heal()

    def test_in_memory_cfg_updated(self):
        """cfg dict must be updated in-place so OCRRunner gets the correct ROI."""
        with _mock_calibration_io():
            pi = _make_initializer(self.tmp)
            pi.auto_detect_and_heal()
        roi = pi._cfg["same_lang"]["ocr"]["roi"]
        self.assertIsInstance(roi, list)
        self.assertEqual(len(roi), 2)


# ===========================================================================
# Test: _calibrate_roi_statistical — unit tests
# ===========================================================================

class TestCalibrateROIStatistical(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)
        (self.tmp / "data" / "raw" / "ep01.mp4").touch()

    def tearDown(self):
        self._td.cleanup()

    def _pi(self, cfg=None):
        return _make_initializer(self.tmp, cfg)

    def test_detects_subtitle_band(self):
        """Boxes at y≈0.67 should produce ROI around that band."""
        # In a 1280px frame, y_top=860, y_bot=900 → ratios 0.672, 0.703
        det = _make_ocr_detection(y_top=860, y_bot=900, x_right=400)
        engine = _make_engine_mock([det])
        with _mock_calibration_io(engine=engine):
            pi = self._pi()
            roi, _ = pi._calibrate_roi_statistical(pi._scan_videos())
        # roi_start ≈ 0.672 - 0.04 = 0.632; roi_end ≈ 0.703 + 0.04 = 0.743
        self.assertLess(roi[0], 0.672)
        self.assertGreater(roi[1], 0.703)

    def test_raises_when_too_few_boxes(self):
        """Fewer than _MIN_BOXES_REQUIRED wide boxes → RuntimeError."""
        # Use a narrow box (width 10px out of 720 → 0.014 < 0.25)
        narrow_det = _make_ocr_detection(x_left=0, x_right=10)
        engine = _make_engine_mock([narrow_det])
        with _mock_calibration_io(engine=engine):
            pi = self._pi()
            with self.assertRaises(RuntimeError) as ctx:
                pi._calibrate_roi_statistical(pi._scan_videos())
        self.assertIn("subtitle box", str(ctx.exception).lower())

    def test_ignores_upper_half_boxes(self):
        """Boxes in the top half (y_center < 0.45) should not count as subtitles."""
        upper_det = _make_ocr_detection(y_top=100, y_bot=200, x_right=400)  # centre ≈ 0.117
        engine = _make_engine_mock([upper_det])
        with _mock_calibration_io(engine=engine):
            pi = self._pi()
            with self.assertRaises(RuntimeError):
                pi._calibrate_roi_statistical(pi._scan_videos())

    def test_ignores_low_confidence_boxes(self):
        """Detections below _MIN_CONFIDENCE threshold must be discarded."""
        low_conf_det = _make_ocr_detection(conf=0.30)
        engine = _make_engine_mock([low_conf_det])
        with _mock_calibration_io(engine=engine):
            pi = self._pi()
            with self.assertRaises(RuntimeError):
                pi._calibrate_roi_statistical(pi._scan_videos())


# ===========================================================================
# Test: _verify_roi — unit tests
# ===========================================================================

class TestVerifyROI(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _make_project(self.tmp)
        (self.tmp / "data" / "raw" / "ep01.mp4").touch()

    def tearDown(self):
        self._td.cleanup()

    def test_passes_when_ocr_detects_text(self):
        """Verification must succeed when OCR finds at least one text block."""
        engine = _make_engine_mock()
        with _mock_calibration_io(engine=engine):
            pi = _make_initializer(self.tmp)
            videos = pi._scan_videos()
            pi._verify_roi(videos, [0.60, 0.75], engine)  # should not raise

    def test_raises_when_ocr_finds_nothing(self):
        """Verification must raise RuntimeError if no text detected in any frame."""
        engine = MagicMock()
        engine.ocr.return_value = [None]   # PaddleOCR returns empty
        with _mock_calibration_io(engine=engine):
            pi = _make_initializer(self.tmp)
            videos = pi._scan_videos()
            with self.assertRaises(RuntimeError) as ctx:
                pi._verify_roi(videos, [0.60, 0.75], engine)
        self.assertIn("FAILED", str(ctx.exception))


# ===========================================================================
# Test: LLMClient.complete json_mode parameter (unchanged)
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
