"""
ProjectInitializer — auto-detects show changes and self-heals the pipeline.

Called once at the top of every pipeline run, before any stage executes.

ROI Calibration (runs on EVERY pipeline start, all scenarios):
    1. Sample up to 5 episodes × 6 timestamps with full-frame PaddleOCR.
    2. Collect all wide, lower-half text boxes as subtitle candidates.
    3. Compute ROI from 5th/95th percentile of detected y-bands + margin.
    4. Patch config/settings.yaml and update in-memory cfg.
    5. Spot-check: run OCR on a frame cropped to the new ROI — abort if empty.

Scenario actions (in addition to ROI calibration):
    Resume scenario     → only ROI calibration; pipeline continues.
    Additive scenario   → ROI calibration + checkpoint update.
    New-show scenario   → purge cache + ROI calibration + fresh checkpoint.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Calibration constants
_SAMPLE_EPISODES    = 5          # max episodes to sample
_SAMPLE_TIMESTAMPS  = [10, 30, 60, 120, 180, 300]   # seconds into video
_MIN_WIDTH_RATIO    = 0.25       # subtitle boxes span ≥25% of frame width
_MIN_Y_CENTER       = 0.45       # subtitle centre must be in lower 55%
_MIN_CONFIDENCE     = 0.50       # PaddleOCR confidence floor
_MIN_BOXES_REQUIRED = 5          # abort if fewer than this many boxes detected
_ROI_MARGIN         = 0.04       # padding added around the detected y-band
_VERIFY_TIMESTAMPS  = [30, 60, 120, 180]   # timestamps tried during spot-check


class ProjectInitializer:
    """
    Stateless guard that runs before the main pipeline and handles
    show-change detection, cache purge, and mandatory ROI calibration.

    Parameters
    ----------
    cfg:
        Parsed ``config/settings.yaml`` dict (same object passed to
        ShortDramaPipeline).  This object is mutated in-place when the
        calibrated ROI is written back.
    _root:
        Override the project root directory (used in unit tests).
    """

    def __init__(self, cfg: dict, _root: Optional[Path] = None) -> None:
        root = _root or _PROJECT_ROOT
        self._cfg = cfg
        self._raw_dir       = root / cfg["paths"]["raw_video_dir"]
        self._cache_dir     = root / cfg["paths"]["cache_dir"]
        self._meta_dir      = root / cfg["paths"]["meta_dir"]
        self._output_dir    = root / cfg["paths"]["output_dir"]
        self._ckpt_path     = self._cache_dir / "checkpoint.json"
        self._settings_path = root / "config" / "settings.yaml"

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def auto_detect_and_heal(self) -> None:
        """
        Detect the current scenario, handle cache/checkpoint accordingly,
        and — in ALL scenarios — run mandatory subtitle ROI calibration.
        """
        current_videos = self._scan_videos()

        if self._is_resume(current_videos):
            logger.info(
                "[Init] Video set unchanged (%d file(s)) — resuming. "
                "Running mandatory ROI calibration.",
                len(current_videos),
            )
            self._run_calibration(current_videos)
            return

        saved_videos = self._get_saved_video_list()
        if saved_videos is not None and self._is_additive(current_videos, saved_videos):
            new_files = sorted(
                set(p.name for p in current_videos) - set(saved_videos)
            )
            logger.info(
                "[Init] %d new episode(s) added — updating checkpoint. "
                "Running mandatory ROI calibration.",
                len(new_files),
            )
            self._run_calibration(current_videos)
            self._write_checkpoint(current_videos)
            return

        logger.info(
            "[Init] New show detected (%d video(s)) — purging cache and "
            "running mandatory ROI calibration.",
            len(current_videos),
        )
        self._purge_stale_data()
        self._run_calibration(current_videos)
        self._write_checkpoint(current_videos)

    # ------------------------------------------------------------------ #
    #  ROI calibration — runs on every pipeline start                     #
    # ------------------------------------------------------------------ #

    def _run_calibration(self, videos: list[Path]) -> None:
        """
        Top-level wrapper: calibrate ROI, update config + YAML, verify.
        Raises RuntimeError on failure (empty-run guard).
        """
        if not videos:
            raise RuntimeError(
                "[ROI Calibration] No .mp4 files found under "
                f"{self._raw_dir} — cannot calibrate ROI."
            )
        roi, engine = self._calibrate_roi_statistical(videos)
        self._apply_roi(roi)
        self._verify_roi(videos, roi, engine)

    def _calibrate_roi_statistical(
        self, videos: list[Path]
    ) -> tuple[list[float], Any]:
        """
        Sample multiple episodes × timestamps with full-frame PaddleOCR.
        Collect wide, lower-half text boxes as subtitle candidates and
        derive the ROI from their y-band percentiles.

        Returns (roi, ocr_engine) so the engine can be reused for verification.
        Raises RuntimeError if fewer than _MIN_BOXES_REQUIRED boxes found.
        """
        mode    = self._cfg.get("pipeline", {}).get("mode", "same_lang")
        lang    = self._cfg.get(mode, {}).get("ocr", {}).get("language", "ch")
        use_gpu = self._cfg.get(mode, {}).get("ocr", {}).get("use_gpu", True)

        from paddleocr import PaddleOCR
        engine = PaddleOCR(
            use_angle_cls=False, lang=lang, use_gpu=use_gpu, show_log=False
        )

        sample_videos = videos[:_SAMPLE_EPISODES]
        logger.info(
            "[ROI Calibration] Scanning %d episode(s) × %d timestamp(s) …",
            len(sample_videos), len(_SAMPLE_TIMESTAMPS),
        )

        # Probe frame dimensions from the first video
        frame_w, frame_h = self._probe_dimensions(videos[0])

        y_tops: list[float] = []
        y_bots: list[float] = []

        for video in sample_videos:
            for ts in _SAMPLE_TIMESTAMPS:
                frame = self._extract_frame(video, ts, frame_w, frame_h)
                if frame is None:
                    continue
                result = engine.ocr(frame, cls=False)
                if not result or not result[0]:
                    continue
                for line in result[0]:
                    if not line or len(line) < 2:
                        continue
                    box, txt_info = line[0], line[1]
                    if not isinstance(txt_info, (list, tuple)) or len(txt_info) < 2:
                        continue
                    conf = float(txt_info[1])
                    if conf < _MIN_CONFIDENCE:
                        continue
                    xs = [int(p[0]) for p in box]
                    ys = [int(p[1]) for p in box]
                    w_ratio  = (max(xs) - min(xs)) / frame_w
                    y_center = (min(ys) + max(ys)) / 2.0 / frame_h
                    if w_ratio >= _MIN_WIDTH_RATIO and y_center >= _MIN_Y_CENTER:
                        y_tops.append(min(ys) / frame_h)
                        y_bots.append(max(ys) / frame_h)

        n = len(y_tops)
        logger.info("[ROI Calibration] %d subtitle box(es) detected.", n)

        if n < _MIN_BOXES_REQUIRED:
            raise RuntimeError(
                f"[ROI Calibration] Only {n} subtitle box(es) detected across "
                f"{len(sample_videos)} episode(s) (minimum required: "
                f"{_MIN_BOXES_REQUIRED}).\n"
                f"Possible causes:\n"
                f"  • Videos have no burned-in subtitles\n"
                f"  • OCR language mismatch (current: '{lang}')\n"
                f"  • Videos are very short (sampled timestamps beyond duration)\n"
                f"Resolve these before re-running the pipeline."
            )

        # Percentile bounds + safety margin
        sorted_tops = sorted(y_tops)
        sorted_bots = sorted(y_bots)
        p5_top  = sorted_tops[max(0, int(0.05 * n))]
        p95_bot = sorted_bots[min(n - 1, int(0.95 * n))]

        roi_start = round(max(0.0, p5_top  - _ROI_MARGIN), 3)
        roi_end   = round(min(1.0, p95_bot + _ROI_MARGIN), 3)
        roi = [roi_start, roi_end]

        logger.info(
            "[ROI Calibration] Computed ROI: %s  "
            "(y_top p5=%.3f, y_bot p95=%.3f, boxes=%d)",
            roi, p5_top, p95_bot, n,
        )
        return roi, engine

    def _apply_roi(self, roi: list[float]) -> None:
        """Write ROI into in-memory cfg and patch settings.yaml."""
        mode = self._cfg.get("pipeline", {}).get("mode", "same_lang")
        self._cfg.setdefault(mode, {}).setdefault("ocr", {})["roi"] = roi
        self._patch_settings_yaml(roi)

    def _verify_roi(
        self, videos: list[Path], roi: list[float], engine: Any
    ) -> None:
        """
        Spot-check: extract several frames cropped to the new ROI and run OCR.
        Raises RuntimeError if NO frame returns at least one text detection.
        This is the empty-run guard — the pipeline refuses to proceed without
        confirmed subtitle coverage.
        """
        frame_w, frame_h = self._probe_dimensions(videos[0])
        y0 = int(frame_h * roi[0])
        y1 = int(frame_h * roi[1])

        logger.info(
            "[ROI Verification] Spot-checking ROI %s (y=[%d:%d] of %d px) …",
            roi, y0, y1, frame_h,
        )

        for ts in _VERIFY_TIMESTAMPS:
            frame = self._extract_frame(videos[0], ts, frame_w, frame_h)
            if frame is None:
                continue
            roi_crop = frame[y0:y1, :]
            result = engine.ocr(roi_crop, cls=False)
            if result and result[0]:
                txt_sample = ""
                for line in result[0][:2]:
                    if line and len(line) >= 2 and isinstance(line[1], (list, tuple)):
                        txt_sample += line[1][0][:12] + " "
                logger.info(
                    "[ROI Verification] OK — subtitles detected at t=%ds: %r",
                    ts, txt_sample.strip(),
                )
                return

        raise RuntimeError(
            f"[ROI Verification] FAILED — no subtitles detected in ROI {roi} "
            f"across timestamps {_VERIFY_TIMESTAMPS} in '{videos[0].name}'.\n"
            f"The calibrated ROI is probably wrong. "
            f"Inspect the video manually and set same_lang.ocr.roi in "
            f"config/settings.yaml before re-running."
        )

    # ------------------------------------------------------------------ #
    #  Frame extraction helpers                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _probe_dimensions(video: Path) -> tuple[int, int]:
        """Return (width, height) of the video stream."""
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(video)],
            capture_output=True, text=True,
        )
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                return int(s["width"]), int(s["height"])
        raise ValueError(f"No video stream found in {video}")

    @staticmethod
    def _extract_frame(
        video: Path, ts: float, frame_w: int, frame_h: int
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame at *ts* seconds via ffmpeg pipe.
        Returns None if the video is shorter than *ts* or extraction fails.
        """
        r = subprocess.run(
            ["ffmpeg", "-ss", str(ts), "-i", str(video),
             "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
            capture_output=True,
        )
        expected = frame_w * frame_h * 3
        if len(r.stdout) < expected:
            return None
        return np.frombuffer(r.stdout, dtype=np.uint8).reshape((frame_h, frame_w, 3))

    # ------------------------------------------------------------------ #
    #  Scenario detection                                                  #
    # ------------------------------------------------------------------ #

    def _scan_videos(self) -> list[Path]:
        if not self._raw_dir.exists():
            return []
        return sorted(self._raw_dir.rglob("*.mp4"), key=lambda p: str(p))

    def _get_saved_video_list(self) -> Optional[list]:
        if not self._ckpt_path.exists():
            return None
        try:
            with self._ckpt_path.open(encoding="utf-8") as fh:
                ckpt = json.load(fh)
            return ckpt.get("global", {}).get("video_list")
        except (json.JSONDecodeError, OSError):
            return None

    def _is_additive(self, current_videos: list[Path], saved: list) -> bool:
        current_rel = {p.relative_to(self._raw_dir).as_posix() for p in current_videos}
        saved_set = set(saved)
        return saved_set.issubset(current_rel) and len(current_rel) > len(saved_set)

    def _is_resume(self, current_videos: list[Path]) -> bool:
        if not self._ckpt_path.exists():
            return False
        try:
            with self._ckpt_path.open(encoding="utf-8") as fh:
                ckpt = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return False
        saved: Optional[list] = ckpt.get("global", {}).get("video_list")
        if saved is None:
            return False
        current_rel = sorted(
            p.relative_to(self._raw_dir).as_posix() for p in current_videos
        )
        return current_rel == sorted(saved)

    # ------------------------------------------------------------------ #
    #  Purge stale data                                                    #
    # ------------------------------------------------------------------ #

    def _purge_stale_data(self) -> None:
        for sub in ("asr", "ocr", "aligned", "map_batches", "drama_map"):
            d = self._cache_dir / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            logger.info("[Init] Purged cache/%s/", sub)

        if self._meta_dir.exists():
            shutil.rmtree(self._meta_dir)
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Init] Purged data/meta/")

        if self._output_dir.exists():
            for srt in self._output_dir.glob("*/*.srt"):
                srt.unlink()
            for report in ("validation_report.json", "cost_report.json",
                           "drama_structure_graph.json"):
                fp = self._output_dir / report
                if fp.exists():
                    fp.unlink()
            logger.info("[Init] Purged output/*/*.srt + reports")

        if self._ckpt_path.exists():
            self._ckpt_path.unlink()
            logger.info("[Init] Deleted checkpoint.json")

    # ------------------------------------------------------------------ #
    #  Patch settings.yaml                                                 #
    # ------------------------------------------------------------------ #

    def _patch_settings_yaml(self, roi: list[float]) -> None:
        """
        Surgically replace the ``roi:`` line inside the active mode's ``ocr:``
        block, preserving all comments and surrounding formatting.
        """
        mode = self._cfg.get("pipeline", {}).get("mode", "same_lang")
        roi_str = f"[{roi[0]}, {roi[1]}]"

        text = self._settings_path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)

        mode_start: Optional[int] = None
        mode_end = len(lines)

        for i, line in enumerate(lines):
            if line.startswith(f"{mode}:"):
                mode_start = i
            elif (
                mode_start is not None
                and i > mode_start
                and line
                and not line[0].isspace()
                and not line.startswith("#")
            ):
                mode_end = i
                break

        if mode_start is None:
            logger.warning(
                "[Init] Section '%s:' not found in settings.yaml — ROI patch skipped.",
                mode,
            )
            return

        replaced = False
        for i in range(mode_start, mode_end):
            stripped = lines[i].lstrip()
            if re.match(r"roi\s*:", stripped):
                indent = " " * (len(lines[i]) - len(lines[i].lstrip()))
                comment_m = re.search(r"(#.*)$", lines[i].rstrip("\n"))
                comment = f"  {comment_m.group(1)}" if comment_m else "  # auto-calibrated"
                lines[i] = f"{indent}roi: {roi_str}{comment}\n"
                replaced = True
                break

        if not replaced:
            logger.warning(
                "[Init] 'roi:' key not found in '%s' section — ROI patch skipped.", mode
            )
            return

        self._settings_path.write_text("".join(lines), encoding="utf-8")
        logger.info(
            "[Init] settings.yaml updated — %s.ocr.roi → %s", mode, roi_str
        )

    # ------------------------------------------------------------------ #
    #  Write checkpoint                                                    #
    # ------------------------------------------------------------------ #

    def _write_checkpoint(self, videos: list[Path]) -> None:
        video_list = sorted(
            p.relative_to(self._raw_dir).as_posix() for p in videos
        )
        self._ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "episodes": {},
            "global": {"video_list": video_list},
        }
        tmp = self._ckpt_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self._ckpt_path)
        logger.info(
            "[Init] Fresh checkpoint written — %d video(s) registered.",
            len(videos),
        )
