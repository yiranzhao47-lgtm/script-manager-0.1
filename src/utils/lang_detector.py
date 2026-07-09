"""
Pre-flight language detection assertion.

Samples frames from the first N episode videos, runs PaddleOCR (CPU) on the
subtitle ROI, and validates that the dominant script family (CJK vs. Latin)
matches the configured pipeline mode.

Raises LanguageMismatchError (fatal by default) when the config and the actual
video content disagree — prevents silent downstream garbage output and wasted
API tokens before the pipeline has processed a single episode.

Typical usage
─────────────
    from src.utils.lang_detector import run_preflight

    run_preflight(cfg, video_dir=Path("data/raw"))   # raises on mismatch
"""
from __future__ import annotations

import logging
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── CJK Unicode ranges used for script classification ────────────────────────
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs (core Han)
    (0x3400, 0x4DBF),  # CJK Extension A
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
)

# Video extensions the detector will consider
_VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".m4v", ".ts"}
)


# ══════════════════════════════════════════════════════════════════════════════
#  Exceptions
# ══════════════════════════════════════════════════════════════════════════════


class LanguageMismatchError(RuntimeError):
    """
    Fatal: detected subtitle language contradicts the configured pipeline mode.
    Downstream processing is aborted to prevent silent garbage output and
    wasted LLM API tokens.
    """


# ══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EpisodeResult:
    path: Path
    frames_sampled: int        # frames successfully extracted from the video
    frames_with_text: int      # frames that produced at least one OCR hit
    total_chars: int           # total non-whitespace characters across all frames
    cjk_chars: int             # subset that are CJK ideographs
    cjk_ratio: float           # cjk_chars / total_chars  (0.0 if total_chars == 0)
    dominant: str              # "cjk" | "latin" | "unknown"


@dataclass
class DetectionReport:
    mode_configured: str
    episodes: list[EpisodeResult] = field(default_factory=list)
    overall_cjk_ratio: float = 0.0
    detected_dominant: str = "unknown"
    match: bool = True


# ══════════════════════════════════════════════════════════════════════════════
#  Character-level helpers
# ══════════════════════════════════════════════════════════════════════════════


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _cjk_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are CJK ideographs."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _is_cjk(c)) / len(chars)


def _natural_sort_key(p: Path) -> tuple:
    """Sort filenames naturally: ep2 < ep10 < ep20."""
    parts = re.split(r"(\d+)", p.stem)
    return tuple(int(c) if c.isdigit() else c.lower() for c in parts)


# ══════════════════════════════════════════════════════════════════════════════
#  LangDetector
# ══════════════════════════════════════════════════════════════════════════════


class LangDetector:
    """
    Validates that subtitle language in the first N episodes matches config mode.

    OCR runs on CPU (use_gpu=False) to avoid VRAM pressure on the pre-flight
    check — accuracy is sufficient for script-family detection.

    Parameters
    ──────────
    cfg        : parsed settings.yaml dict
    video_dir  : directory containing raw episode video files
    """

    def __init__(self, cfg: dict, video_dir: Path) -> None:
        self._mode: str = cfg["pipeline"]["mode"]
        self._source_language: str = cfg["pipeline"].get("source_language", "zh")

        ld = cfg.get("lang_detection", {})
        self._n_episodes: int = ld.get("episodes_to_check", 3)
        self._frames_per_ep: int = ld.get("frames_per_episode", 5)
        self._cjk_threshold: float = float(ld.get("cjk_ratio_threshold", 0.30))
        self._fatal: bool = bool(ld.get("fatal_on_mismatch", True))

        # Pull ROI from the mode-specific OCR config
        ocr_cfg = cfg.get(self._mode, {}).get("ocr", {})
        roi = ocr_cfg.get("roi", [0.80, 0.95])
        self._roi: tuple[float, float] = (float(roi[0]), float(roi[1]))
        # Use source_language to select OCR model; "ch" is PaddleOCR's Chinese code
        self._ocr_lang: str = "ch" if self._source_language == "zh" else "en"

        self._video_dir = Path(video_dir)
        self._ocr: Optional[object] = None  # lazy-initialised on first frame

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> DetectionReport:
        """
        Execute pre-flight detection across the first N episodes.

        Returns a DetectionReport.
        Raises LanguageMismatchError if a mismatch is detected and
        lang_detection.fatal_on_mismatch == true.
        """
        videos = self._discover_videos()
        if not videos:
            logger.warning(
                "LangDetector: no video files found in '%s' — skipping pre-flight",
                self._video_dir,
            )
            return DetectionReport(mode_configured=self._mode)

        logger.info(
            "LangDetector starting — mode=%s  checking %d/%d episode(s)",
            self._mode,
            min(self._n_episodes, len(videos)),
            len(videos),
        )

        report = DetectionReport(mode_configured=self._mode)
        for vp in videos[: self._n_episodes]:
            result = self._detect_episode(vp)
            report.episodes.append(result)
            logger.info(
                "  %-30s  frames=%d/%d  cjk=%.1f%%  dominant=%s",
                vp.name,
                result.frames_with_text,
                result.frames_sampled,
                result.cjk_ratio * 100,
                result.dominant,
            )

        report = self._finalize(report)
        self._check_mismatch(report)
        return report

    # ------------------------------------------------------------------ #
    #  Per-episode detection                                               #
    # ------------------------------------------------------------------ #

    def _detect_episode(self, video_path: Path) -> EpisodeResult:
        frames = self._sample_frames(video_path)
        text, frames_with_text = self._run_ocr(frames)

        non_ws = [c for c in text if not c.isspace()]
        total = len(non_ws)
        cjk = sum(1 for c in non_ws if _is_cjk(c))
        ratio = cjk / total if total > 0 else 0.0
        dominant = (
            "cjk"
            if ratio >= self._cjk_threshold
            else ("latin" if total > 0 else "unknown")
        )
        return EpisodeResult(
            path=video_path,
            frames_sampled=len(frames),
            frames_with_text=frames_with_text,
            total_chars=total,
            cjk_chars=cjk,
            cjk_ratio=ratio,
            dominant=dominant,
        )

    # ------------------------------------------------------------------ #
    #  Frame sampling                                                      #
    # ------------------------------------------------------------------ #

    def _sample_frames(self, video_path: Path) -> list[np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for LangDetector frame sampling. "
                "Install with: pip install opencv-python"
            ) from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("LangDetector: cannot open video '%s'", video_path)
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total <= 0 or h <= 0:
            cap.release()
            logger.warning(
                "LangDetector: invalid video metadata (frames=%d h=%d) for '%s'",
                total,
                h,
                video_path,
            )
            return []

        # Crop subtitle ROI
        y_start = int(h * self._roi[0])
        y_end = int(h * self._roi[1])

        # Sample uniformly, avoiding first/last 5% (credits, black frames)
        margin = max(int(total * 0.05), 1)
        range_start = margin
        range_end = max(total - margin, range_start + 1)
        usable = list(range(range_start, range_end))
        n = min(self._frames_per_ep, len(usable))

        if n <= 0:
            cap.release()
            return []

        indices = sorted(random.sample(usable, n))
        frames: list[np.ndarray] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                cropped = frame[y_start:y_end, :]
                frames.append(cropped)

        cap.release()
        return frames

    # ------------------------------------------------------------------ #
    #  OCR                                                                 #
    # ------------------------------------------------------------------ #

    def _run_ocr(self, frames: list[np.ndarray]) -> tuple[str, int]:
        """
        Run OCR on all sampled frames.

        Returns (combined_text, n_frames_that_yielded_text).
        Low-confidence results (< 0.5) are discarded to avoid noisy chars
        inflating the ratio in either direction.
        """
        if not frames:
            return "", 0

        ocr = self._get_ocr()
        if ocr is None:
            logger.warning(
                "LangDetector: PaddleOCR unavailable — detection degraded"
            )
            return "", 0

        collected: list[str] = []
        frames_with_text = 0

        for frame in frames:
            try:
                results = ocr.ocr(frame, cls=False)
                if not results or results[0] is None:
                    continue

                frame_texts: list[str] = []
                for line in results[0]:
                    if not line or len(line) < 2 or not line[1]:
                        continue
                    text_item = line[1]
                    if not isinstance(text_item, (list, tuple)) or len(text_item) < 2:
                        continue
                    text, conf = text_item[0], text_item[1]
                    if isinstance(text, str) and isinstance(conf, float) and conf >= 0.5:
                        frame_texts.append(text)

                if frame_texts:
                    collected.extend(frame_texts)
                    frames_with_text += 1

            except Exception as exc:
                logger.debug("LangDetector: OCR frame error (skipped): %s", exc)

        return " ".join(collected), frames_with_text

    def _get_ocr(self) -> Optional[object]:
        """Lazy-init PaddleOCR on CPU to avoid pipeline VRAM pressure."""
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_gpu=False,       # CPU — sufficient for script-family detection
                lang=self._ocr_lang,
                show_log=False,
            )
            logger.debug(
                "PaddleOCR initialised (CPU, lang=%s) for pre-flight check",
                self._ocr_lang,
            )
        except Exception as exc:
            logger.warning("LangDetector: PaddleOCR init failed: %s", exc)
            self._ocr = None
        return self._ocr

    # ------------------------------------------------------------------ #
    #  Result aggregation and mismatch check                               #
    # ------------------------------------------------------------------ #

    def _finalize(self, report: DetectionReport) -> DetectionReport:
        valid = [e for e in report.episodes if e.total_chars > 0]
        if not valid:
            report.overall_cjk_ratio = 0.0
            report.detected_dominant = "unknown"
            report.match = True  # cannot determine — warn only, don't abort
            logger.warning(
                "LangDetector: all sampled frames yielded no OCR text.  "
                "Check roi settings or video file integrity."
            )
            return report

        total_ch = sum(e.total_chars for e in valid)
        total_cjk = sum(e.cjk_chars for e in valid)
        report.overall_cjk_ratio = total_cjk / total_ch
        report.detected_dominant = (
            "cjk" if report.overall_cjk_ratio >= self._cjk_threshold else "latin"
        )

        expected = "cjk" if self._source_language == "zh" else "latin"
        report.match = report.detected_dominant == expected
        return report

    def _check_mismatch(self, report: DetectionReport) -> None:
        if report.detected_dominant == "unknown":
            logger.warning(
                "LangDetector: dominant language undetermined "
                "(no usable OCR text) — proceeding with caution"
            )
            return

        if report.match:
            logger.info(
                "LangDetector PASSED — mode=%s  detected=%s  cjk_ratio=%.1f%%",
                report.mode_configured,
                report.detected_dominant,
                report.overall_cjk_ratio * 100,
            )
            return

        # Build a detailed, actionable error message
        expected_label = (
            "CJK / Chinese" if report.mode_configured == "same_lang" else "Latin / English"
        )
        detected_label = (
            "CJK / Chinese" if report.detected_dominant == "cjk" else "Latin / English"
        )
        breakdown = "\n".join(
            f"    {e.path.name:<30}  cjk={e.cjk_ratio:.1%}  "
            f"chars={e.total_chars}  frames_with_text={e.frames_with_text}"
            for e in report.episodes
        )
        msg = (
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║           FATAL: LANGUAGE MISMATCH DETECTED                 ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
            f"  Configured mode   : {report.mode_configured}\n"
            f"  Expected script   : {expected_label}\n"
            f"  Detected script   : {detected_label}\n"
            f"  Overall CJK ratio : {report.overall_cjk_ratio:.1%}\n"
            f"  CJK threshold     : {self._cjk_threshold:.1%}\n"
            f"\n"
            f"  Episode breakdown:\n{breakdown}\n"
            f"\n"
            f"  ► Fix: set 'pipeline.mode' in config/settings.yaml to match\n"
            f"         the actual subtitle language and re-run the pipeline.\n"
        )

        if self._fatal:
            logger.critical(msg)
            raise LanguageMismatchError(msg)
        else:
            logger.warning(msg)

    # ------------------------------------------------------------------ #
    #  Video discovery                                                     #
    # ------------------------------------------------------------------ #

    def _discover_videos(self) -> list[Path]:
        if not self._video_dir.is_dir():
            logger.error(
                "LangDetector: video directory '%s' does not exist",
                self._video_dir,
            )
            return []

        videos = sorted(
            (p for p in self._video_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS),
            key=_natural_sort_key,
        )

        if not videos:
            logger.warning(
                "LangDetector: no video files found in '%s'", self._video_dir
            )
        else:
            logger.debug(
                "LangDetector: found %d video(s) in '%s'",
                len(videos),
                self._video_dir,
            )

        return videos


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════


def run_preflight(cfg: dict, video_dir: Path) -> DetectionReport:
    """
    Convenience wrapper: construct LangDetector, run, return report.

    Raises LanguageMismatchError if mismatch detected and fatal_on_mismatch=true.
    """
    return LangDetector(cfg, video_dir).run()
