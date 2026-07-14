"""
OCR runner — PaddleOCR wrapper for subtitle region extraction.

Samples video frames at the configured FPS, crops to the subtitle ROI, and
runs PaddleOCR.  Output is cached to data/cache/ocr/{episode_id}_ocr.json.

same_lang (OCR as slave / context):
  • Target FPS : 2  (from config)
  • Output     : List of OCRBlock — consecutive identical frames are merged
                 into time-ranged blocks ready for the overlap aligner.

cross_lang (OCR as master timeline):
  • Target FPS : 5  (from config)
  • Output     : List of raw OCRFrame entries (one per sampled frame).
                 Block merging / deduplication is handled by ocr_dedup.py.

Output JSON schema — same_lang:
    { "mode": "same_lang", "blocks": [
        {"start": 12.00, "end": 13.50, "combined_text": "你给我等着",
         "avg_confidence": 0.95, "frame_count": 3}, ... ] }

Output JSON schema — cross_lang:
    { "mode": "cross_lang", "frames": [
        {"frame_time": 12.20, "combined_text": "I w1ll k1ll you",
         "avg_confidence": 0.82,
         "lines": [{"text": "I w1ll k1ll", "confidence": 0.84},
                   {"text": "you",         "confidence": 0.80}]}, ... ] }
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level fallback default (used when not set in settings.yaml)
_MIN_LINE_CONFIDENCE_DEFAULT = 0.50


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OCRLine:
    text: str
    confidence: float


@dataclass
class OCRFrame:
    frame_time: float
    combined_text: str    # space-joined, top-to-bottom sorted lines
    avg_confidence: float
    lines: list[OCRLine]

    @property
    def empty(self) -> bool:
        return not self.combined_text.strip()


@dataclass
class OCRBlock:
    """same_lang only: a run of consecutive frames sharing the same subtitle text."""
    start: float
    end: float
    combined_text: str
    avg_confidence: float
    frame_count: int


# ══════════════════════════════════════════════════════════════════════════════
#  Frame extraction
# ══════════════════════════════════════════════════════════════════════════════


def _iter_frames(
    video_path: Path,
    target_fps: float,
    roi: tuple[float, float],
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Yield (timestamp_sec, roi_cropped_bgr_frame) at approximately target_fps.

    Uses sequential read — no random seeks, which is more efficient for most
    video codecs (especially H.264/H.265 on spinning disk).

    Args:
        video_path : Path to the episode video file.
        target_fps : Desired sampling rate (e.g. 2 for same_lang, 5 for cross_lang).
        roi        : (y_start_ratio, y_end_ratio) as fractions of frame height.

    Yields:
        (timestamp_sec, cropped_frame_array)
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for frame extraction.\n"
            "  pip install opencv-python"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_fps <= 0 or frame_h <= 0:
        cap.release()
        raise ValueError(
            f"Invalid video metadata (fps={video_fps}, h={frame_h}): {video_path}"
        )

    y_start = int(frame_h * roi[0])
    y_end = int(frame_h * roi[1])

    # How many native video frames to skip per sample
    frame_step: float = video_fps / target_fps
    next_target: float = 0.0
    current_idx: int = 0

    logger.debug(
        "Frame extraction — %s  video_fps=%.2f  target_fps=%.1f  "
        "total_frames=%d  roi=[%.0f%%,%.0f%%]",
        video_path.name, video_fps, target_fps,
        total_frames, roi[0] * 100, roi[1] * 100,
    )

    try:
        while current_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx >= next_target:
                timestamp = current_idx / video_fps
                yield timestamp, frame[y_start:y_end, :]
                next_target += frame_step
            current_idx += 1
    finally:
        cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _atomic_json_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parse_ocr_result(
    raw_result: list,
    frame_time: float,
    min_confidence: float = _MIN_LINE_CONFIDENCE_DEFAULT,
) -> Optional[OCRFrame]:
    """
    Convert raw PaddleOCR output for one frame into an OCRFrame.

    Returns None only on an unexpected exception.  Empty frames (no text
    detected, or all detections below confidence threshold) are returned
    as OCRFrame with empty combined_text — callers check frame.empty.

    PaddleOCR result layout:
        raw_result[0] = list of detections, each detection:
            [bbox, (text, confidence)]
        bbox = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    if not raw_result or raw_result[0] is None:
        return OCRFrame(
            frame_time=round(frame_time, 3),
            combined_text="",
            avg_confidence=0.0,
            lines=[],
        )

    # Collect lines with their top-left y coordinate for sorting
    lines_with_y: list[tuple[float, OCRLine]] = []
    for detection in raw_result[0]:
        if not detection or len(detection) < 2:
            continue
        bbox, text_info = detection[0], detection[1]
        if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
            continue
        text, conf = text_info[0], text_info[1]
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(conf, (int, float)) or float(conf) < min_confidence:
            continue
        # y-coordinate of top-left corner for top-to-bottom ordering
        top_y = float(bbox[0][1]) if isinstance(bbox, (list, tuple)) and bbox else 0.0
        lines_with_y.append(
            (top_y, OCRLine(text=text.strip(), confidence=round(float(conf), 4)))
        )

    if not lines_with_y:
        return OCRFrame(
            frame_time=round(frame_time, 3),
            combined_text="",
            avg_confidence=0.0,
            lines=[],
        )

    lines_with_y.sort(key=lambda x: x[0])
    lines = [l for _, l in lines_with_y]
    combined = " ".join(l.text for l in lines)
    avg_conf = sum(l.confidence for l in lines) / len(lines)

    return OCRFrame(
        frame_time=round(frame_time, 3),
        combined_text=combined,
        avg_confidence=round(avg_conf, 4),
        lines=lines,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  OCRRunner
# ══════════════════════════════════════════════════════════════════════════════


class OCRRunner:
    """
    Extracts subtitle text from episode videos using PaddleOCR.

    Cache: data/cache/ocr/{episode_id}_ocr.json
    Cache hit → skips model load entirely (no VRAM used).

    VRAM lifecycle is managed through gpu_manager.scope("paddleocr").
    """

    def __init__(self, cfg: dict) -> None:
        mode: str = cfg["pipeline"]["mode"]
        self._mode = mode
        ocr_cfg: dict = cfg.get(mode, {}).get("ocr", {})

        default_fps = 2.0 if mode == "same_lang" else 5.0
        self._target_fps: float = float(ocr_cfg.get("fps", default_fps))
        roi_raw = ocr_cfg.get("roi", [0.80, 0.95])
        self._roi: tuple[float, float] = (float(roi_raw[0]), float(roi_raw[1]))
        self._language: str = ocr_cfg.get("language", "ch" if mode == "same_lang" else "en")
        self._use_gpu: bool = bool(ocr_cfg.get("use_gpu", True))
        self._min_line_confidence: float = float(
            ocr_cfg.get("min_line_confidence", _MIN_LINE_CONFIDENCE_DEFAULT))

        cache_root = Path(cfg["paths"]["cache_dir"])
        self._cache_dir = cache_root / "ocr"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run_episode(self, video_path: Path, episode_id: str) -> Path:
        """
        Extract OCR for one episode.  Returns path to output JSON.
        Skips silently if cache already exists.

        Raises FileNotFoundError if video_path does not exist.
        """
        out_path = self._cache_dir / f"{episode_id}_ocr.json"
        if out_path.exists():
            logger.info("OCR cache hit — [%s] skipped", episode_id)
            return out_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(
            "OCR start — [%s]  mode=%s  fps=%.1f  lang=%s  roi=%s  use_gpu=%s",
            episode_id, self._mode, self._target_fps,
            self._language, self._roi, self._use_gpu,
        )

        from src.utils.gpu_manager import gpu_manager

        with gpu_manager.scope("paddleocr") as scope:
            ocr_engine = self._init_ocr()
            scope.register(ocr_engine)
            frames = self._collect_frames(ocr_engine, video_path, episode_id)
        # ocr_engine freed here

        if self._mode == "same_lang":
            payload = self._serialize_same_lang(episode_id, video_path, frames)
        else:
            payload = self._serialize_cross_lang(episode_id, video_path, frames)

        _atomic_json_write(out_path, payload)
        logger.info(
            "OCR done — [%s]  frames_extracted=%d  → %s",
            episode_id, len(frames), out_path.name,
        )
        return out_path

    # ------------------------------------------------------------------ #
    #  Engine initialisation                                               #
    # ------------------------------------------------------------------ #

    def _init_ocr(self) -> Any:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required.  Install with:\n"
                "  pip install paddleocr\n"
                "  pip install paddlepaddle-gpu  # or paddlepaddle for CPU"
            ) from exc

        logger.debug(
            "Initialising PaddleOCR — lang=%s  use_gpu=%s", self._language, self._use_gpu
        )
        return PaddleOCR(
            use_gpu=self._use_gpu,
            lang=self._language,
            show_log=False,
        )

    # ------------------------------------------------------------------ #
    #  Frame processing                                                    #
    # ------------------------------------------------------------------ #

    def _collect_frames(
        self,
        ocr_engine: Any,
        video_path: Path,
        episode_id: str,
    ) -> list[OCRFrame]:
        """Iterate sampled frames, run OCR on each, return collected OCRFrame list."""
        frames: list[OCRFrame] = []
        n_empty = 0

        try:
            from tqdm import tqdm
            frame_iter = tqdm(
                _iter_frames(video_path, self._target_fps, self._roi),
                desc=f"OCR [{episode_id}]",
                unit="frame",
                dynamic_ncols=True,
            )
        except ImportError:
            frame_iter = _iter_frames(video_path, self._target_fps, self._roi)

        for frame_time, roi_crop in frame_iter:
            ocr_frame = self._ocr_one_frame(ocr_engine, frame_time, roi_crop)
            if ocr_frame is None:
                continue  # unexpected OCR error — already logged at DEBUG
            frames.append(ocr_frame)
            if ocr_frame.empty:
                n_empty += 1

        non_empty = len(frames) - n_empty
        logger.info(
            "[%s] Frame summary — total=%d  with_text=%d  empty=%d",
            episode_id, len(frames), non_empty, n_empty,
        )
        return frames

    def _ocr_one_frame(
        self,
        ocr_engine: Any,
        frame_time: float,
        roi_crop: np.ndarray,
    ) -> Optional[OCRFrame]:
        try:
            raw = ocr_engine.ocr(roi_crop, cls=False)
            return _parse_ocr_result(raw, frame_time, self._min_line_confidence)
        except Exception as exc:
            logger.debug("OCR error at t=%.3fs (frame skipped): %s", frame_time, exc)
            return None

    # ------------------------------------------------------------------ #
    #  same_lang: merge consecutive frames into OCR blocks                 #
    # ------------------------------------------------------------------ #

    def _merge_into_blocks(self, frames: list[OCRFrame]) -> list[OCRBlock]:
        """
        Group consecutive frames whose normalized text matches into OCRBlocks.

        Uses exact match on stripped/lowercased text — sufficient at 2fps where
        OCR noise across consecutive frames of the same subtitle is minimal.
        Consecutive empty frames reset the current block (subtitle gap).
        """
        interval = 1.0 / self._target_fps
        blocks: list[OCRBlock] = []
        current: Optional[OCRBlock] = None
        current_norm: str = ""

        for frame in frames:
            if frame.empty:
                if current is not None:
                    blocks.append(current)
                    current = None
                    current_norm = ""
                continue

            norm = frame.combined_text.strip().lower()

            if current is None:
                current = OCRBlock(
                    start=frame.frame_time,
                    end=frame.frame_time + interval,
                    combined_text=frame.combined_text,
                    avg_confidence=frame.avg_confidence,
                    frame_count=1,
                )
                current_norm = norm

            elif norm == current_norm:
                # Extend block; update running average confidence
                n = current.frame_count
                current.avg_confidence = (
                    (current.avg_confidence * n + frame.avg_confidence) / (n + 1)
                )
                current.end = frame.frame_time + interval
                current.frame_count = n + 1

            else:
                blocks.append(current)
                current = OCRBlock(
                    start=frame.frame_time,
                    end=frame.frame_time + interval,
                    combined_text=frame.combined_text,
                    avg_confidence=frame.avg_confidence,
                    frame_count=1,
                )
                current_norm = norm

        if current is not None:
            blocks.append(current)

        return blocks

    # ------------------------------------------------------------------ #
    #  Serialization                                                       #
    # ------------------------------------------------------------------ #

    def _serialize_same_lang(
        self,
        episode_id: str,
        video_path: Path,
        frames: list[OCRFrame],
    ) -> dict:
        blocks = self._merge_into_blocks(frames)
        logger.debug("[%s] same_lang OCR — %d blocks from %d frames", episode_id, len(blocks), len(frames))
        return {
            "episode": episode_id,
            "source": str(video_path),
            "mode": "same_lang",
            "fps_sampled": self._target_fps,
            "roi": list(self._roi),
            "block_count": len(blocks),
            "blocks": [
                {
                    "start": round(b.start, 3),
                    "end": round(b.end, 3),
                    "combined_text": b.combined_text,
                    "avg_confidence": round(b.avg_confidence, 4),
                    "frame_count": b.frame_count,
                }
                for b in blocks
            ],
        }

    def _serialize_cross_lang(
        self,
        episode_id: str,
        video_path: Path,
        frames: list[OCRFrame],
    ) -> dict:
        return {
            "episode": episode_id,
            "source": str(video_path),
            "mode": "cross_lang",
            "fps_sampled": self._target_fps,
            "roi": list(self._roi),
            "frame_count": len(frames),
            "frames": [
                {
                    "frame_time": f.frame_time,
                    "combined_text": f.combined_text,
                    "avg_confidence": f.avg_confidence,
                    "lines": [
                        {"text": l.text, "confidence": l.confidence}
                        for l in f.lines
                    ],
                }
                for f in frames
            ],
        }
