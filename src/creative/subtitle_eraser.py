"""
Subtitle eraser with two backends.

backend="delogo"  (default, fast)
──────────────────────────────────
Single FFmpeg invocation using the built-in ``delogo`` filter.  The filter
reconstructs the subtitle band by interpolating from its border pixels — no
Python frame loop, no model inference.  GPU encode via h264_nvenc when
available (auto-detected).

Expected throughput: 3–10× realtime (vs ~0.1× realtime with the inpaint
backend when PaddleOCR falls back to CPU).

backend="inpaint"  (accurate, slow)
─────────────────────────────────────
Frame-level PaddleOCR detection + cv2.inpaint TELEA.  Each frame is decoded
through a pipe, OCR-detected, inpainted, and re-encoded.  Use this backend
only when delogo reconstruction quality is visually unacceptable (e.g. very
complex backgrounds in the subtitle zone).
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SubtitleEraser:
    """
    Erase burned-in subtitles from a video clip.

    Uses PaddleOCR to detect subtitle bounding boxes and cv2.inpaint(TELEA)
    to reconstruct the background.  No external model downloads required.

    Parameters
    ----------
    roi:
        (y_start_ratio, y_end_ratio) — vertical extent of the subtitle band,
        as fractions of frame height.  Matches same_lang.ocr.roi in
        settings.yaml.
    use_gpu:
        Enable GPU for PaddleOCR detection.
    dilation_px:
        Extra pixels to expand each OCR bounding box before masking.
    inpaint_radius:
        Neighbourhood radius for cv2.inpaint TELEA.  3–5 is typical for
        subtitle strokes; increase if characters are very thick.
    ocr_lang:
        PaddleOCR language code.  "ch" for Chinese subtitles (default).
    """

    def __init__(
        self,
        roi: tuple[float, float] = (0.78, 0.94),
        use_gpu: bool = True,
        dilation_px: int = 8,
        inpaint_radius: int = 3,
        ocr_lang: str = "ch",
        backend: str = "delogo",
    ) -> None:
        self._roi            = roi
        self._use_gpu        = use_gpu
        self._dilation       = dilation_px
        self._inpaint_radius = inpaint_radius
        self._ocr_lang       = ocr_lang
        self._backend        = backend
        self._ocr            = None
        self._nvenc_ok: Optional[bool] = None   # lazily probed

    # ------------------------------------------------------------------ #
    #  Lazy model loading                                                  #
    # ------------------------------------------------------------------ #

    def _ensure_ocr(self) -> None:
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang=self._ocr_lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info("SubtitleEraser: PaddleOCR ready (lang=%s)", self._ocr_lang)

    # ------------------------------------------------------------------ #
    #  Video probing                                                       #
    # ------------------------------------------------------------------ #

    def _probe_video(self, path: Path) -> tuple[int, int, float]:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(path)],
            capture_output=True, text=True,
        )
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                w   = int(s["width"])
                h   = int(s["height"])
                num, den = s.get("r_frame_rate", "24/1").split("/")
                fps = float(num) / max(float(den), 1e-6)
                return w, h, fps
        raise ValueError(f"No video stream found in {path}")

    # ------------------------------------------------------------------ #
    #  OCR helpers                                                         #
    # ------------------------------------------------------------------ #

    def _run_ocr(self, roi_band: np.ndarray) -> list:
        try:
            result = self._ocr.ocr(roi_band, cls=False)
            return result if result else []
        except Exception as exc:
            logger.debug("SubtitleEraser: OCR exception — %s", exc)
            return []

    @staticmethod
    def _extract_text(ocr_result: list) -> str:
        parts: list[str] = []
        for page in ocr_result:
            if not page:
                continue
            for line in page:
                if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                    parts.append(str(line[1][0]))
        return "".join(parts)

    def _build_mask(
        self,
        ocr_result: list,
        roi_h: int,
        roi_w: int,
    ) -> Optional[np.ndarray]:
        d    = self._dilation
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        found = False
        for page in ocr_result:
            if not page:
                continue
            for line in page:
                box = line[0]
                xs  = [int(p[0]) for p in box]
                ys  = [int(p[1]) for p in box]
                x1  = max(min(xs) - d, 0)
                x2  = min(max(xs) + d, roi_w)
                y1  = max(min(ys) - d, 0)
                y2  = min(max(ys) + d, roi_h)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255
                    found = True
        return mask if found else None

    # ------------------------------------------------------------------ #
    #  Inpainting                                                          #
    # ------------------------------------------------------------------ #

    def _inpaint(self, roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run cv2.inpaint TELEA on the ROI crop."""
        return cv2.inpaint(roi_bgr, mask, self._inpaint_radius, cv2.INPAINT_TELEA)

    # ------------------------------------------------------------------ #
    #  GPU / encoder helpers                                               #
    # ------------------------------------------------------------------ #

    def _check_nvenc(self) -> bool:
        """Return True if h264_nvenc is usable on this machine (probed once)."""
        if self._nvenc_ok is None:
            r = subprocess.run(
                ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.04",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True,
            )
            self._nvenc_ok = r.returncode == 0
            logger.info(
                "SubtitleEraser: h264_nvenc probe → %s",
                "available" if self._nvenc_ok else "not available (will use libx264)",
            )
        return self._nvenc_ok

    def _encode_args(self) -> list[str]:
        """Return ffmpeg video-encode flags (GPU if nvenc available)."""
        if self._check_nvenc():
            return ["-c:v", "h264_nvenc", "-cq", "18", "-preset", "p4",
                    "-pix_fmt", "yuv420p"]
        return ["-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-pix_fmt", "yuv420p"]

    # ------------------------------------------------------------------ #
    #  delogo backend (default)                                            #
    # ------------------------------------------------------------------ #

    def _process_video_delogo(self, input_path: Path, output_path: Path) -> bool:
        """
        Erase the subtitle ROI band using FFmpeg's delogo filter.

        The filter fills the rectangle by interpolating from its border
        pixels — no per-frame OCR or Python frame loop.  GPU encode is used
        when h264_nvenc is available.
        """
        try:
            w, h, fps = self._probe_video(input_path)
        except Exception as exc:
            logger.error("SubtitleEraser(delogo): probe failed — %s", exc)
            return False

        roi_y0 = int(h * self._roi[0])
        roi_y1 = int(h * self._roi[1])

        # FFmpeg 8 bug: x=0 is treated as "unset" (falsy default), and the
        # right/bottom edges must be strictly inside the frame (< not <=).
        # Fix: start x at 1, reduce w by 2 to keep x+w = w-1 < frame_width.
        dl_x = 1
        dl_w = w - 2
        dl_y = roi_y0
        dl_h = roi_y1 - roi_y0 - 1   # ensure y+h < frame_height

        delogo = f"delogo=x={dl_x}:y={dl_y}:w={dl_w}:h={dl_h}:show=0"
        logger.info(
            "SubtitleEraser(delogo): %s  %dx%d@%.2ffps  ROI y=[%d:%d]  nvenc=%s",
            input_path.name, w, h, fps, roi_y0, roi_y1, self._check_nvenc(),
        )

        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path),
             "-vf", delogo,
             *self._encode_args(),
             "-c:a", "copy",
             "-movflags", "+faststart",
             str(output_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            tail = r.stderr.strip().splitlines()
            logger.error("SubtitleEraser(delogo): %s", tail[-1] if tail else "(empty)")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def process_video(self, input_path: Path, output_path: Path) -> bool:
        """
        Erase subtitles from *input_path* and write the result to *output_path*.
        Audio is preserved.  Returns True on success.
        Dispatches to delogo or inpaint backend based on self._backend.
        """
        if self._backend == "delogo":
            return self._process_video_delogo(input_path, output_path)
        return self._process_video_inpaint(input_path, output_path)

    def _process_video_inpaint(self, input_path: Path, output_path: Path) -> bool:
        """Original per-frame PaddleOCR + cv2.inpaint path."""
        self._ensure_ocr()

        try:
            width, height, fps = self._probe_video(input_path)
        except Exception as exc:
            logger.error("SubtitleEraser: probe failed — %s", exc)
            return False

        roi_y0      = int(height * self._roi[0])
        roi_y1      = int(height * self._roi[1])
        roi_h       = roi_y1 - roi_y0
        frame_bytes = width * height * 3

        logger.info(
            "SubtitleEraser: %s  %dx%d @ %.2ffps  ROI y=[%d:%d]",
            input_path.name, width, height, fps, roi_y0, roi_y1,
        )

        # ── Decode: ffmpeg → raw BGR frames ──────────────────────────────
        decode_proc = subprocess.Popen(
            ["ffmpeg", "-i", str(input_path),
             "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # ── Encode: processed frames + original audio ─────────────────────
        encode_proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "pipe:0",
                "-i", str(input_path),
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # ── Frame processing loop ─────────────────────────────────────────
        prev_text: str               = "\x00"   # sentinel — never matches OCR
        prev_mask: Optional[np.ndarray] = None
        n_frames = n_inpaint = n_clean = 0
        t0 = time.monotonic()

        try:
            while True:
                raw = decode_proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break

                frame   = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (height, width, 3)
                ).copy()
                roi_bgr = frame[roi_y0:roi_y1, :]

                ocr_result = self._run_ocr(roi_bgr)
                cur_text   = self._extract_text(ocr_result)

                if cur_text:
                    # Rebuild mask only when text changes (text usually holds
                    # for 2-4 s; background changes every frame so inpaint runs
                    # every frame regardless)
                    if cur_text != prev_text or prev_mask is None:
                        prev_mask = self._build_mask(ocr_result, roi_h, width)
                        prev_text = cur_text

                    if prev_mask is not None:
                        frame[roi_y0:roi_y1, :] = self._inpaint(roi_bgr, prev_mask)
                        n_inpaint += 1
                else:
                    prev_text = "\x00"
                    prev_mask = None
                    n_clean  += 1

                encode_proc.stdin.write(frame.tobytes())
                n_frames += 1

        except BrokenPipeError:
            logger.error(
                "SubtitleEraser: encode pipe broke after %d frames", n_frames
            )
        finally:
            try:
                encode_proc.stdin.close()
            except OSError:
                pass

        decode_proc.wait()
        rc = encode_proc.wait()

        elapsed   = time.monotonic() - t0
        video_sec = n_frames / max(fps, 1)
        logger.info(
            "SubtitleEraser: done  frames=%d  inpainted=%d  clean=%d  "
            "%.1fs (%.2fx realtime)",
            n_frames, n_inpaint, n_clean, elapsed,
            elapsed / max(video_sec, 0.01),
        )

        if rc != 0:
            logger.error("SubtitleEraser: ffmpeg encode exited rc=%d", rc)
            return False
        return True
