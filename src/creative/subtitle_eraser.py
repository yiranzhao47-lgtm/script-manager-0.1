"""
Frame-level subtitle eraser using LaMa inpainting + PaddleOCR detection.

Algorithm per frame
───────────────────
1. Crop the subtitle ROI band from the frame (bottom ~16 % by default).
2. Run PaddleOCR on the crop to detect text bounding boxes.
3. Temporal cache: if the detected text is identical to the previous frame's
   text, reuse the existing mask and skip LaMa (subtitles stay on-screen for
   2-4 s ≈ 48-96 frames at 24 fps; this eliminates ~95 % of LaMa calls within
   each subtitle's on-screen window).
4. If text changed or first occurrence: build a dilated binary mask from the
   bounding boxes and call LaMa to inpaint the ROI crop.
5. Paste the inpainted crop back into the full frame.
6. Write the processed frame to the ffmpeg encode pipe.

Frame pipeline
──────────────
ffmpeg decode (stdout pipe) → numpy BGR frames
    → process_frame()
        → PaddleOCR on ROI
        → LaMa on ROI (mask cached between identical frames)
        → paste ROI back
→ ffmpeg encode (stdin pipe)   [reads processed video; audio copied from source]

Dependencies (must be installed separately)
────────────────────────────────────────────
    pip install simple-lama-inpainting
    paddlepaddle-gpu and paddleocr are already used elsewhere in this project.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_IMPORT_ERROR = (
    "simple-lama-inpainting is required for subtitle erasure.\n"
    "Install with: pip install simple-lama-inpainting"
)


class SubtitleEraser:
    """
    Erase burned-in subtitles from a video clip using PaddleOCR + LaMa.

    Parameters
    ----------
    roi:
        (y_start_ratio, y_end_ratio) — vertical extent of the subtitle band,
        expressed as fractions of frame height.  Matches same_lang.ocr.roi in
        settings.yaml so the eraser targets exactly the zone OCR was tuned for.
    use_gpu:
        Forward GPU flag to both PaddleOCR and LaMa.  Falls back to CPU
        silently if CUDA is unavailable.
    dilation_px:
        Extra pixels to expand each detected bounding box before masking.
        Catches anti-aliased edges that OCR boxes occasionally clip.
    ocr_lang:
        PaddleOCR language code for the subtitle text being erased.
        "ch" for Chinese (default), "en" for English.
    """

    def __init__(
        self,
        roi: tuple[float, float] = (0.78, 0.94),
        use_gpu: bool = True,
        dilation_px: int = 8,
        ocr_lang: str = "ch",
    ) -> None:
        self._roi = roi
        self._use_gpu = use_gpu
        self._dilation_px = dilation_px
        self._ocr_lang = ocr_lang
        self._lama = None
        self._ocr = None

    # ------------------------------------------------------------------ #
    #  Lazy model loading                                                  #
    # ------------------------------------------------------------------ #

    def _ensure_models(self) -> None:
        if self._lama is None:
            try:
                from simple_lama_inpainting import SimpleLama
            except ImportError as exc:
                raise ImportError(_IMPORT_ERROR) from exc

            device = "cuda" if self._use_gpu else "cpu"
            try:
                self._lama = SimpleLama(device=device)
                logger.info("SubtitleEraser: LaMa loaded (device=%s)", device)
            except Exception:
                logger.warning(
                    "SubtitleEraser: LaMa CUDA init failed — falling back to CPU"
                )
                self._lama = SimpleLama(device="cpu")

        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang=self._ocr_lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info("SubtitleEraser: PaddleOCR loaded (lang=%s)", self._ocr_lang)

    # ------------------------------------------------------------------ #
    #  Video probing                                                       #
    # ------------------------------------------------------------------ #

    def _probe_video(self, path: Path) -> tuple[int, int, float]:
        """Return (width, height, fps) via ffprobe."""
        r = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", str(path),
            ],
            capture_output=True, text=True,
        )
        data = json.loads(r.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                w = int(stream["width"])
                h = int(stream["height"])
                fps_str = stream.get("r_frame_rate", "24/1")
                num, den = fps_str.split("/")
                fps = float(num) / max(float(den), 1e-6)
                return w, h, fps
        raise ValueError(f"No video stream found in {path}")

    # ------------------------------------------------------------------ #
    #  OCR helpers                                                         #
    # ------------------------------------------------------------------ #

    def _ocr_on_roi(self, roi_band: np.ndarray) -> list:
        """Run PaddleOCR on the ROI crop.  Returns raw OCR result list."""
        try:
            result = self._ocr.ocr(roi_band, cls=False)
            return result if result else []
        except Exception as exc:
            logger.debug("SubtitleEraser: OCR error — %s", exc)
            return []

    def _extract_text(self, ocr_result: list) -> str:
        """Concatenate all detected text strings for temporal cache key."""
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
        """
        Convert OCR bounding boxes to a binary mask (uint8, 0/255).
        Returns None when no boxes are found.
        """
        d = self._dilation_px
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        found = False

        for page in ocr_result:
            if not page:
                continue
            for line in page:
                box = line[0]   # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                x1 = max(min(xs) - d, 0)
                x2 = min(max(xs) + d, roi_w)
                y1 = max(min(ys) - d, 0)
                y2 = min(max(ys) + d, roi_h)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255
                    found = True

        return mask if found else None

    # ------------------------------------------------------------------ #
    #  LaMa inpainting                                                    #
    # ------------------------------------------------------------------ #

    def _inpaint_roi(
        self,
        roi_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Run LaMa inpainting on a BGR ROI crop.
        Returns the inpainted BGR array at the original size.
        """
        from PIL import Image

        roi_h, roi_w = roi_bgr.shape[:2]

        # BGR → RGB PIL
        roi_pil = Image.fromarray(roi_bgr[:, :, ::-1])
        mask_pil = Image.fromarray(mask)

        result_pil = self._lama(roi_pil, mask_pil)

        # Resize back if LaMa changed dimensions
        if result_pil.size != (roi_w, roi_h):
            result_pil = result_pil.resize((roi_w, roi_h), Image.LANCZOS)

        # RGB → BGR numpy
        return np.array(result_pil)[:, :, ::-1]

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def process_video(self, input_path: Path, output_path: Path) -> bool:
        """
        Erase subtitles from *input_path* and write the result to *output_path*.

        Audio is preserved by muxing from the original file.
        Returns True on success, False on failure.
        """
        self._ensure_models()

        try:
            width, height, fps = self._probe_video(input_path)
        except Exception as exc:
            logger.error(
                "SubtitleEraser: probe failed for %s — %s", input_path.name, exc
            )
            return False

        roi_y0 = int(height * self._roi[0])
        roi_y1 = int(height * self._roi[1])
        roi_h  = roi_y1 - roi_y0
        frame_bytes = width * height * 3  # BGR, uint8

        logger.info(
            "SubtitleEraser: %s  %dx%d @ %.2ffps  ROI y=[%d:%d]",
            input_path.name, width, height, fps, roi_y0, roi_y1,
        )

        # ── Decode subprocess ─────────────────────────────────────────────
        decode_proc = subprocess.Popen(
            [
                "ffmpeg", "-i", str(input_path),
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # ── Encode subprocess ─────────────────────────────────────────────
        # Input 0: raw BGR frames from stdin (processed video)
        # Input 1: original file (audio track only, stream-copied)
        encode_proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "pipe:0",
                "-i", str(input_path),
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # ── Frame processing loop ─────────────────────────────────────────
        prev_text: str = "\x00"   # sentinel — never matches real OCR output
        prev_mask: Optional[np.ndarray] = None
        n_frames = n_lama = n_cached = n_clean = 0
        t0 = time.monotonic()

        try:
            while True:
                raw = decode_proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (height, width, 3)
                ).copy()  # writeable copy

                roi_band = frame[roi_y0:roi_y1, :]

                # OCR on subtitle band
                ocr_result = self._ocr_on_roi(roi_band)
                cur_text   = self._extract_text(ocr_result)

                if cur_text:
                    if cur_text == prev_text and prev_mask is not None:
                        # Temporal cache hit — subtitle unchanged, reuse mask
                        mask = prev_mask
                        n_cached += 1
                    else:
                        mask = self._build_mask(ocr_result, roi_h, width)
                        prev_text = cur_text
                        prev_mask = mask

                    if mask is not None:
                        frame[roi_y0:roi_y1, :] = self._inpaint_roi(roi_band, mask)
                        n_lama += 1
                else:
                    # No subtitle detected — reset cache
                    prev_text = "\x00"
                    prev_mask = None
                    n_clean += 1

                encode_proc.stdin.write(frame.tobytes())
                n_frames += 1

        except BrokenPipeError:
            logger.error(
                "SubtitleEraser: encode pipe broke after %d frames — "
                "output may be incomplete",
                n_frames,
            )
        finally:
            try:
                encode_proc.stdin.close()
            except OSError:
                pass

        decode_proc.wait()
        rc = encode_proc.wait()

        elapsed = time.monotonic() - t0
        video_sec = n_frames / max(fps, 1)
        realtime_ratio = elapsed / max(video_sec, 0.01)

        logger.info(
            "SubtitleEraser: %s done  frames=%d  LaMa=%d  cached=%d  clean=%d  "
            "%.1fs (%.2fx realtime)",
            input_path.name, n_frames, n_lama, n_cached, n_clean,
            elapsed, realtime_ratio,
        )

        if rc != 0:
            logger.error("SubtitleEraser: ffmpeg encode exited rc=%d", rc)
            return False
        return True
