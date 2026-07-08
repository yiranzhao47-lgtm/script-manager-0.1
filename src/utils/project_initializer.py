"""
ProjectInitializer — auto-detects show changes and self-heals the pipeline.

Called once at the top of every pipeline run, before any stage executes:

  Resume scenario  (same video set as last run):
      → silently exits; the pipeline continues its checkpoint-based resume.

  New-show scenario (video set changed, or no checkpoint exists):
      1. Purges all stale cache from the previous show.
      2. Samples the frame at t=15 s from the first video for a full-screen OCR scan.
      3. Calls the LLM (system prompt: config/prompts/init_agent_skill.md) to
         infer the correct subtitle ROI from the detected text-block positions.
      4. Patches config/settings.yaml with the recommended ROI value.
      5. Writes a fresh checkpoint seeded with the current video list so the
         next run can distinguish resume from new-show again.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Resolved once at import time; allows tests to override via the _root param.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class ProjectInitializer:
    """
    Stateless guard that runs before the main pipeline and handles
    show-change detection, cache purge, and ROI self-healing.

    Parameters
    ----------
    cfg:
        Parsed ``config/settings.yaml`` dict (same object passed to
        ShortDramaPipeline).
    _root:
        Override the project root directory (used in unit tests to avoid
        touching real project files).
    """

    def __init__(self, cfg: dict, _root: Optional[Path] = None) -> None:
        root = _root or _PROJECT_ROOT
        self._cfg = cfg
        self._raw_dir       = root / cfg["paths"]["raw_video_dir"]
        self._cache_dir     = root / cfg["paths"]["cache_dir"]
        self._meta_dir      = root / cfg["paths"]["meta_dir"]
        self._output_dir    = root / cfg["paths"]["output_dir"]
        self._ckpt_path     = self._cache_dir / "checkpoint.json"
        self._prompt_path   = root / "config" / "prompts" / "init_agent_skill.md"
        self._settings_path = root / "config" / "settings.yaml"

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def auto_detect_and_heal(self) -> None:
        """
        Detect the current scenario and act accordingly.

        Resume scenario   → prints one info line, returns immediately.
        New-show scenario → purge → OCR sample → LLM ROI → patch YAML → write checkpoint.
        """
        current_videos = self._scan_videos()

        if self._is_resume(current_videos):
            logger.info(
                "[Init] Video set unchanged (%d file(s)) — resuming from checkpoint.",
                len(current_videos),
            )
            return

        logger.info(
            "[Init] New show detected (%d video(s)) — activating self-heal workflow.",
            len(current_videos),
        )

        self._purge_stale_data()

        if current_videos:
            try:
                ocr_payload = self._sample_and_ocr(current_videos)
                roi = self._call_llm_for_roi(ocr_payload)
                self._patch_settings_yaml(roi)
            except Exception as exc:
                logger.warning(
                    "[Init] ROI self-heal failed — keeping existing ROI from settings.yaml.  "
                    "Error: %s", exc,
                )
        else:
            logger.warning(
                "[Init] No .mp4 files found under %s — OCR/ROI heal skipped.",
                self._raw_dir,
            )

        self._write_checkpoint(current_videos)

    # ------------------------------------------------------------------ #
    #  Step 1: Scenario detection                                          #
    # ------------------------------------------------------------------ #

    def _scan_videos(self) -> list[Path]:
        """Return all .mp4 files under raw_dir, sorted by full relative path."""
        if not self._raw_dir.exists():
            return []
        return sorted(self._raw_dir.rglob("*.mp4"), key=lambda p: str(p))

    def _is_resume(self, current_videos: list[Path]) -> bool:
        """
        Return True only when the checkpoint exists AND its stored video list
        exactly matches the current scan (same paths, same count).
        """
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
    #  Step 2: Purge stale data                                            #
    # ------------------------------------------------------------------ #

    def _purge_stale_data(self) -> None:
        """
        Physically remove all cache artifacts from the previous show.

        Cache sub-directories are re-created empty so downstream code never
        has to handle a missing directory.  Per-episode output files (SRTs,
        reports) are deleted so they don't masquerade as cache hits for a
        new show that shares the same episode numbering.
        """
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

        # Delete stale per-episode SRTs and run-level reports from output dir
        if self._output_dir.exists():
            for srt in self._output_dir.glob("*.srt"):
                srt.unlink()
            for report in ("validation_report.json", "cost_report.json",
                           "drama_structure_graph.json"):
                fp = self._output_dir / report
                if fp.exists():
                    fp.unlink()
            logger.info("[Init] Purged data/output/ (SRTs + reports)")

        if self._ckpt_path.exists():
            self._ckpt_path.unlink()
            logger.info("[Init] Deleted checkpoint.json")

    # ------------------------------------------------------------------ #
    #  Step 3: Full-screen OCR sampling                                    #
    # ------------------------------------------------------------------ #

    def _sample_and_ocr(self, videos: list[Path]) -> dict:
        """
        Open the first video, seek to t=15 s, run PaddleOCR on the full frame,
        and return a structured payload containing every detected text block with
        its normalised Y-axis coordinates.
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required.  Install with: pip install opencv-python"
            ) from exc

        first = videos[0]
        logger.info("[Init] Opening '%s' for full-screen OCR at t=15 s …", first.name)

        cap = cv2.VideoCapture(str(first))
        if not cap.isOpened():
            logger.warning("[Init] Cannot open '%s' — returning empty OCR payload.", first.name)
            return _empty_ocr_payload(first.name)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(15 * fps))
        ok, frame = cap.read()
        cap.release()

        if not ok:
            logger.warning("[Init] Could not read frame at t=15 s from '%s'.", first.name)
            return _empty_ocr_payload(first.name)

        height, width = frame.shape[:2]

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required.  Install with: pip install paddleocr"
            ) from exc

        engine = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
        raw = engine.ocr(frame, cls=False)

        blocks: list[dict] = []
        if raw and raw[0]:
            for detection in raw[0]:
                if not detection or len(detection) < 2:
                    continue
                box, text_info = detection[0], detection[1]
                if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                    continue
                text, conf = text_info[0], text_info[1]
                if not isinstance(text, str) or not text.strip():
                    continue
                ys = [float(pt[1]) for pt in box]
                blocks.append({
                    "text": text.strip(),
                    "confidence": round(float(conf), 3),
                    "y_top": round(min(ys) / height, 4),
                    "y_bottom": round(max(ys) / height, 4),
                })

        logger.info("[Init] Full-screen OCR complete — %d text block(s) detected.", len(blocks))
        return {
            "frame_info": {
                "video": first.name,
                "timestamp_sec": 15,
                "frame_width": width,
                "frame_height": height,
            },
            "ocr_blocks": blocks,
        }

    # ------------------------------------------------------------------ #
    #  Step 4: LLM ROI inference                                           #
    # ------------------------------------------------------------------ #

    def _call_llm_for_roi(self, ocr_payload: dict) -> list[float]:
        """
        Feed the OCR payload to the LLM using init_agent_skill.md as the
        system prompt.  Parse and clamp the returned recommended_roi.
        """
        system_prompt = self._prompt_path.read_text(encoding="utf-8")
        user_prompt = json.dumps(ocr_payload, ensure_ascii=False, indent=2)

        from src.utils.llm_client import LLMClient, extract_json

        llm = LLMClient(self._cfg)
        raw = llm.complete(
            system=system_prompt,
            user=user_prompt,
            json_mode=True,
            module_name="ROI_Auto_Heal",
        )
        data = extract_json(raw)

        roi_raw = data.get("recommended_roi", [0.78, 0.94])
        roi = [max(0.0, min(1.0, float(v))) for v in roi_raw[:2]]

        logger.info(
            "[Init] LLM ROI → %s  (scenario=%s  reason: %s)",
            roi,
            data.get("detected_scenario", "?"),
            data.get("reason", "—"),
        )
        return roi

    # ------------------------------------------------------------------ #
    #  Step 5: Patch settings.yaml                                         #
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

        # Locate the target top-level section (e.g. "same_lang:")
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

        # Within that section, replace the first ``roi:`` line
        replaced = False
        for i in range(mode_start, mode_end):
            stripped = lines[i].lstrip()
            if re.match(r"roi\s*:", stripped):
                indent = " " * (len(lines[i]) - len(stripped))
                comment_m = re.search(r"(#.*)$", lines[i].rstrip("\n"))
                comment = f"  {comment_m.group(1)}" if comment_m else ""
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
            "[Init] ✓ settings.yaml auto-healed — %s.ocr.roi → %s",
            mode, roi_str,
        )

    # ------------------------------------------------------------------ #
    #  Step 6: Write fresh checkpoint                                      #
    # ------------------------------------------------------------------ #

    def _write_checkpoint(self, videos: list[Path]) -> None:
        """
        Write a minimal checkpoint that records the current video list so the
        next pipeline run can correctly identify the resume scenario.

        Uses the same atomic write convention as src.utils.checkpoint.Checkpoint.
        """
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
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._ckpt_path)
        logger.info(
            "[Init] Fresh checkpoint written — %d video(s) registered.",
            len(videos),
        )


# ── Module-level helper ───────────────────────────────────────────────────────


def _empty_ocr_payload(video_name: str) -> dict:
    return {
        "frame_info": {"video": video_name, "timestamp_sec": 15},
        "ocr_blocks": [],
    }
