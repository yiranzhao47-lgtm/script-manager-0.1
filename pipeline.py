"""
cc_script_manager — Short Drama Subtitle Pipeline
══════════════════════════════════════════════════
Orchestrates all pipeline stages end-to-end for a full drama series:

  Stage 0  Pre-flight     — language / mode mismatch assertion (LangDetector)
  Stage 1  Ingestion      — ASR + OCR (+ OCR dedup for cross_lang) per episode
  Stage 2  Alignment      — time-axis ASR↔OCR merge → aligned/*.json
  Stage 3  MapReduce      — entity extraction → meta_raw.json + meta.json
  Stage 4  Refinement     — per-episode LLM polish → output/*.srt

Resume / checkpoint
───────────────────
State is tracked in data/cache/checkpoint.json.  Re-running the pipeline after
a partial failure picks up exactly where it left off — no re-work.  Individual
stages also have their own file-based checkpoints (cache hit → skip).

Usage
─────
    python pipeline.py                              # defaults from config/settings.yaml
    python pipeline.py --mode cross_lang            # override mode for one run
    python pipeline.py --video-dir /data/drama/raw  # override video directory
    python pipeline.py --skip-preflight             # skip lang detection (fast restart)
    python pipeline.py --config my_settings.yaml    # use a custom config file
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

# Pre-load torch before any PaddleOCR/PaddlePaddle import to prevent DLL
# ordering conflicts on Windows (shm.dll fails if paddle loads its DLLs first).
try:
    import torch as _torch  # noqa: F401
except ImportError:
    pass

# ── Logging setup (done before any module imports that log at load time) ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("pipeline")

# ── Project root (resolved once at import time) ───────────────────────────────
_ROOT = Path(__file__).resolve().parent

# ── Windows sleep prevention ──────────────────────────────────────────────────
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def _prevent_sleep() -> None:
    """Tell Windows not to sleep or turn off the display while pipeline runs."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
        )
        logger.info("Sleep prevention enabled (SetThreadExecutionState)")
    except Exception:
        pass  # Non-Windows or restricted environment — silently skip


def _allow_sleep() -> None:
    """Restore normal Windows sleep behaviour after pipeline finishes."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    except Exception:
        pass

# ── Supported video extensions ────────────────────────────────────────────────
_VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".flv", ".ts", ".m4v", ".wmv", ".mp2t"}
)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _natural_key(path: Path) -> list:
    """
    Natural sort key for episode filenames.
    Ensures ep2 < ep10 (lexicographic sort would give ep10 < ep2).
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    ]


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _derive_paths(drama_name: str, cfg: dict) -> dict:
    """Return a copy of cfg with the paths section overridden for drama_name."""
    cfg = {**cfg, "paths": {**cfg.get("paths", {})}}
    cfg["paths"]["raw_video_dir"] = str(_ROOT / "data" / "raw"    / drama_name)
    cfg["paths"]["cache_dir"]     = str(_ROOT / "data" / "cache"  / drama_name)
    cfg["paths"]["meta_dir"]      = str(_ROOT / "data" / "meta"   / drama_name)
    cfg["paths"]["output_dir"]    = str(_ROOT / "data" / "output" / drama_name)
    return cfg


def _apply_lang_for_mode(cfg: dict, source_language: str) -> None:
    """
    Set pipeline.source_language and update the active mode's ASR/OCR language
    keys in-place.

    For same_lang: both ASR and OCR language follow source_language.
    For cross_lang: source_language records the subtitle language (en) but ASR
    must remain Chinese (zh audio) — so we only update OCR, not ASR.
    The cross_lang.asr section already has no explicit language key, which
    causes ASRRunner to default to "zh" (correct for Chinese audio).
    """
    cfg["pipeline"]["source_language"] = source_language
    mode = cfg["pipeline"]["mode"]
    mode_cfg = cfg.setdefault(mode, {})

    if mode == "cross_lang":
        # OCR always targets the subtitle script (en for cross_lang)
        mode_cfg.setdefault("ocr", {})["language"] = "en"
        # Do NOT override ASR language — ASRRunner defaults to "zh" when absent
    else:
        asr_lang = "zh" if source_language == "zh" else "en"
        ocr_lang = "ch" if source_language == "zh" else "en"
        mode_cfg.setdefault("asr", {})["language"] = asr_lang
        mode_cfg.setdefault("ocr", {})["language"] = ocr_lang


def _discover_dramas(raw_base: Path) -> list[str]:
    """Return sorted list of drama names (immediate subdirectory names)."""
    if not raw_base.exists():
        return []
    return sorted(d.name for d in raw_base.iterdir() if d.is_dir())


def _is_drama_complete(drama_name: str) -> bool:
    """Return True if the drama's checkpoint shows all pipeline stages done."""
    ckpt_path = _ROOT / "data" / "cache" / drama_name / "checkpoint.json"
    if not ckpt_path.exists():
        return False
    try:
        with ckpt_path.open(encoding="utf-8") as f:
            data = json.load(f)
        g = data.get("global", {})
        if not (g.get("map_done") and g.get("reduce_done")):
            return False
        # Verify no new video files were added since last run
        video_list = set(g.get("video_list", []))
        raw_dir = _ROOT / "data" / "raw" / drama_name
        current_videos = {
            v.name for v in raw_dir.iterdir()
            if v.suffix.lower() in _VIDEO_EXTS
        } if raw_dir.exists() else set()
        if current_videos != video_list:
            return False
        # Verify every tracked episode has reached at least the 'refined' state
        episodes: dict = data.get("episodes", {})
        if not episodes:
            return False
        from src.utils.checkpoint import _STATE_RANK
        refined_rank = _STATE_RANK.get("refined", 4)
        for ep_id, state in episodes.items():
            if _STATE_RANK.get(state, 0) < refined_rank:
                return False
        return True
    except (json.JSONDecodeError, OSError, ImportError):
        return False


def _apply_source_language(cfg: dict, source_language: str) -> dict:
    """
    Return a copy of cfg with source_language and all derived language keys
    (ASR language, OCR language) set consistently.
    """
    cfg  = {**cfg}
    cfg["pipeline"] = {**cfg.get("pipeline", {}), "source_language": source_language}
    mode = cfg["pipeline"]["mode"]
    mode_cfg = {**cfg.get(mode, {})}
    mode_cfg["asr"] = {**mode_cfg.get("asr", {})}
    mode_cfg["ocr"] = {**mode_cfg.get("ocr", {})}
    if source_language == "zh":
        mode_cfg["asr"]["language"] = "zh"
        mode_cfg["ocr"]["language"] = "ch"
    else:
        mode_cfg["asr"]["language"] = "en"
        mode_cfg["ocr"]["language"] = "en"
    cfg[mode] = mode_cfg
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  ShortDramaPipeline
# ══════════════════════════════════════════════════════════════════════════════


class ShortDramaPipeline:
    """
    Top-level orchestrator for the short-drama subtitle pipeline.

    Instantiate once per run; call ``run()`` to execute all stages.
    Each stage is guarded by the checkpoint, so the pipeline is safe to
    re-run after partial failures.

    Parameters
    ----------
    cfg:
        Parsed ``config/settings.yaml`` dict.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._mode: str = cfg["pipeline"]["mode"]

        # Directory handles
        self._video_dir  = Path(cfg["paths"]["raw_video_dir"])
        self._cache_dir  = Path(cfg["paths"]["cache_dir"])
        self._meta_dir   = Path(cfg["paths"]["meta_dir"])
        self._output_dir = Path(cfg["paths"]["output_dir"])

        for d in (self._cache_dir, self._meta_dir, self._output_dir):
            d.mkdir(parents=True, exist_ok=True)

        # GPU manager (singleton configured once)
        from src.utils.gpu_manager import GPUManager
        GPUManager.configure(cfg)

        # Checkpoint
        from src.utils.checkpoint import Checkpoint
        self._ckpt = Checkpoint(self._cache_dir / "checkpoint.json")

        # LLM client (shared across MapReduce and Execution stages)
        from src.utils.llm_client import LLMClient
        self._llm = LLMClient(cfg)

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(
        self,
        video_dir: Optional[Path] = None,
        skip_preflight: bool = False,
        yes: bool = False,
    ) -> None:
        """
        Run the full pipeline end-to-end.

        Parameters
        ----------
        video_dir:
            Override the raw_video_dir from config.
        skip_preflight:
            Skip the language pre-flight detection (useful for fast restarts
            when you have already verified the videos manually).
        yes:
            Skip the dry-run confirmation prompt (for CI / scripted runs).
        """
        if video_dir is not None:
            self._video_dir = video_dir

        episodes = self._discover_episodes(self._video_dir)
        if not episodes:
            logger.error(
                "No video files found in %s.  "
                "Supported extensions: %s",
                self._video_dir, ", ".join(sorted(_VIDEO_EXTS)),
            )
            sys.exit(1)

        episode_ids = [ep_id for ep_id, _ in episodes]
        n = len(episodes)

        logger.info(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        logger.info(
            "cc_script_manager  |  mode=%-10s  episodes=%d  video_dir=%s",
            self._mode, n, self._video_dir,
        )
        logger.info(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        self._log_checkpoint_summary(episode_ids)

        # ── Dry-run guard — first pass ────────────────────────────────────
        # On resume: aligned data exists → prints estimate, no prompt.
        # On new show: aligned data not yet available → silent, will re-check
        # below after Stage 2 produces the aligned cache.
        from src.utils.dry_run import DryRunGuard
        _guard = DryRunGuard(self._cfg)
        if not _guard.check_and_confirm(episode_ids, yes=yes):
            sys.exit(0)

        # ── Stage 0: Pre-flight ───────────────────────────────────────────
        if not skip_preflight:
            self._run_preflight()

        # ── Stage 1+2: Ingestion + Alignment ──────────────────────────────
        self._phase_ingestion(episodes)

        # ── ASR language sanity check (fail-fast before any LLM tokens) ───
        self._validate_asr_language(episodes)

        # ── Dry-run guard — second pass (new show only) ───────────────────
        # Aligned cache now exists → print estimate + prompt before Stage 3
        # spends any tokens.  On resume this is a no-op (_is_resume → True).
        if not _guard.check_and_confirm(episode_ids, yes=yes, prompt_on_new=not yes):
            sys.exit(0)

        # ── Stage 3: MapReduce ─────────────────────────────────────────────
        meta = self._phase_map_reduce(episode_ids)

        # ── Stage 4: LLM Refinement ────────────────────────────────────────
        self._phase_execution(episodes, meta)

        # ── Review pause (zh source only) ─────────────────────────────────
        source_lang = self._cfg.get("pipeline", {}).get("source_language", "zh")
        if source_lang == "zh":
            marker = self._meta_dir / "REVIEW_PENDING"
            marker.write_text(
                "CN subtitles are ready for operator review.\n"
                "Edit files in the cn/ folder, then run:\n"
                f"  python pipeline.py \"<drama_name>\" --post-review\n",
                encoding="utf-8",
            )
            logger.info(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            logger.info("⏸  PAUSED — CN subtitles ready for operator review.")
            logger.info("   Folder : %s", self._output_dir / "cn")
            logger.info("   Format : {头衔 姓名}：台词  (name cards, optional)")
            logger.info("   Resume : python pipeline.py \"<drama>\" --post-review")
            logger.info(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            return

        # ── Stage 4.5: Auto-Term Anchoring ────────────────────────────────
        from src.intelligence.term_anchoring import TermAnchoring
        TermAnchoring(self._cfg).build()

        # ── Done ──────────────────────────────────────────────────────────
        logger.info(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        logger.info("Pipeline complete.  SRT files → %s", self._output_dir)
        self._log_checkpoint_summary(episode_ids)

        report = self._output_dir / "validation_report.json"
        if report.exists():
            with report.open(encoding="utf-8") as f:
                vr = json.load(f)
            n_fail = len(vr.get("failures", []))
            if n_fail:
                logger.warning(
                    "%d episode(s) need manual review — see %s",
                    n_fail, report,
                )

        # ── Stage 5: Drama Rhythm Analysis (optional) ─────────────────────
        self._phase_rhythm_analysis(episode_ids, meta)

        # ── Stage 5.5: Translation — must precede creatives so EN SRTs exist
        translation_cfg = self._cfg.get("intelligence", {}).get("translation", {})
        if translation_cfg.get("enabled", False):
            _run_translation_only(self._cfg)

        # ── Stage 6: Creatives (marketing clips) ──────────────────────────
        self._phase_creatives()

        # ── FinOps: emit cost report ───────────────────────────────────────
        from src.intelligence.cost_auditor import CostAuditor
        CostAuditor(self._llm, output_dir=self._output_dir).emit_financial_report(
            self._cfg, cfg_key="llm", report_filename="cost_report_deepseek.json"
        )

    # ------------------------------------------------------------------ #
    #  Episode discovery                                                   #
    # ------------------------------------------------------------------ #

    def _discover_episodes(self, video_dir: Path) -> list[tuple[str, Path]]:
        """
        Return ``[(episode_id, video_path), ...]`` sorted in natural order.

        ``episode_id`` is the file stem (e.g. "ep01", "episode_02", "E03").
        """
        if not video_dir.exists():
            logger.error("Video directory does not exist: %s", video_dir)
            return []

        videos = [
            p for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
        ]
        if not videos:
            return []

        videos.sort(key=_natural_key)
        logger.info(
            "Discovered %d episode(s) in %s (first=%s, last=%s)",
            len(videos), video_dir, videos[0].name, videos[-1].name,
        )
        return [(p.stem, p) for p in videos]

    # ------------------------------------------------------------------ #
    #  Stage 0: Pre-flight                                                 #
    # ------------------------------------------------------------------ #

    def _run_preflight(self) -> None:
        from src.utils.lang_detector import run_preflight
        logger.info("▶ [0/4] Pre-flight — language detection")
        run_preflight(self._cfg, self._video_dir)
        logger.info("   Pre-flight passed")

    # ------------------------------------------------------------------ #
    #  Stage 1+2: Ingestion + Alignment                                    #
    # ------------------------------------------------------------------ #

    def _phase_ingestion(self, episodes: list[tuple[str, Path]]) -> None:
        """
        Run ASR, OCR (+ dedup for cross_lang), and alignment per episode.

        Episodes already at the 'aligned' checkpoint state are skipped.
        If a single episode fails, an error is logged and the pipeline
        continues with the remaining episodes.
        """
        from src.ingestion.asr_runner import ASRRunner
        from src.ingestion.ocr_runner import OCRRunner
        from src.alignment.overlap_aligner import OverlapAligner

        asr     = ASRRunner(self._cfg)
        ocr     = OCRRunner(self._cfg)
        aligner = OverlapAligner(self._cfg)

        if self._mode == "cross_lang":
            from src.ingestion.ocr_dedup import OCRDedup
            dedup: Optional[OCRDedup] = OCRDedup(self._cfg)
        else:
            dedup = None

        todo = [
            (ep_id, vp) for ep_id, vp in episodes
            if not self._ckpt.is_at_least(ep_id, "aligned")
        ]
        n_skip = len(episodes) - len(todo)

        if not todo:
            logger.info(
                "▶ [1/4] Ingestion + Alignment — all %d episode(s) already aligned",
                len(episodes),
            )
            return

        logger.info(
            "▶ [1/4] Ingestion + Alignment — %d episode(s) to process  "
            "(%d already aligned)",
            len(todo), n_skip,
        )

        # Lazy import of tqdm; degrade to a plain iterator if not installed
        try:
            from tqdm import tqdm as _tqdm
            bar = _tqdm(
                todo,
                desc="Ingestion+Align",
                unit="ep",
                dynamic_ncols=True,
                leave=True,
            )
        except ImportError:
            bar = todo  # type: ignore[assignment]

        n_ok = n_err = 0
        for ep_id, video_path in bar:
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(ep_id, refresh=False)
            try:
                self._ingest_one(ep_id, video_path, asr, ocr, dedup, aligner)
                n_ok += 1
            except Exception as exc:
                n_err += 1
                _tqdm_write(f"ERROR [{ep_id}]: {exc}")
                logger.exception("Ingestion failed for [%s] — episode skipped", ep_id)

        if hasattr(bar, "close"):
            bar.close()

        logger.info(
            "   Ingestion+Alignment done — ok=%d  errors=%d", n_ok, n_err
        )

    def _ingest_one(
        self,
        ep_id: str,
        video_path: Path,
        asr: "ASRRunner",
        ocr: "OCRRunner",
        dedup: "Optional[OCRDedup]",
        aligner: "OverlapAligner",
    ) -> None:
        # ── ASR ───────────────────────────────────────────────────────────
        if not self._ckpt.is_at_least(ep_id, "asr_done"):
            asr.run_episode(video_path, ep_id)
            self._ckpt.set_state(ep_id, "asr_done")

        # ── OCR (+ dedup for cross_lang) ──────────────────────────────────
        if not self._ckpt.is_at_least(ep_id, "ocr_done"):
            ocr_path = ocr.run_episode(video_path, ep_id)
            if dedup is not None:
                # OCR dedup collapses raw frames into subtitle timeline
                dedup.run_episode(ocr_path, ep_id)
            self._ckpt.set_state(ep_id, "ocr_done")

        # ── Alignment ─────────────────────────────────────────────────────
        if not self._ckpt.is_at_least(ep_id, "aligned"):
            aligner.run_episode(ep_id)
            self._ckpt.set_state(ep_id, "aligned")

    # ------------------------------------------------------------------ #
    #  ASR language validation (between Stage 2 and Stage 3)              #
    # ------------------------------------------------------------------ #

    def _validate_asr_language(self, episodes: list[tuple[str, Path]]) -> None:
        """
        Fail-fast guard: verify ASR output language matches source_language
        before any LLM tokens are spent.

        Whisper produces fluent-sounding hallucinated English when forced with
        language="en" on Chinese audio — CJK ratio of the transcript exposes
        this mismatch immediately. Aborts with a clear, actionable error
        message instead of silently burning the entire refinement budget.
        """
        from src.utils.lang_detector import cjk_ratio as _cjk_ratio

        source_language = self._cfg["pipeline"].get("source_language", "zh")
        asr_dir = self._cache_dir / "asr"

        # Collect sample text from the first 3 episodes for robustness
        sample_text = ""
        for ep_id, _ in episodes[:3]:
            asr_path = asr_dir / f"{ep_id}_asr.json"
            if not asr_path.exists():
                continue
            try:
                data = json.loads(asr_path.read_text(encoding="utf-8"))
                segs = data.get("segments", [])
                sample_text += " ".join(s.get("text", "") for s in segs)
                if len(sample_text.strip()) >= 100:
                    break
            except (json.JSONDecodeError, OSError):
                continue

        if len(sample_text.strip()) < 20:
            logger.warning(
                "ASR language validation skipped — not enough sample text "
                "(ASR may have produced 0 segments)"
            )
            return

        ratio = _cjk_ratio(sample_text)
        logger.info(
            "ASR language validation — source_language=%s  cjk_ratio=%.3f  "
            "sample_len=%d chars",
            source_language, ratio, len(sample_text),
        )

        if source_language == "zh" and ratio < 0.20:
            raise RuntimeError(
                f"\n"
                f"  ╔══ ASR LANGUAGE MISMATCH — ABORTING BEFORE LLM STAGE ══╗\n"
                f"  ║  Configured: source_language='zh'                      ║\n"
                f"  ║  Detected:   CJK ratio={ratio:.1%} (expected >20%)        ║\n"
                f"  ║                                                         ║\n"
                f"  ║  Whisper produced English-looking text from Chinese     ║\n"
                f"  ║  audio — language= was likely set to 'en' when the      ║\n"
                f"  ║  ASR cache was written.                                 ║\n"
                f"  ║                                                         ║\n"
                f"  ║  Fix: delete data/cache/<drama>/asr/ and re-run.        ║\n"
                f"  ╚═════════════════════════════════════════════════════════╝"
            )

        if source_language == "en" and ratio > 0.60:
            raise RuntimeError(
                f"\n"
                f"  ╔══ ASR LANGUAGE MISMATCH — ABORTING BEFORE LLM STAGE ══╗\n"
                f"  ║  Configured: source_language='en'                      ║\n"
                f"  ║  Detected:   CJK ratio={ratio:.1%} (expected <60%)        ║\n"
                f"  ║                                                         ║\n"
                f"  ║  ASR output is predominantly Chinese — check pipeline   ║\n"
                f"  ║  mode or audio language detection result.               ║\n"
                f"  ║                                                         ║\n"
                f"  ║  Fix: delete data/cache/<drama>/asr/ and re-run.        ║\n"
                f"  ╚═════════════════════════════════════════════════════════╝"
            )

    # ------------------------------------------------------------------ #
    #  Stage 3: MapReduce                                                  #
    # ------------------------------------------------------------------ #

    def _phase_map_reduce(self, episode_ids: list[str]) -> dict:
        """
        Run MapReduce entity extraction and return the parsed ``meta.json``.

        Both MapPhase and ReducePhase have their own file-based checkpoints
        (batch files and meta_raw.json) so they skip themselves on re-runs.
        """
        logger.info("▶ [2/4] MapReduce — entity extraction + meta.json")

        from src.metadata.map_phase import MapPhase
        from src.metadata.reduce_phase import ReducePhase

        map_phase = MapPhase(self._cfg, llm_client=self._llm)
        batch_paths = map_phase.run(episode_ids)
        self._ckpt.set_global("map_done", True)

        reduce_phase = ReducePhase(self._cfg, llm_client=self._llm)
        reduce_phase.run(batch_paths)
        self._ckpt.set_global("reduce_done", True)

        meta_path = self._meta_dir / "meta.json"
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            n_chars = len(meta.get("characters", {}))
            logger.info(
                "   meta.json loaded — %d character(s)%s",
                n_chars,
                "  ← edit meta.json to override canonical names before re-running"
                if n_chars > 0 else "",
            )
            return meta

        logger.warning(
            "   meta.json not found at %s — proceeding without character reference",
            meta_path,
        )
        return {"characters": {}}

    # ------------------------------------------------------------------ #
    #  Stage 4: LLM Refinement                                            #
    # ------------------------------------------------------------------ #

    def _phase_execution(
        self,
        episodes: list[tuple[str, Path]],
        meta: dict,
    ) -> None:
        """
        Refine each episode's aligned JSON into a polished .srt via the LLM.

        Episodes already at the 'refined' checkpoint state are skipped.
        Per-episode SRT files also act as their own checkpoint (EpisodeRefiner
        returns immediately if the .srt already exists).
        """
        from src.execution.episode_refiner import EpisodeRefiner

        refiner      = EpisodeRefiner(self._llm, self._cfg, meta)
        aligned_dir  = self._cache_dir / "aligned"

        todo = [
            ep_id for ep_id, _ in episodes
            if not self._ckpt.is_at_least(ep_id, "refined")
        ]
        n_skip = len(episodes) - len(todo)

        if not todo:
            logger.info(
                "▶ [3/4] Refinement — all %d episode(s) already refined",
                len(episodes),
            )
            return

        logger.info(
            "▶ [3/4] LLM Refinement — %d episode(s) to refine  "
            "(%d already done)",
            len(todo), n_skip,
        )

        try:
            from tqdm import tqdm as _tqdm
            bar = _tqdm(
                todo,
                desc="LLM Refinement",
                unit="ep",
                dynamic_ncols=True,
                leave=True,
            )
        except ImportError:
            bar = todo  # type: ignore[assignment]

        n_ok = n_err = 0
        for ep_id in bar:
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(ep_id, refresh=False)

            aligned_path = aligned_dir / f"{ep_id}_aligned.json"
            if not aligned_path.exists():
                _tqdm_write(f"WARNING [{ep_id}]: aligned JSON missing — skipping")
                logger.warning(
                    "Aligned JSON not found for [%s] — skipping refinement", ep_id
                )
                n_err += 1
                continue

            try:
                refiner.refine_episode(str(aligned_path))
                self._ckpt.set_state(ep_id, "refined")
                self._ckpt.set_state(ep_id, "complete")
                n_ok += 1
            except Exception as exc:
                n_err += 1
                _tqdm_write(f"ERROR [{ep_id}]: {exc}")
                logger.exception("Refinement failed for [%s]", ep_id)

        if hasattr(bar, "close"):
            bar.close()

        logger.info("   Refinement done — ok=%d  errors=%d", n_ok, n_err)

    # ------------------------------------------------------------------ #
    #  Stage 5: Drama Rhythm Analysis                                      #
    # ------------------------------------------------------------------ #

    def _phase_rhythm_analysis(self, episode_ids: list[str], meta: dict) -> None:
        """
        Run the two-phase drama rhythm analysis when enabled in config.

        Guarded by ``intelligence.drama_analysis.enabled``; silently skips
        when disabled (default: false) so existing pipelines are unaffected.
        """
        enabled = (
            self._cfg.get("intelligence", {})
            .get("drama_analysis", {})
            .get("enabled", False)
        )
        if not enabled:
            logger.info(
                "▶ [5/5] Drama Rhythm Analysis — disabled "
                "(set intelligence.drama_analysis.enabled: true to activate)"
            )
            return

        logger.info("▶ [5/5] Drama Rhythm Analysis — starting")

        from src.intelligence.rhythm_analyzer import RhythmAnalyzer
        analyzer = RhythmAnalyzer(
            cfg=self._cfg,
            llm_client=self._llm,
            output_dir=self._output_dir,
            cache_dir=self._cache_dir,
            meta=meta,
        )
        try:
            result = analyzer.run(episode_ids)
            n_eps = result.get("total_episodes_analysed", 0)
            logger.info(
                "   Drama Rhythm Analysis done — %d episode(s) analysed  "
                "→ drama_structure_graph.json",
                n_eps,
            )
        except Exception as exc:
            logger.error(
                "   Drama Rhythm Analysis failed — pipeline continues without it.  "
                "Error: %s", exc,
            )

    # ------------------------------------------------------------------ #
    #  Stage 6: Creatives (marketing clip generation)                      #
    # ------------------------------------------------------------------ #

    def _phase_creatives(self) -> None:
        """
        Run extract → plan → assemble when intelligence.creatives.enabled is true.

        Prerequisite: drama_structure_graph.json must exist (Stage 5 output).
        Each step has an idempotency guard so re-runs are safe.
        """
        enabled = (
            self._cfg.get("intelligence", {})
            .get("creatives", {})
            .get("enabled", False)
        )
        if not enabled:
            logger.info(
                "▶ [6/6] Creatives — disabled "
                "(set intelligence.creatives.enabled: true to activate)"
            )
            return

        drama_name = self._output_dir.name
        graph_path = self._output_dir / "drama_structure_graph.json"
        if not graph_path.exists():
            logger.warning(
                "▶ [6/6] Creatives — skipped: drama_structure_graph.json not found "
                "(Stage 5 must complete successfully first)"
            )
            return

        logger.info("▶ [6/6] Creatives — starting  drama=%s", drama_name)

        from scripts.extract_clips     import main as _extract
        from scripts.plan_clips        import main as _plan
        from scripts.process_creatives import main as _assemble
        from scripts.write_copy        import main as _write_copy

        # Step 1: extract anchor clips (idempotent ffmpeg, fast)
        logger.info("   [6/6] Step 1/3 — extracting anchor clips")
        rc = _extract(drama_name)
        if rc != 0:
            logger.error("   [6/6] extract_clips failed (rc=%d) — skipping plan+assemble", rc)
            return

        # Step 2: LLM clip planning (skip if plans already exist)
        plans_path = self._output_dir / "clip_plans.json"
        if plans_path.exists():
            logger.info("   [6/6] Step 2/3 — clip_plans.json already exists, skipping LLM planning")
        else:
            logger.info("   [6/6] Step 2/3 — running LLM clip planner")
            rc = _plan(drama_name)
            if rc != 0:
                logger.error("   [6/6] plan_clips failed (rc=%d) — skipping assemble", rc)
                return

        # Step 3: assemble ext_*.mp4 (skip only if the plan is fully assembled)
        creatives_dir = self._output_dir / "creatives"
        try:
            import json as _json
            _plans = _json.loads(plans_path.read_text(encoding="utf-8"))
            _expected = len(_plans.get("clip_plans", []))
        except Exception:
            _expected = 0
        existing = list(creatives_dir.glob("ext_*.mp4"))
        if _expected > 0 and len(existing) >= _expected:
            logger.info(
                "   [6/6] Step 3/3 — %d/%d assembled clip(s) already exist, skipping",
                len(existing), _expected,
            )
        else:
            logger.info("   [6/6] Step 3/3 — assembling final marketing clips")
            rc = _assemble(drama_name, None)
            if rc != 0:
                logger.error("   [6/6] process_creatives failed (rc=%d)", rc)
                return

        # Step 4: Meta ad copy (skip if CSV already exists)
        csv_path = self._output_dir / "creatives" / "ad_copy.csv"
        if csv_path.exists():
            logger.info("   [6/6] Step 4/4 — ad_copy.csv already exists, skipping")
        else:
            logger.info("   [6/6] Step 4/4 — generating Meta ad copy")
            rc = _write_copy(drama_name)
            if rc != 0:
                logger.warning("   [6/6] write_copy returned rc=%d — CSV may be incomplete", rc)

        logger.info("   [6/6] Creatives done → %s/creatives/", self._output_dir)

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _log_checkpoint_summary(self, episode_ids: list[str]) -> None:
        """Log a one-line progress snapshot from the checkpoint."""
        summary = self._ckpt.summary()
        parts = [
            f"{state}={count}"
            for state, count in summary.items()
            if count > 0
        ]
        logger.info(
            "Checkpoint snapshot: %s  (total=%d)",
            "  ".join(parts) if parts else "all-pending",
            len(episode_ids),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  tqdm compat helper
# ══════════════════════════════════════════════════════════════════════════════


def _tqdm_write(msg: str) -> None:
    """Write *msg* to stderr without disrupting a live tqdm bar."""
    try:
        from tqdm import tqdm
        tqdm.write(msg, file=sys.stderr)
    except ImportError:
        print(msg, file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python pipeline.py",
        description="cc_script_manager — Short Drama Subtitle Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single drama by name (videos in data/raw/<drama_name>/):
  python pipeline.py "dollar baby"

  # Process ALL dramas found in data/raw/ sequentially:
  python pipeline.py --all

  # Backward compat — reads paths.raw_video_dir from settings.yaml:
  python pipeline.py

  # Switch to cross_lang without editing settings.yaml:
  python pipeline.py "dollar baby" --mode cross_lang

  # Resume after a crash (checkpoint auto-skips completed episodes):
  python pipeline.py "dollar baby"

  # Skip lang detection (already verified, want a faster restart):
  python pipeline.py "dollar baby" --skip-preflight

  # Skip the cost-estimate confirmation prompt (CI / batch runs):
  python pipeline.py "dollar baby" --yes

  # Translate already-refined episodes (no GPU required):
  python pipeline.py "dollar baby" --translate-only
  python pipeline.py "dollar baby" --translate-only --episodes 01,02,05
        """,
    )
    parser.add_argument(
        "drama_name",
        nargs="?",
        default=None,
        metavar="DRAMA_NAME",
        help=(
            "Name of the drama to process — must match a subdirectory under "
            "data/raw/.  All cache/meta/output paths are derived automatically. "
            "Omit to fall back to paths.raw_video_dir in settings.yaml."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every drama directory found under data/raw/ sequentially.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        metavar="PATH",
        help="Path to settings.yaml  (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        metavar="DIR",
        help="Override paths.raw_video_dir from config",
    )
    parser.add_argument(
        "--mode",
        choices=["same_lang", "cross_lang"],
        default=None,
        help="Override pipeline.mode from config for this run",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the language pre-flight detection",
    )
    parser.add_argument(
        "--translate-only",
        action="store_true",
        help=(
            "Run only the TranslationMatrix layer on already-refined episodes. "
            "No GPU required — reads aligned cache and calls LLM APIs only."
        ),
    )
    parser.add_argument(
        "--episodes",
        default=None,
        metavar="IDs",
        help="Comma-separated episode IDs to process (e.g. 01,02,05). "
             "Used with --translate-only to limit scope.",
    )
    parser.add_argument(
        "--paywall-report",
        action="store_true",
        help=(
            "Generate an AI-driven paywall & marketing strategy report from "
            "drama_structure_graph.json. No GPU required — calls Claude only. "
            "Output: data/output/paywall_strategy_report.md"
        ),
    )
    parser.add_argument(
        "--post-review",
        action="store_true",
        help=(
            "Resume pipeline after operator CN subtitle review. "
            "Runs Stage 4.5 (term anchoring) → 5 (analysis) → 5.5 (translation) → 6 (creatives). "
            "Reads corrected cn/ SRTs as translation source. "
            "Only valid for same_lang (zh) dramas."
        ),
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the dry-run confirmation prompt (for CI / scripted runs)",
    )
    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  Translation-only entry point
# ══════════════════════════════════════════════════════════════════════════════


def _run_translation_only(cfg: dict, episode_filter: Optional[list[str]] = None) -> None:
    """
    Run the TranslationMatrix layer on all already-refined episodes.

    Does not touch ASR/OCR/alignment or the main checkpoint state machine.
    Requires data/meta/meta.json to exist (produced by the MapReduce stage).
    """
    from src.utils.checkpoint import Checkpoint
    from src.intelligence.translation_matrix import TranslationMatrix
    from src.intelligence.cost_auditor import CostAuditor

    cache_dir  = Path(cfg["paths"]["cache_dir"])
    meta_dir   = Path(cfg["paths"]["meta_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])

    # Require meta.json — character map is needed for name constraints
    meta_path = meta_dir / "meta.json"
    if not meta_path.exists():
        logger.error(
            "meta.json not found at %s — run the full pipeline first to generate "
            "character metadata, then re-run with --translate-only",
            meta_path,
        )
        sys.exit(1)

    # Discover refinable episodes from aligned cache directory
    aligned_dir = cache_dir / "aligned"
    if not aligned_dir.exists():
        logger.error(
            "Aligned cache directory not found: %s — run the full pipeline first",
            aligned_dir,
        )
        sys.exit(1)

    ckpt = Checkpoint(cache_dir / "checkpoint.json")

    all_aligned = sorted(
        aligned_dir.glob("*_aligned.json"),
        key=lambda p: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", p.stem)
        ],
    )
    episode_ids = [p.stem.replace("_aligned", "") for p in all_aligned]

    # Filter to refined/complete episodes only
    episode_ids = [ep for ep in episode_ids if ckpt.is_at_least(ep, "refined")]

    # Apply --episodes filter if provided
    if episode_filter:
        episode_ids = [ep for ep in episode_ids if ep in episode_filter]
        missing = [ep for ep in episode_filter if ep not in episode_ids]
        if missing:
            logger.warning(
                "--episodes filter: %s not found in refined episodes — skipping those",
                missing,
            )

    if not episode_ids:
        logger.warning(
            "No refined episodes found. Run the main pipeline to refine episodes first."
        )
        return

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    target_langs = cfg.get("intelligence", {}).get("translation", {}).get("target_languages", [])
    logger.info(
        "TranslationMatrix  |  episodes=%d  languages=en+%s",
        len(episode_ids), target_langs,
    )
    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # Stage 4.5: build en_terms.json if not yet present (idempotent)
    from src.intelligence.term_anchoring import TermAnchoring
    TermAnchoring(cfg).build()

    matrix = TranslationMatrix(cfg)
    matrix.run_all(episode_ids)

    # FinOps: translation costs go to separate files so they never overwrite
    # the main pipeline's cost_report_deepseek.json (Stage 3+4 costs).
    logger.info("─── FinOps: DeepSeek (translation skeleton + minor langs) ───")
    CostAuditor(matrix.llm_ds, output_dir=output_dir).emit_financial_report(
        cfg, cfg_key="llm", report_filename="cost_report_deepseek_translation.json"
    )

    if matrix.llm_claude is not matrix.llm_ds:
        logger.info("─── FinOps: Claude (English refinement) ───")
        CostAuditor(matrix.llm_claude, output_dir=output_dir).emit_financial_report(
            cfg, cfg_key="llm_claude", report_filename="cost_report_claude.json"
        )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    logger.info(
        "Translation complete.  SRT files → %s",
        output_dir / "translations",
    )


def _run_post_review_only(cfg: dict) -> None:
    """
    Resume pipeline after operator CN subtitle review (zh source dramas only).

    Clears REVIEW_PENDING marker, then runs:
      Stage 4.5  — Auto-Term Anchoring (builds en_terms.json from reviewed cn/ SRTs)
      Stage 5    — Drama Rhythm Analysis
      Stage 5.5  — Translation (reads corrected cn/ SRTs as source)
      Stage 6    — Creatives
    """
    from src.utils.checkpoint import Checkpoint
    from src.intelligence.term_anchoring import TermAnchoring
    from src.intelligence.cost_auditor import CostAuditor

    cache_dir  = Path(cfg["paths"]["cache_dir"])
    meta_dir   = Path(cfg["paths"]["meta_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])

    # Clear review marker
    marker = meta_dir / "REVIEW_PENDING"
    if marker.exists():
        marker.unlink()
        logger.info("REVIEW_PENDING cleared — resuming post-review stages")

    # Discover refined episode IDs
    aligned_dir = cache_dir / "aligned"
    if not aligned_dir.exists():
        logger.error("Aligned cache not found at %s — run the full pipeline first", aligned_dir)
        sys.exit(1)

    ckpt = Checkpoint(cache_dir / "checkpoint.json")
    all_aligned = sorted(
        aligned_dir.glob("*_aligned.json"),
        key=lambda p: [
            int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", p.stem)
        ],
    )
    episode_ids = [p.stem.replace("_aligned", "") for p in all_aligned]
    episode_ids = [ep for ep in episode_ids if ckpt.is_at_least(ep, "refined")]

    if not episode_ids:
        logger.error("No refined episodes found — run the main pipeline first")
        sys.exit(1)

    meta_path = meta_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    logger.info("▶ Post-review pipeline — %d episode(s)", len(episode_ids))
    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # Stage 4.5 — Term Anchoring (re-reads reviewed cn/ SRTs for sampling)
    TermAnchoring(cfg).build()

    # Stage 5 — Drama Rhythm Analysis + Stage 6 via pipeline instance
    pipeline = ShortDramaPipeline(cfg)
    pipeline._phase_rhythm_analysis(episode_ids, meta)

    # Stage 5.5 — Translation (reads corrected cn/ SRTs via _load_aligned_segs)
    _run_translation_only(cfg)

    # Stage 6 — Creatives
    pipeline._phase_creatives()

    # FinOps
    from src.intelligence.translation_matrix import TranslationMatrix
    from src.intelligence.cost_auditor import CostAuditor as CA
    matrix_tmp = TranslationMatrix.__new__(TranslationMatrix)
    logger.info("Post-review stages complete.  Outputs → %s", output_dir)


def _run_paywall_report(cfg: dict) -> None:
    """
    Generate the AI paywall strategy report from drama_structure_graph.json.

    Requires drama_structure_graph.json to exist (produced by Stage 5 drama
    analysis).  No GPU required — calls Claude via OpenRouter only.
    """
    from src.intelligence.paywall_strategist import PaywallStrategist
    from src.intelligence.cost_auditor import CostAuditor

    output_dir = Path(cfg["paths"]["output_dir"])

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    logger.info("PaywallStrategist  |  reading drama_structure_graph.json")
    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    strategist = PaywallStrategist(cfg)
    try:
        out_path = strategist.run()
    except FileNotFoundError as exc:
        logger.error("Paywall report failed: %s", exc)
        sys.exit(1)

    logger.info("─── FinOps: Claude (paywall strategy) ───")
    CostAuditor(strategist.llm_claude, output_dir=output_dir).emit_financial_report(
        cfg, cfg_key="llm_claude", report_filename="cost_report_paywall.json"
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    logger.info("Paywall report complete → %s", out_path)


def _run_single(
    drama_name: str,
    base_cfg: dict,
    args: argparse.Namespace,
    cfg_path: Path,
) -> None:
    """Run the full pipeline (or a sub-mode) for a single drama."""
    cfg = _derive_paths(drama_name, base_cfg)

    # ── Auto-detect pipeline mode and source_language from video content ────
    # When --mode is given, we trust the user and only auto-detect language.
    # When --mode is absent, we also auto-detect mode (same_lang vs cross_lang)
    # by sampling both subtitle OCR and a 30-second audio clip.
    from src.utils.lang_detector import detect_language, detect_pipeline_mode
    video_dir_detect = Path(cfg["paths"]["raw_video_dir"])

    if args.mode:
        cfg["pipeline"]["mode"] = args.mode
        logger.info("Mode overridden via --mode: %s", args.mode)
        detected_lang = detect_language(video_dir_detect, cfg)
        logger.info(
            "Auto-detected source_language='%s' for '%s'",
            detected_lang, drama_name,
        )
        _apply_lang_for_mode(cfg, detected_lang)
    else:
        detected_mode, detected_lang = detect_pipeline_mode(video_dir_detect, cfg)
        configured_mode = cfg["pipeline"].get("mode", "same_lang")
        configured_lang = cfg["pipeline"].get("source_language", "zh")
        if detected_mode != configured_mode:
            logger.info(
                "Auto-detected mode='%s' for '%s' (settings.yaml had '%s') — overriding",
                detected_mode, drama_name, configured_mode,
            )
        if detected_lang != configured_lang:
            logger.info(
                "Auto-detected source_language='%s' for '%s' (settings.yaml had '%s') — overriding",
                detected_lang, drama_name, configured_lang,
            )
        cfg["pipeline"]["mode"] = detected_mode
        _apply_lang_for_mode(cfg, detected_lang)

    # ── Paywall report shortcut ───────────────────────────────────────────
    if args.paywall_report:
        _run_paywall_report(cfg)
        return

    # ── Post-review shortcut (zh dramas only) ────────────────────────────
    if args.post_review:
        _run_post_review_only(cfg)
        return

    # ── Translation-only shortcut ─────────────────────────────────────────
    if args.translate_only:
        episode_filter = (
            [ep.strip() for ep in args.episodes.split(",") if ep.strip()]
            if args.episodes else None
        )
        _run_translation_only(cfg, episode_filter=episode_filter)
        return

    # ── Auto-detect show change and self-heal ROI ─────────────────────────
    from src.utils.project_initializer import ProjectInitializer
    ProjectInitializer(cfg).auto_detect_and_heal()
    # Reload settings.yaml in case the ROI was just patched by the initializer,
    # then re-derive drama paths (yaml only stores non-path settings like ROI).
    # Re-apply the auto-detected mode/language so the reload doesn't lose them.
    resolved_mode = cfg["pipeline"]["mode"]
    resolved_lang = cfg["pipeline"]["source_language"]
    base_cfg_reloaded = load_config(cfg_path)
    cfg = _derive_paths(drama_name, base_cfg_reloaded)
    cfg["pipeline"]["mode"] = resolved_mode
    _apply_lang_for_mode(cfg, resolved_lang)

    video_dir = Path(args.video_dir) if args.video_dir else None

    pipeline = ShortDramaPipeline(cfg)
    pipeline.run(video_dir=video_dir, skip_preflight=args.skip_preflight, yes=args.yes)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    base_cfg = load_config(cfg_path)

    # ── Resolve drama_name and dispatch ──────────────────────────────────────
    if args.all:
        raw_base = _ROOT / "data" / "raw"
        dramas = _discover_dramas(raw_base)
        if not dramas:
            logger.error("No drama directories found in %s", raw_base)
            sys.exit(1)
        logger.info("--all: found %d drama(s): %s", len(dramas), dramas)
        pending = [d for d in dramas if not _is_drama_complete(d)]
        skipped = [d for d in dramas if _is_drama_complete(d)]
        if skipped:
            logger.info("--all: skipping %d already-complete drama(s): %s", len(skipped), skipped)
        if not pending:
            logger.info("--all: all dramas already complete — nothing to do")
            return
        logger.info("--all: %d drama(s) to process: %s", len(pending), pending)
        # Batch mode is always unattended — suppress the per-drama dry-run prompt.
        args.yes = True
        for drama_name in pending:
            logger.info(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            logger.info("Drama: %s", drama_name)
            _run_single(drama_name, base_cfg, args, cfg_path)
        return

    if args.drama_name:
        drama_name = args.drama_name
    elif args.video_dir:
        # --video-dir given but no positional: derive drama name from folder name
        drama_name = Path(args.video_dir).name
    else:
        # Backward compat: read from settings.yaml
        drama_name = Path(base_cfg["paths"]["raw_video_dir"]).name

    _run_single(drama_name, base_cfg, args, cfg_path)


if __name__ == "__main__":
    _prevent_sleep()
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted (Ctrl+C) — checkpoint preserved, re-run to resume")
        _allow_sleep()
        sys.exit(0)
    except SystemExit:
        _allow_sleep()
        raise
    except Exception:
        logger.exception("Unhandled exception — pipeline aborted")
        _allow_sleep()
        sys.exit(1)
    else:
        _allow_sleep()
