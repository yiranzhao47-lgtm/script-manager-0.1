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

        # ── Stage 0: Pre-flight ───────────────────────────────────────────
        if not skip_preflight:
            self._run_preflight()

        # ── Stage 1+2: Ingestion + Alignment ──────────────────────────────
        self._phase_ingestion(episodes)

        # ── Stage 3: MapReduce ─────────────────────────────────────────────
        meta = self._phase_map_reduce(episode_ids)

        # ── Stage 4: LLM Refinement ────────────────────────────────────────
        self._phase_execution(episodes, meta)

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

        # ── FinOps: emit cost report ───────────────────────────────────────
        from src.intelligence.cost_auditor import CostAuditor
        CostAuditor(self._llm, output_dir=self._output_dir).emit_financial_report(self._cfg)

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
  # First run — same_lang mode, videos in data/raw/:
  python pipeline.py

  # Switch to cross_lang (English subs) without editing settings.yaml:
  python pipeline.py --mode cross_lang

  # Resume after a crash (checkpoint auto-skips completed episodes):
  python pipeline.py

  # Point to a different video folder:
  python pipeline.py --video-dir /mnt/nas/drama_ep01-80

  # Skip lang detection (already verified, want a faster restart):
  python pipeline.py --skip-preflight

  # Use a custom config:
  python pipeline.py --config config/my_project.yaml

  # Translate already-refined episodes (no GPU required):
  python pipeline.py --translate-only
  python pipeline.py --translate-only --episodes 01,02,05
        """,
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

    matrix = TranslationMatrix(cfg)
    matrix.run_all(episode_ids)

    # FinOps: report costs for both LLM clients
    logger.info("─── FinOps: DeepSeek (skeleton + minor langs) ───")
    CostAuditor(matrix.llm_ds, output_dir=output_dir).emit_financial_report(cfg)

    if matrix.llm_claude is not matrix.llm_ds:
        logger.info("─── FinOps: Claude (English refinement) ───")
        CostAuditor(matrix.llm_claude, output_dir=output_dir).emit_financial_report(cfg)

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    logger.info(
        "Translation complete.  SRT files → %s",
        output_dir / "translations",
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    cfg = load_config(cfg_path)

    # Apply CLI overrides
    if args.mode:
        cfg["pipeline"]["mode"] = args.mode
        logger.info("Mode overridden via --mode: %s", args.mode)

    # ── Translation-only shortcut (no GPU, no preflight, no main pipeline) ───
    if args.translate_only:
        episode_filter = (
            [ep.strip() for ep in args.episodes.split(",") if ep.strip()]
            if args.episodes else None
        )
        _run_translation_only(cfg, episode_filter=episode_filter)
        return

    # ── Auto-detect show change and self-heal ROI (must run before any stage) ──
    from src.utils.project_initializer import ProjectInitializer
    ProjectInitializer(cfg).auto_detect_and_heal()
    # Reload settings.yaml in case the ROI was just patched by the initializer
    cfg = load_config(cfg_path)
    if args.mode:
        cfg["pipeline"]["mode"] = args.mode

    video_dir = Path(args.video_dir) if args.video_dir else None

    pipeline = ShortDramaPipeline(cfg)
    pipeline.run(video_dir=video_dir, skip_preflight=args.skip_preflight)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted (Ctrl+C) — checkpoint preserved, re-run to resume")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        logger.exception("Unhandled exception — pipeline aborted")
        sys.exit(1)
