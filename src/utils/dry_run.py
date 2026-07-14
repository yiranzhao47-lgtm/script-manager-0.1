"""
Dry-run guard — pre-flight cost estimator with optional user confirmation.

pipeline.py calls check_and_confirm() twice on the same guard instance:
  Pass 1  — before Stage 0, immediately at startup
  Pass 2  — after Stage 1+2, before Stage 3 (new-show path only)

Behavior
────────
  Checkpoint resume (≥1 episode past "pending")
    Pass 1: aligned cache exists → print budget table, no prompt.
    Pass 2: estimate already printed → silent no-op.
  New show — no prior progress
    Pass 1: no aligned cache yet → silent (cannot estimate).
    Pass 2: aligned cache produced by Stage 2 → print table → prompt "Proceed? [Y/n]"
  --yes flag
    Skip prompt regardless of checkpoint state; estimate table still printed.

Usage
─────
    from src.utils.dry_run import DryRunGuard

    guard = DryRunGuard(cfg)
    # Pass 1 — at startup
    if not guard.check_and_confirm(episode_ids, yes=args.yes):
        sys.exit(0)
    # ... Stage 1+2 ...
    # Pass 2 — after ingestion (new show only; resume is a no-op)
    if not guard.check_and_confirm(episode_ids, yes=args.yes, prompt_on_new=not args.yes):
        sys.exit(0)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class DryRunGuard:
    """
    Pre-flight cost estimator with user confirmation gate.

    Parameters
    ----------
    cfg:
        Parsed config/settings.yaml dict (same object used by the pipeline).
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        cache_dir = Path(cfg["paths"]["cache_dir"])
        self._ckpt_path = cache_dir / "checkpoint.json"
        self._estimate_shown = False  # avoid double-printing on two-pass call

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def check_and_confirm(
        self,
        episode_ids: list[str],
        yes: bool = False,
        prompt_on_new: bool = True,
    ) -> bool:
        """
        Return True if the pipeline should proceed, False if the user aborted.

        Parameters
        ----------
        episode_ids:
            Full list of episode IDs discovered for this run.
        yes:
            If True, skip the interactive prompt (for CI / scripted runs).
        prompt_on_new:
            If False, show the estimate but skip the Y/n prompt even for new
            shows.  Used by the post-ingestion call so the estimate is always
            printed but never double-prompts.
        """
        is_resume = self._is_resume(episode_ids)

        # Always attempt token estimation — prints the table when data exists.
        estimate = self._run_estimate()

        if is_resume:
            if estimate:
                logger.debug("DryRunGuard: resume — token estimate shown above (no prompt)")
            else:
                logger.debug("DryRunGuard: resume — no aligned cache yet, estimate unavailable")
            return True

        # New show path
        if not estimate:
            # Aligned data not yet available — will re-check after Stage 2
            logger.debug(
                "DryRunGuard: no aligned cache found — "
                "estimate will appear after Stage 2 completes"
            )
            return True

        if yes or not prompt_on_new:
            if yes:
                logger.info("DryRunGuard: --yes flag set — skipping confirmation prompt")
            return True

        return self._prompt_user()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _is_resume(self, episode_ids: list[str]) -> bool:
        """Return True if any episode in this run has progressed past 'pending'."""
        if not self._ckpt_path.exists():
            return False

        try:
            with self._ckpt_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        ep_states = data.get("episodes", {})
        return any(
            ep_states.get(ep_id, "pending") != "pending"
            for ep_id in episode_ids
        )

    def _run_estimate(self) -> dict:
        """Run TokenCounter; return empty dict on failure or no data.
        Skips silently if the estimate table has already been printed this run."""
        if self._estimate_shown:
            return {}
        try:
            from src.utils.token_counter import TokenCounter
            result = TokenCounter(self._cfg).estimate()
            if result:
                self._estimate_shown = True
            return result
        except Exception as exc:
            logger.debug("DryRunGuard: TokenCounter raised %s — skipping estimate", exc)
            return {}

    @staticmethod
    def _prompt_user() -> bool:
        """
        Prompt for user confirmation.

        Default (empty input / Enter) means Yes.
        Returns False only on explicit 'n' / 'no' or Ctrl+C / EOF.
        """
        try:
            answer = input("\nProceed? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            return False

        if answer in ("n", "no"):
            print("Aborted by user.", file=sys.stderr)
            return False

        # "", "y", "yes", or anything else → proceed
        return True
