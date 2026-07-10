"""
Dry-run guard — Form B interaction.

Triggered automatically on a fresh run (no checkpoint progress for the
current episode list).  Silent on checkpoint resume or when --yes is passed.

Behavior
────────
  New show (no checkpoint progress)
    → Run TokenCounter, print budget table, prompt "Proceed? [Y/n]"
    → If aligned cache is empty (Stages 1+2 not yet run), skip silently
  Checkpoint resume (≥1 episode past "pending")
    → Return immediately — no output, no prompt
  --yes flag
    → Skip prompt regardless of checkpoint state

Usage
─────
    from src.utils.dry_run import DryRunGuard

    guard = DryRunGuard(cfg)
    if not guard.check_and_confirm(episode_ids, yes=args.yes):
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

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def check_and_confirm(self, episode_ids: list[str], yes: bool = False) -> bool:
        """
        Return True if the pipeline should proceed, False if the user aborted.

        Parameters
        ----------
        episode_ids:
            Full list of episode IDs discovered for this run.
        yes:
            If True, skip the interactive prompt (for CI / scripted runs).
        """
        if self._is_resume(episode_ids):
            logger.debug("DryRunGuard: checkpoint resume detected — skipping pre-flight estimate")
            return True

        # New show — attempt token estimation (prints the table internally)
        estimate = self._run_estimate()
        if not estimate:
            # No aligned data yet (Stages 1+2 haven't run) — nothing to estimate
            logger.debug(
                "DryRunGuard: no aligned cache found — "
                "skipping pre-flight estimate (run Stages 1+2 first for cost data)"
            )
            return True

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
        """Run TokenCounter; return empty dict on failure or no data."""
        try:
            from src.utils.token_counter import TokenCounter
            return TokenCounter(self._cfg).estimate()
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
