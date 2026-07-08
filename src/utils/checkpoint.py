"""
Pipeline state persistence for resume support.

Tracks per-episode status as a simple JSON file (data/cache/checkpoint.json).
All writes are atomic (write to .tmp, then rename) — crash-safe.

State machine
─────────────
  pending → asr_done → ocr_done → aligned → refined → complete

  same_lang  : all five transitions apply
  cross_lang : same states; OCR dedup is folded into the ocr_done transition
               (pipeline marks ocr_done only after both OCRRunner and
               OCRDedup have completed)

Global flags (stored under "global" key, separate from per-episode states)
───────────────────────────────────────────────────────────────────────────
  map_done    : all Map-phase batch files produced
  reduce_done : meta.json produced

These are informational only — MapPhase/ReducePhase have their own file-based
checkpoints and skip themselves when their output files already exist.

Usage
─────
    from src.utils.checkpoint import Checkpoint

    ckpt = Checkpoint(Path("data/cache/checkpoint.json"))
    ckpt.set_state("ep01", "asr_done")
    if ckpt.is_at_least("ep01", "ocr_done"):
        # skip OCR step
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Ordered state sequence (earlier states have lower index)
_STATES: tuple[str, ...] = (
    "pending",
    "asr_done",
    "ocr_done",
    "aligned",
    "refined",
    "complete",
)
_STATE_RANK: dict[str, int] = {s: i for i, s in enumerate(_STATES)}


class Checkpoint:
    """
    Persistent, crash-safe checkpoint for one pipeline run.

    The JSON file is read once at construction time and re-written atomically
    on every ``set_state`` / ``set_global`` call.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data = self._load()

    # ------------------------------------------------------------------ #
    #  Per-episode state                                                   #
    # ------------------------------------------------------------------ #

    def get_state(self, episode_id: str) -> str:
        """Return current state for *episode_id*, defaulting to 'pending'."""
        return self._data["episodes"].get(episode_id, "pending")

    def set_state(self, episode_id: str, state: str) -> None:
        """
        Advance *episode_id* to *state* and persist immediately.

        Raises ``ValueError`` if *state* is not a recognised state name.
        """
        if state not in _STATE_RANK:
            raise ValueError(
                f"Unknown checkpoint state {state!r}.  "
                f"Valid states: {list(_STATES)}"
            )
        self._data["episodes"][episode_id] = state
        self._save()
        logger.debug("Checkpoint  [%s] → %s", episode_id, state)

    def is_at_least(self, episode_id: str, state: str) -> bool:
        """
        Return True if *episode_id* has reached *state* or any later state.

        Examples::

            ckpt.is_at_least("ep01", "asr_done")   # True when asr_done/ocr_done/…
            ckpt.is_at_least("ep01", "pending")    # always True
        """
        current_rank = _STATE_RANK.get(self.get_state(episode_id), 0)
        target_rank = _STATE_RANK.get(state, 0)
        return current_rank >= target_rank

    def all_at_least(self, episode_ids: list[str], state: str) -> bool:
        """Return True if ALL supplied episode IDs have reached *state*."""
        return all(self.is_at_least(ep, state) for ep in episode_ids)

    # ------------------------------------------------------------------ #
    #  Global flags                                                         #
    # ------------------------------------------------------------------ #

    def get_global(self, key: str, default=None):
        """Read a pipeline-level flag (e.g. 'reduce_done')."""
        return self._data.get("global", {}).get(key, default)

    def set_global(self, key: str, value) -> None:
        """Write a pipeline-level flag atomically."""
        self._data.setdefault("global", {})[key] = value
        self._save()

    # ------------------------------------------------------------------ #
    #  Reporting                                                            #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, int]:
        """
        Return a count of episodes in each state.

        Useful for logging an at-a-glance progress snapshot::

            {"pending": 40, "aligned": 30, "complete": 10}
        """
        counts: dict[str, int] = {s: 0 for s in _STATES}
        for state in self._data["episodes"].values():
            if state in counts:
                counts[state] += 1
        return counts

    # ------------------------------------------------------------------ #
    #  Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with self._path.open(encoding="utf-8") as f:
                    data = json.load(f)
                # Ensure required top-level keys exist (forward-compat)
                data.setdefault("version", 1)
                data.setdefault("episodes", {})
                data.setdefault("global", {})
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Checkpoint file corrupt or unreadable — starting fresh.  "
                    "Error: %s  Path: %s",
                    exc, self._path,
                )
        return {"version": 1, "episodes": {}, "global": {}}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)
