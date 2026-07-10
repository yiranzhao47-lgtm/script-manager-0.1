"""
MapReduce — Reduce phase.

Loads all Map-phase batch files, merges them locally via set-union on aliases,
then makes one final LLM call to canonicalize same-phonetic variants and OCR
misreads across the whole series.

Outputs
───────
data/meta/meta_raw.json  — always overwritten (auto-generated, reviewable)
data/meta/meta.json      — written ONLY if it does not already exist
                           (human-edit checkpoint: manual corrections survive
                           subsequent runs)

Checkpoint
──────────
If meta_raw.json already exists, the Reduce phase is skipped entirely.
Delete it to force a re-run.

Python merge algorithm
──────────────────────
Group batch entries by their canonical string (exact match).  For each group:
  •  Keep the first-seen canonical form (batches are deterministically ordered)
  •  Union all aliases (excluding the canonical itself)
  •  Keep the first non-null canonical_en
  •  Union all ocr_en_variants

LLM canonicalization (optional, same-phonetic / OCR-typo merging)
──────────────────────────────────────────────────────────────────
After Python merge, the merged characters dict is passed to the LLM via the
reduce_canonicalize.j2 template.  The LLM may:
  •  Merge entries that are same-phonetic variants of the same character
  •  Add discarded canonical forms as aliases
  •  Remove incidental non-character entries

If the LLM call fails or returns malformed JSON, the Python-merged result is
used as-is and a warning is logged.  The pipeline never stalls here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _atomic_json_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _build_jinja_env():
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as exc:
        raise ImportError("jinja2 required: pip install jinja2") from exc
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )


def _normalise_entry(canonical: str, raw: dict) -> dict:
    """Ensure every character entry has all required fields, no self-alias."""
    aliases = [a for a in raw.get("aliases", []) if isinstance(a, str) and a != canonical]
    return {
        "canonical": canonical,
        "aliases": sorted(set(aliases)),
        "canonical_en": raw.get("canonical_en") or None,
        "ocr_en_variants": sorted(
            {v for v in raw.get("ocr_en_variants", []) if isinstance(v, str)}
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ReducePhase
# ══════════════════════════════════════════════════════════════════════════════


class ReducePhase:
    """
    Runs the Reduce step: merge Map-phase batch files → LLM canonicalize →
    write meta_raw.json and meta.json.

    Instantiate once per pipeline run; call run(batch_paths).
    """

    def __init__(self, cfg: dict, llm_client=None) -> None:
        self._mode: str = cfg["pipeline"]["mode"]
        self._use_llm: bool = bool(
            cfg.get("metadata", {}).get("reduce_llm_canonicalize", True)
        )

        meta_dir_cfg: str = cfg.get("paths", {}).get("meta_dir", "data/meta")
        self._meta_dir = Path(meta_dir_cfg)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

        self._raw_path = self._meta_dir / "meta_raw.json"
        self._meta_path = self._meta_dir / "meta.json"

        self._prompt_name: str = (
            cfg.get("execution", {})
            .get("prompts", {})
            .get("reduce_canonicalize", "reduce_canonicalize.j2")
        )
        self._jinja = _build_jinja_env()

        # Accept a pre-configured shared client (for unified FinOps ledger) or
        # create a private one as a fallback (backward compatibility).
        if llm_client is not None:
            self._llm = llm_client
        else:
            from src.utils.llm_client import LLMClient
            self._llm = LLMClient(cfg)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run(self, batch_paths: list[Path]) -> Path:
        """
        Run Reduce phase.

        Returns path to meta_raw.json (or existing cached path if skipped).
        """
        if self._raw_path.exists():
            logger.info("Reduce phase cache hit — meta_raw.json exists, skipped")
            return self._raw_path

        if not batch_paths:
            logger.warning("ReducePhase.run() called with empty batch_paths")
            empty = {"characters": {}}
            _atomic_json_write(self._raw_path, empty)
            if not self._meta_path.exists():
                _atomic_json_write(self._meta_path, empty)
            return self._raw_path

        logger.info("Reduce phase start — merging %d batch file(s)", len(batch_paths))

        batches = self._load_batches(batch_paths)
        merged = self._python_merge(batches)
        logger.info("Python merge complete — %d unique canonical(s)", len(merged))

        if self._use_llm and merged:
            canonicalized = self._canonicalize_with_llm(merged, len(batches))
        else:
            if not self._use_llm:
                logger.info("LLM canonicalization disabled — using Python merge")
            canonicalized = merged

        payload = {"characters": canonicalized}
        _atomic_json_write(self._raw_path, payload)
        logger.info("meta_raw.json written — %d character(s)", len(canonicalized))

        if not self._meta_path.exists():
            _atomic_json_write(self._meta_path, payload)
            logger.info("meta.json written (first time)")
        else:
            logger.info("meta.json already exists — human edits preserved, not overwritten")

        return self._raw_path

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    def _load_batches(self, batch_paths: list[Path]) -> list[dict]:
        """Load and validate each batch JSON file; skip corrupt files with a warning."""
        batches: list[dict] = []
        for path in batch_paths:
            if not path.exists():
                logger.warning("Reduce: batch file missing — %s (skipped)", path.name)
                continue
            try:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Reduce: failed to load %s — %s (skipped)", path.name, exc)
                continue
            chars = data.get("characters", {})
            if not isinstance(chars, dict):
                logger.warning(
                    "Reduce: 'characters' not a dict in %s (skipped)", path.name
                )
                continue
            batches.append(chars)
        return batches

    # ------------------------------------------------------------------ #
    #  Python merge                                                        #
    # ------------------------------------------------------------------ #

    def _python_merge(self, batches: list[dict]) -> dict:
        """
        Set-union merge of all batch character dicts.

        Groups entries by exact canonical string.  Aliases are unioned across
        all batches; first non-null canonical_en wins; ocr_en_variants unioned.
        """
        merged: dict[str, dict] = {}

        for chars in batches:
            for canonical, entry in chars.items():
                if not isinstance(canonical, str) or not isinstance(entry, dict):
                    continue

                if canonical not in merged:
                    merged[canonical] = {
                        "canonical": canonical,
                        "aliases": set(
                            a for a in entry.get("aliases", [])
                            if isinstance(a, str) and a != canonical
                        ),
                        "canonical_en": entry.get("canonical_en") or None,
                        "ocr_en_variants": set(
                            v for v in entry.get("ocr_en_variants", [])
                            if isinstance(v, str)
                        ),
                    }
                else:
                    acc = merged[canonical]
                    acc["aliases"].update(
                        a for a in entry.get("aliases", [])
                        if isinstance(a, str) and a != canonical
                    )
                    if acc["canonical_en"] is None:
                        acc["canonical_en"] = entry.get("canonical_en") or None
                    acc["ocr_en_variants"].update(
                        v for v in entry.get("ocr_en_variants", [])
                        if isinstance(v, str)
                    )

        # Convert sets to sorted lists for deterministic output
        return {
            canonical: {
                "canonical": canonical,
                "aliases": sorted(acc["aliases"]),
                "canonical_en": acc["canonical_en"],
                "ocr_en_variants": sorted(acc["ocr_en_variants"]),
            }
            for canonical, acc in merged.items()
        }

    # ------------------------------------------------------------------ #
    #  LLM canonicalization                                                #
    # ------------------------------------------------------------------ #

    def _canonicalize_with_llm(self, characters: dict, n_batches: int) -> dict:
        """
        Send merged characters to the LLM for same-phonetic / OCR-typo merging.

        Returns the LLM-canonicalized dict, or falls back to the Python-merged
        input if the LLM call fails or returns malformed JSON.
        """
        from src.utils.llm_client import LLMCallError, extract_json

        batch_results_json = json.dumps({"characters": characters}, ensure_ascii=False, indent=2)

        template = self._jinja.get_template(self._prompt_name)
        user_prompt = template.render(
            n_batches=n_batches,
            batch_results_json=batch_results_json,
        )

        logger.info(
            "Reduce phase — calling LLM for canonicalization  (chars=%d  input_size=%d)",
            len(characters),
            len(batch_results_json),
        )

        try:
            raw_response = self._llm.complete(
                system=(
                    "You are a precise Chinese drama character analyst. "
                    "Your response must be valid JSON only — no prose, "
                    "no explanations, no markdown."
                ),
                user=user_prompt,
                max_tokens=6000,
                module_name="Reduce_Canonicalize",
            )
        except LLMCallError as exc:
            logger.warning(
                "Reduce LLM call failed — falling back to Python merge. Error: %s", exc
            )
            return characters

        try:
            data = extract_json(raw_response)
        except ValueError as exc:
            logger.warning(
                "Reduce LLM returned non-JSON — falling back to Python merge. "
                "Error: %s  Raw (first 400): %.400r",
                exc,
                raw_response,
            )
            return characters

        raw_chars = data.get("characters", {})
        if not isinstance(raw_chars, dict):
            logger.warning(
                "Reduce LLM 'characters' is not a dict — falling back to Python merge"
            )
            return characters

        canonicalized = {
            canonical: _normalise_entry(canonical, entry)
            for canonical, entry in raw_chars.items()
            if isinstance(canonical, str) and isinstance(entry, dict)
        }

        if not canonicalized:
            logger.warning(
                "Reduce LLM returned empty characters dict — falling back to Python merge"
            )
            return characters

        # Sanity-check: LLM should not drastically inflate the character count
        # (merging can only reduce or hold the count, never grow it significantly)
        if len(canonicalized) > len(characters) * 2:
            logger.warning(
                "Reduce LLM returned suspicious character count: %d → %d "
                "(expected ≤ %d) — falling back to Python merge",
                len(characters), len(canonicalized), len(characters) * 2,
            )
            return characters

        # Validate each entry has the required fields
        _REQUIRED_FIELDS = ("aliases", "canonical_en", "ocr_en_variants")
        malformed = [
            c for c, e in canonicalized.items()
            if not all(f in e for f in _REQUIRED_FIELDS)
        ]
        if malformed:
            logger.warning(
                "Reduce LLM: %d character(s) missing required fields %s: %s "
                "— _normalise_entry() will fill defaults",
                len(malformed), _REQUIRED_FIELDS, malformed[:5],
            )

        logger.info(
            "Reduce LLM canonicalization done — %d → %d character(s)",
            len(characters),
            len(canonicalized),
        )
        return canonicalized
