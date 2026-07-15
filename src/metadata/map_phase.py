"""
MapReduce — Map phase.

Splits episodes into batches of `metadata.map_batch_size` (default 20),
builds a condensed text representation from each episode's aligned JSON,
and calls the LLM once per batch to extract character names and aliases.

Output: data/cache/map_batches/batch_{NN}_entities.json  (one file per batch)

Checkpoint: existing batch files are skipped — safe to re-run after a
partial failure; only missing or new batches are re-processed.

Text extraction strategy
────────────────────────
source     : data/cache/aligned/{episode_id}_aligned.json  (master_text +
              context_text, produced by OverlapAligner)

same_lang  : emit master_text lines (Chinese ASR), adjacent-dedup'd
cross_lang : emit paired [OCR] / [ASR] lines for segments where both
              tracks are available (context_available == True)

Character budget: max_chars_per_batch (30 000 default) split evenly across
episodes in the batch.  Episodes are truncated individually — no borrowing.
The budget keeps LLM input tokens predictable and avoids 400 errors.

Batch entity JSON schema (output):
    {
      "batch_index": 0,
      "episodes": ["ep01", ..., "ep20"],
      "mode": "same_lang",
      "characters": {
        "霍建华": {
          "canonical": "霍建华",
          "aliases": ["霍总", "霍少爷"],
          "canonical_en": null,
          "ocr_en_variants": []
        }, ...
      }
    }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root is three levels up from this file (src/metadata/map_phase.py)
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"

# Hard cap on characters per batch LLM call (~30 k chars ≈ 20 k tokens)
_DEFAULT_MAX_CHARS = 30_000


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════���══════════════════════���══════════════════════════════════════════════════


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
    """Ensure every LLM-returned character entry has all required fields."""
    aliases = [a for a in raw.get("aliases", []) if isinstance(a, str) and a != canonical]
    return {
        "canonical": canonical,
        "aliases": sorted(set(aliases)),
        "canonical_en": raw.get("canonical_en") or None,
        "ocr_en_variants": sorted(
            {v for v in raw.get("ocr_en_variants", []) if isinstance(v, str)}
        ),
    }


def _try_recover_truncated_json(text: str) -> "dict | None":
    """
    Extract complete character entries from a truncated JSON response.
    Scans character-by-character; stops at the first incomplete entry.
    Returns {"characters": {...}} or None if nothing could be recovered.
    """
    import re

    m = re.search(r'"characters"\s*:\s*\{', text)
    if not m:
        return None

    pos = m.end()
    n = len(text)
    characters: dict = {}

    while pos < n and text[pos] in " \t\n\r":
        pos += 1

    while pos < n and text[pos] == '"':
        # Parse canonical name
        key_start = pos + 1
        j = key_start
        esc = False
        while j < n:
            if esc:
                esc = False
            elif text[j] == "\\":
                esc = True
            elif text[j] == '"':
                break
            j += 1
        if j >= n:
            break
        canonical = text[key_start:j]
        pos = j + 1

        # Skip ": "
        while pos < n and text[pos] in " \t\n\r:":
            pos += 1
        if pos >= n or text[pos] != "{":
            break

        # Brace-track the value object
        depth = 0
        k = pos
        in_str = False
        esc = False
        while k < n:
            ch = text[k]
            if esc:
                esc = False
            elif ch == "\\" and in_str:
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            k += 1

        if depth != 0:
            break  # Incomplete entry — stop here

        try:
            characters[canonical] = json.loads(text[pos : k + 1])
        except json.JSONDecodeError:
            break

        pos = k + 1
        while pos < n and text[pos] in " \t\n\r,":
            pos += 1

    return {"characters": characters} if characters else None


# ══════���══════════════════════════��═══════════════════════════════���════════════
#  MapPhase
# ══════════════════════��═════════════════════════════════��═════════════════════


class MapPhase:
    """
    Runs the Map step of entity extraction.

    Instantiate once per pipeline run; call run(episode_ids) to process all
    episodes in the current project.
    """

    def __init__(self, cfg: dict, llm_client=None) -> None:
        self._mode: str = cfg["pipeline"]["mode"]
        self._source_language: str = cfg.get("pipeline", {}).get("source_language", "zh")
        self._batch_size: int = int(cfg.get("metadata", {}).get("map_batch_size", 20))
        self._max_chars: int = _DEFAULT_MAX_CHARS

        cache = Path(cfg["paths"]["cache_dir"])
        self._aligned_dir = cache / "aligned"
        self._batch_dir = cache / "map_batches"
        self._batch_dir.mkdir(parents=True, exist_ok=True)

        self._jinja = _build_jinja_env()
        self._prompt_name: str = (
            cfg.get("execution", {}).get("prompts", {}).get("map_extract", "map_extract.j2")
        )

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

    def run(self, episode_ids: list[str]) -> list[Path]:
        """
        Run Map phase on all provided episode IDs.

        Returns a list of batch JSON paths (in batch order).
        Episodes are split into batches of self._batch_size.
        Existing batch files are skipped (checkpoint resume).
        """
        if not episode_ids:
            logger.warning("MapPhase.run() called with empty episode list")
            return []

        batches = [
            episode_ids[i : i + self._batch_size]
            for i in range(0, len(episode_ids), self._batch_size)
        ]
        logger.info(
            "Map phase start — %d episode(s)  →  %d batch(es)  (batch_size=%d)",
            len(episode_ids), len(batches), self._batch_size,
        )

        paths: list[Path] = []
        for idx, batch in enumerate(batches):
            path = self._run_batch(idx, batch)
            paths.append(path)

        logger.info("Map phase complete — %d batch file(s) ready", len(paths))
        return paths

    # ------------------------------------------------------------------ #
    #  Per-batch processing                                                #
    # ------------------------------------------------------------------ #

    def _run_batch(self, batch_idx: int, episode_ids: list[str]) -> Path:
        out_path = self._batch_dir / f"batch_{batch_idx:02d}_entities.json"
        if out_path.exists():
            logger.info("Map batch %02d cache hit — skipped", batch_idx)
            return out_path

        ep_range = f"{episode_ids[0]}–{episode_ids[-1]}"
        logger.info(
            "Map batch %02d — episodes=%s  building text ...",
            batch_idx, ep_range,
        )

        text_content = self._build_batch_text(episode_ids)
        if not text_content.strip():
            logger.warning(
                "Map batch %02d: no text extracted — no aligned files found.  "
                "Writing empty batch.",
                batch_idx,
            )
            payload = {
                "batch_index": batch_idx,
                "episodes": episode_ids,
                "mode": self._mode,
                "characters": {},
            }
            _atomic_json_write(out_path, payload)
            return out_path

        user_prompt = self._render_prompt(batch_idx, ep_range, text_content)

        logger.info(
            "Map batch %02d — calling LLM  (text_chars=%d)",
            batch_idx, len(text_content),
        )
        from src.utils.llm_client import extract_json, LLMCallError

        lang_label = "English" if self._source_language == "en" else "Chinese"
        try:
            raw_response = self._llm.complete(
                system=(
                    f"You are a precise {lang_label} drama script analyst. "
                    "Your response must be valid JSON only — no prose, "
                    "no explanations, no markdown."
                ),
                user=user_prompt,
                max_tokens=8192,
                module_name="Map_Extract",
            )
        except LLMCallError as exc:
            logger.error("Map batch %02d LLM call failed: %s", batch_idx, exc)
            raise

        characters = self._parse_response(raw_response, batch_idx)

        payload = {
            "batch_index": batch_idx,
            "episodes": episode_ids,
            "mode": self._mode,
            "characters": characters,
        }
        _atomic_json_write(out_path, payload)
        logger.info(
            "Map batch %02d done — %d character(s) extracted  → %s",
            batch_idx, len(characters), out_path.name,
        )
        return out_path

    # ------------------------------------------------------------------ #
    #  Text extraction                                                     #
    # ------------------------------------------------------------------ #

    def _build_batch_text(self, episode_ids: list[str]) -> str:
        """
        Load aligned JSON for each episode and format it as LLM input text.
        Applies per-episode char budget to keep the batch within _max_chars.
        """
        per_ep_budget = self._max_chars // max(len(episode_ids), 1)
        parts: list[str] = []

        for ep_id in episode_ids:
            aligned_path = self._aligned_dir / f"{ep_id}_aligned.json"
            if not aligned_path.exists():
                logger.warning(
                    "MapPhase: aligned file missing for [%s] — episode skipped", ep_id
                )
                continue

            with aligned_path.open(encoding="utf-8") as f:
                data = json.load(f)

            segments: list[dict] = data.get("segments", [])
            ep_text = self._format_episode(ep_id, segments, per_ep_budget)
            if ep_text:
                parts.append(ep_text)

        return "\n\n".join(parts)

    def _format_episode(
        self,
        ep_id: str,
        segments: list[dict],
        char_budget: int,
    ) -> str:
        """
        Convert one episode's aligned segments into a condensed text block.
        Adjacent-deduplicates on master_text to remove repeated subtitle frames.
        Stops when char_budget is reached.
        """
        lines: list[str] = []
        chars_used = 0
        prev_master = ""

        for seg in segments:
            master: str = seg.get("master_text", "").strip()
            if not master or master == prev_master:
                continue
            prev_master = master

            if self._mode == "same_lang":
                line = master
            else:
                # cross_lang: emit paired [OCR]/[ASR] lines
                if not seg.get("context_available"):
                    continue
                context: str = seg.get("context_text", "").strip()
                if not context:
                    continue
                line = f"[OCR] {master}\n[ASR] {context}"

            line_len = len(line) + 1  # +1 for newline
            if chars_used + line_len > char_budget:
                break

            lines.append(line)
            chars_used += line_len

        if not lines:
            return ""

        return f"# {ep_id}\n" + "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Prompt rendering                                                    #
    # ------------------------------------------------------------------ #

    def _render_prompt(
        self,
        batch_idx: int,
        episode_range: str,
        text_content: str,
    ) -> str:
        template = self._jinja.get_template(self._prompt_name)
        return template.render(
            batch_index=batch_idx,
            mode=self._mode,
            source_language=self._source_language,
            episode_range=episode_range,
            text_content=text_content,
        )

    # ------------------------------------------------------------------ #
    #  Response parsing                                                    #
    # ------------------------------------------------------------------ #

    def _parse_response(self, text: str, batch_idx: int) -> dict:
        """
        Parse LLM JSON response into a normalised characters dict.
        Falls back to partial recovery if the response was truncated.
        Raises ValueError only when nothing can be recovered.
        """
        from src.utils.llm_client import extract_json

        data = None
        try:
            data = extract_json(text)
        except ValueError:
            data = _try_recover_truncated_json(text)
            if data is not None:
                logger.warning(
                    "Map batch %02d: truncated response — recovered %d character(s)",
                    batch_idx, len(data.get("characters", {})),
                )
            else:
                raise ValueError(
                    f"Map batch {batch_idx:02d}: LLM returned non-JSON.  "
                    f"Raw (first 400): {text[:400]!r}"
                )

        raw_chars: dict = data.get("characters", {})
        if not isinstance(raw_chars, dict):
            logger.warning(
                "Map batch %02d: 'characters' is not a dict — empty result used",
                batch_idx,
            )
            return {}

        return {
            canonical: _normalise_entry(canonical, entry)
            for canonical, entry in raw_chars.items()
            if isinstance(canonical, str) and isinstance(entry, dict)
        }
