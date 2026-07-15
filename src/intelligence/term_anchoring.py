"""
Stage 4.5 — Auto-Term Anchoring.

Generates data/meta/{drama}/en_terms.json before translation begins.
The file anchors:
  - Every character's canonical English name + gender/pronoun
  - Drama-specific glossary (locations, orgs, world-building terms)

TranslationMatrix reads en_terms.json to populate its char_map and glossary,
ensuring Qilin/Kirin-style drift never reaches the translated SRTs.

Idempotent: if en_terms.json already covers all characters in meta.json,
the LLM call is skipped.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"
_SRT_TEXT_RE = re.compile(r"^\d+$|^\d{2}:\d{2}:\d{2},\d{3} -->")
_MAX_SAMPLE_LINES = 120


def _extract_srt_text_lines(srt_path: Path) -> list[str]:
    """Return only the dialogue text lines from an SRT file (skip index/timestamp)."""
    lines: list[str] = []
    try:
        content = srt_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not _SRT_TEXT_RE.match(stripped):
                lines.append(stripped)
    except Exception as exc:
        logger.warning("Could not read SRT %s: %s", srt_path, exc)
    return lines


class TermAnchoring:
    """Build en_terms.json for a drama using Claude."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        paths = cfg.get("paths", {})
        self._meta_dir    = Path(paths.get("meta_dir",   "data/meta"))
        self._output_dir  = Path(paths.get("output_dir", "data/output"))
        self._source_lang = cfg.get("pipeline", {}).get("source_language", "zh")
        self._en_terms_path = self._meta_dir / "en_terms.json"

        from jinja2 import Environment, FileSystemLoader
        self._jinja = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            autoescape=False,
            keep_trailing_newline=True,
        )

    # ------------------------------------------------------------------ #

    def build(self) -> Optional[dict]:
        """
        Generate en_terms.json if needed.  Returns the terms dict (from cache
        or freshly generated), or None if meta.json is missing.
        """
        meta_path = self._meta_dir / "meta.json"
        if not meta_path.exists():
            logger.warning("TermAnchoring: meta.json not found at %s — skipping", meta_path)
            return None

        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        all_chars: set[str] = set(meta.get("characters", {}).keys())

        # Idempotency check: skip if file covers all current characters
        if self._en_terms_path.exists():
            try:
                with self._en_terms_path.open(encoding="utf-8") as f:
                    existing = json.load(f)
                covered = set(existing.get("characters", {}).keys())
                if all_chars <= covered:
                    logger.info(
                        "TermAnchoring: en_terms.json covers all %d character(s) — skipping",
                        len(all_chars),
                    )
                    return existing
                logger.info(
                    "TermAnchoring: %d new character(s) not in en_terms.json — rebuilding",
                    len(all_chars - covered),
                )
            except Exception as exc:
                logger.warning("TermAnchoring: failed to read en_terms.json: %s — rebuilding", exc)

        logger.info(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        logger.info("▶ [4.5] Term Anchoring — building en_terms.json")
        logger.info(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        characters_payload = self._build_characters_payload(meta)
        srt_sample = self._sample_srt_text()
        drama_title = self._meta_dir.name
        genre_context = self._infer_genre_context(meta)

        terms = self._call_llm(drama_title, genre_context, characters_payload, srt_sample)
        if terms:
            self._write(terms)
        return terms

    # ------------------------------------------------------------------ #

    def _build_characters_payload(self, meta: dict) -> list[dict]:
        """Build a compact list for the prompt — zh name + aliases + existing romanization."""
        result: list[dict] = []
        for zh_canonical, entry in meta.get("characters", {}).items():
            aliases = entry.get("aliases", [])
            en_aliases = [a for a in aliases if not _is_cjk_text(a)]
            item: dict = {"zh_canonical": zh_canonical}
            if aliases:
                item["aliases"] = aliases
            if entry.get("canonical_en"):
                item["existing_romanization"] = entry["canonical_en"]
            if en_aliases:
                item["note"] = f"established English name from show: {en_aliases[0]} ★"
            result.append(item)
        return result

    def _sample_srt_text(self, n_eps: int = 3) -> str:
        """Collect up to _MAX_SAMPLE_LINES dialogue lines from the first N CN SRTs."""
        subdir = "cn" if self._source_lang == "zh" else self._source_lang
        srt_dir = self._output_dir / subdir
        if not srt_dir.exists():
            return "(no SRT samples available)"

        srts = sorted(srt_dir.glob("*.srt"), key=lambda p: [
            int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", p.stem)
        ])[:n_eps]

        collected: list[str] = []
        for srt in srts:
            lines = _extract_srt_text_lines(srt)
            remaining = _MAX_SAMPLE_LINES - len(collected)
            if remaining <= 0:
                break
            collected.extend(lines[:remaining])

        return "\n".join(collected) if collected else "(no SRT samples available)"

    def _infer_genre_context(self, meta: dict) -> str:
        drama_title = self._meta_dir.name
        return f"Chinese short drama: {drama_title}"

    def _call_llm(
        self,
        drama_title: str,
        genre_context: str,
        characters: list[dict],
        srt_sample: str,
    ) -> Optional[dict]:
        from src.utils.llm_client import LLMClient, extract_json

        if self._cfg.get("execution", {}).get("llm_claude"):
            llm = LLMClient.from_cfg_key(self._cfg, "llm_claude")
        else:
            logger.warning("TermAnchoring: llm_claude not configured — using DeepSeek")
            llm = LLMClient.from_cfg_key(self._cfg, "llm")

        tmpl = self._jinja.get_template("anchor_terms.j2")
        prompt = tmpl.render(
            drama_title=drama_title,
            genre_context=genre_context,
            source_language=self._source_lang,
            characters_json=json.dumps(characters, ensure_ascii=False, indent=2),
            srt_sample=srt_sample,
        )
        system = "You are a drama localization specialist. Output only valid JSON."

        try:
            raw = llm.complete(system=system, user=prompt, module_name="TermAnchoring")
            terms = extract_json(raw)
            if "characters" not in terms:
                raise ValueError("LLM response missing 'characters' key")
            logger.info(
                "TermAnchoring: LLM returned %d character(s) and %d glossary term(s)",
                len(terms.get("characters", {})),
                len(terms.get("glossary", {})),
            )
            return terms
        except Exception as exc:
            logger.error("TermAnchoring: LLM call failed: %s — en_terms.json not written", exc)
            return None

    def _write(self, terms: dict) -> None:
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._en_terms_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(terms, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._en_terms_path)
        logger.info("TermAnchoring: en_terms.json written → %s", self._en_terms_path)


def _is_cjk_text(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)
