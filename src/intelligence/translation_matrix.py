"""
Multi-language dual-track translation matrix.

Architecture (per episode):
  Step 1  DeepSeek  ZH → en_skeleton    (faithful, all plot facts preserved)
  Step 2A Claude    en_skeleton → en_refined  (US English polish, screen constraints)
  Step 2B DeepSeek  en_skeleton → th / vi / … (concurrent, one call per language)
  Step 3  Validate  code-layer ≤3 lines / ≤40 chars / ≤140 total + correction retry
  Step 4  Write     data/cache/translation/{ep}_translation.json
                    data/output/translations/{ep}_{lang}.srt

Entry points
────────────
    matrix = TranslationMatrix(cfg)
    matrix.run_all(episode_ids)        # all episodes
    matrix.run_episode("01")           # single episode

Cache is hit if data/cache/translation/{ep}_translation.json already exists and
covers the same target_languages as the current config.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# ── Known ISO 639-1 codes → full language name for prompts ────────────────────
_LANG_NAMES: dict[str, str] = {
    "th": "Thai",
    "vi": "Vietnamese",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "ms": "Malay",
    "ar": "Arabic",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "en": "English",
}

# ── Screen safety hard limits (same values enforced in prompts) ───────────────
_MAX_LINES    = 3
_MAX_LINE_LEN = 40
_MAX_TOTAL    = 140


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _secs_to_srt_ts(secs: float) -> str:
    """Convert seconds (float) to SRT timestamp  HH:MM:SS,mmm."""
    h   = int(secs // 3600)
    m   = int((secs % 3600) // 60)
    s   = int(secs % 60)
    ms  = int(round((secs % 1) * 1000))
    if ms == 1000:          # rounding edge case
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _is_cjk_text(text: str) -> bool:
    """Return True if text contains at least one CJK character."""
    return bool(re.search(r'[一-鿿㐀-䶿]', text))


def _check_screen_limits(text: str) -> bool:
    """Return True if *text* passes all screen safety constraints."""
    lines = text.split("\n")
    if len(lines) > _MAX_LINES:
        return False
    if any(len(line) > _MAX_LINE_LEN for line in lines):
        return False
    if len(text) > _MAX_TOTAL:
        return False
    return True


def _atomic_json_write(path: Path, data: dict | list) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ══════════════════════════════════════════════════════════════════════════════
#  TranslationMatrix
# ══════════════════════════════════════════════════════════════════════════════


class TranslationMatrix:
    """
    Orchestrates the three-stage multi-language translation pipeline for all
    episodes in a drama series.

    Parameters
    ----------
    cfg:
        Parsed ``config/settings.yaml`` dict.
    """

    def __init__(self, cfg: dict) -> None:
        trans_cfg = cfg.get("intelligence", {}).get("translation", {})
        self._enabled: bool = bool(trans_cfg.get("enabled", True))
        self._target_langs: list[str] = list(trans_cfg.get("target_languages", ["en"]))
        glossary_path_str: str = trans_cfg.get(
            "genre_glossary_path", "data/meta/fantasy_glossary.json"
        )

        paths = cfg.get("paths", {})
        self._cache_dir   = Path(paths.get("cache_dir",   "data/cache"))
        self._meta_dir    = Path(paths.get("meta_dir",    "data/meta"))
        self._output_dir  = Path(paths.get("output_dir",  "data/output"))
        self._aligned_dir = self._cache_dir / "aligned"
        self._trans_cache = self._cache_dir / "translation"
        self._trans_out   = self._output_dir / "translations"
        self._trans_cache.mkdir(parents=True, exist_ok=True)
        self._trans_out.mkdir(parents=True, exist_ok=True)

        # Jinja2 environment
        prompts_dir = Path("config/prompts")
        self._jinja = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=False,
            keep_trailing_newline=True,
        )

        # Character name map (zh aliases → English canonical name)
        self._char_map: dict[str, str] = self._build_char_map()

        # Glossary (domain-specific terms) — graceful fallback if file missing
        self._glossary: dict[str, str] = self._load_glossary(glossary_path_str)

        # LLM clients
        from src.utils.llm_client import LLMClient
        self.llm_ds = LLMClient.from_cfg_key(cfg, "llm")          # DeepSeek — skeleton + minor langs

        # Claude client — falls back to DeepSeek if llm_claude not configured
        if cfg.get("execution", {}).get("llm_claude"):
            self.llm_claude = LLMClient.from_cfg_key(cfg, "llm_claude")
        else:
            logger.warning(
                "TranslationMatrix: execution.llm_claude not configured — "
                "Track A (English refinement) will use DeepSeek instead of Claude"
            )
            self.llm_claude = self.llm_ds

    # ------------------------------------------------------------------ #
    #  Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def run_all(self, episode_ids: list[str]) -> None:
        """Translate all *episode_ids* sequentially (Step 2 within each episode is parallel)."""
        if not self._enabled:
            logger.info("TranslationMatrix disabled in config — skipping")
            return

        # Ensure Western-style names are assigned before any translation
        self._ensure_name_override()
        self._char_map = self._build_char_map()

        n = len(episode_ids)
        logger.info(
            "TranslationMatrix — %d episode(s)  languages=%s",
            n, self._target_langs,
        )
        n_ok = n_skip = n_err = 0
        for ep_id in episode_ids:
            try:
                skipped = self.run_episode(ep_id)
                if skipped:
                    n_skip += 1
                else:
                    n_ok += 1
            except Exception as exc:
                n_err += 1
                logger.error("TranslationMatrix failed for [%s]: %s", ep_id, exc, exc_info=True)

        logger.info(
            "TranslationMatrix done — translated=%d  skipped(cache)=%d  errors=%d",
            n_ok, n_skip, n_err,
        )
        self._log_coverage_report(episode_ids)

    def run_episode(self, ep_id: str) -> bool:
        """
        Translate one episode.  Returns True if cache hit (skipped).

        Raises on unrecoverable failure after all retries.
        """
        cache_path = self._trans_cache / f"{ep_id}_translation.json"

        # Cache hit: if cache covers current target_langs → emit SRTs and return
        if cache_path.exists():
            existing = self._load_cache(cache_path)
            cached_langs = set(existing.get("target_languages", []))
            if set(self._target_langs) <= cached_langs:
                logger.info(
                    "Translation cache hit — [%s] skipping LLM calls", ep_id
                )
                self._emit_srts(ep_id, existing)
                return True
            # Cache exists but covers fewer languages than requested → partial redo
            logger.info(
                "[%s] Cache found but missing languages %s — re-running",
                ep_id, sorted(set(self._target_langs) - cached_langs),
            )

        # Load aligned segments
        segs = self._load_aligned_segs(ep_id)
        if not segs:
            logger.warning("[%s] No aligned segments — translation skipped", ep_id)
            return False

        logger.info(
            "Translating [%s] — %d segment(s)  languages=%s",
            ep_id, len(segs), ["en"] + self._target_langs,
        )

        # Step 1 — ZH → EN skeleton (DeepSeek)
        skeleton = self._step1_skeleton(segs, ep_id)

        # Step 2 — EN refine (Claude) + minor langs (DeepSeek), concurrent
        en_refined, lang_results = self._step2_parallel(skeleton, segs, ep_id)

        # Assemble, validate, cache, emit
        cache_data = self._assemble_cache(ep_id, segs, skeleton, en_refined, lang_results)
        _atomic_json_write(cache_path, cache_data)
        logger.info("Translation cache written — [%s]  → %s", ep_id, cache_path.name)

        self._emit_srts(ep_id, cache_data)
        return False

    # ------------------------------------------------------------------ #
    #  Step 1: ZH → EN skeleton                                            #
    # ------------------------------------------------------------------ #

    def _step1_skeleton(
        self, segs: list[dict], ep_id: str
    ) -> list[str]:
        """
        Translate Chinese master_text → English skeleton via DeepSeek.

        Returns a list of EN skeleton strings aligned 1:1 with *segs*.
        Falls back to empty string per segment on unrecoverable error.
        """
        input_arr = [{"idx": i, "text": s["source_zh"]} for i, s in enumerate(segs)]
        system, user = self._render_skeleton_prompt(ep_id, input_arr)

        raw = self.llm_ds.complete(
            system=system, user=user, module_name="Translation_Skeleton"
        )
        result_arr = self._parse_translation_array(raw, len(segs), ep_id, "skeleton")

        # Correction retry if screen constraint violations detected
        result_arr = self._validate_and_correct(
            result_arr, ep_id, "skeleton (EN)", self.llm_ds,
            lambda violations: self._render_skeleton_correction(ep_id, input_arr, violations, result_arr),
        )

        # Fill any entries that are still empty after idx-scatter + screen correction
        result_arr = self._fill_missing_skeletons(result_arr, segs, ep_id)

        return [item.get("text", "") for item in result_arr]

    def _render_skeleton_prompt(
        self, ep_id: str, input_arr: list[dict]
    ) -> tuple[str, str]:
        tmpl = self._jinja.get_template("translate_en_skeleton.j2")
        combined = tmpl.render(
            episode_number=ep_id,
            char_map_json=json.dumps(self._char_map, ensure_ascii=False, indent=2),
            glossary_json=json.dumps(self._glossary, ensure_ascii=False, indent=2),
            segments_json=json.dumps(input_arr, ensure_ascii=False, indent=2),
        )
        return "You are a professional subtitle translator.", combined

    def _render_skeleton_correction(
        self,
        ep_id: str,
        input_arr: list[dict],
        violations: list[int],
        prev_result: list[dict],
    ) -> tuple[str, str]:
        bad_items = [{"idx": v, "original_zh": input_arr[v]["text"], "bad_en": prev_result[v].get("text", "")} for v in violations]
        user = (
            f"Episode {ep_id}: The following entries violate the single-line rule "
            f"(they must have no \\n). Fix ONLY these entries. "
            f"Return a JSON array of ONLY the fixed entries.\n\n"
            f"{json.dumps(bad_items, ensure_ascii=False, indent=2)}"
        )
        return "You are a professional subtitle translator.", user

    # ------------------------------------------------------------------ #
    #  Step 2: Parallel EN refine + minor langs                            #
    # ------------------------------------------------------------------ #

    def _step2_parallel(
        self,
        skeleton: list[str],
        segs: list[dict],
        ep_id: str,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """
        Run Track A (Claude EN refine) and Track B (DeepSeek minor langs) concurrently.

        Returns (en_refined, {lang: [translated_texts]}).
        """
        input_arr = [{"idx": i, "text": text} for i, text in enumerate(skeleton)]
        lang_results: dict[str, list[str]] = {}

        tasks: dict[str, object] = {}
        n_workers = 1 + len(self._target_langs)  # Claude track + one per minor lang

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Track A — English refinement (Claude)
            tasks["en"] = executor.submit(
                self._step2a_refine_en, input_arr, ep_id
            )
            # Track B — minor languages (DeepSeek, concurrent)
            for lang in self._target_langs:
                tasks[lang] = executor.submit(
                    self._step2b_minor_lang, input_arr, ep_id, lang
                )

            for key, future in tasks.items():
                try:
                    lang_results[key] = future.result()
                except Exception as exc:
                    logger.error(
                        "[%s] Step 2 failed for lang=%s: %s — filling with skeleton",
                        ep_id, key, exc,
                    )
                    lang_results[key] = list(skeleton)  # fallback to skeleton

        en_refined = lang_results.pop("en", list(skeleton))
        return en_refined, lang_results

    def _step2a_refine_en(self, input_arr: list[dict], ep_id: str) -> list[str]:
        """Track A: English polish via Claude."""
        tmpl = self._jinja.get_template("refine_en_claude.j2")
        user = tmpl.render(
            episode_number=ep_id,
            char_map_json=json.dumps(self._char_map, ensure_ascii=False, indent=2),
            segments_json=json.dumps(input_arr, ensure_ascii=False, indent=2),
        )
        system = "You are a US English subtitle polisher for short drama streaming content."
        raw = self.llm_claude.complete(
            system=system, user=user, module_name="Translation_EN_Refine"
        )
        result_arr = self._parse_translation_array(raw, len(input_arr), ep_id, "en_refined")
        result_arr = self._validate_and_correct(
            result_arr, ep_id, "en_refined", self.llm_claude,
            lambda violations: self._render_screen_correction(ep_id, input_arr, violations, result_arr, "English"),
        )
        result_arr = self._fill_missing_refined(result_arr, input_arr, ep_id)
        return [item.get("text", "") for item in result_arr]

    def _fill_missing_refined(
        self,
        result_arr: list[dict],
        skeleton_arr: list[dict],
        ep_id: str,
    ) -> list[dict]:
        """
        Retry Claude refinement for entries where en_refined is empty but
        en_skeleton has content.  Sends only the missing (idx, skeleton_text)
        pairs to avoid re-processing the full episode.
        """
        missing = [
            i for i, item in enumerate(result_arr)
            if not item.get("text", "").strip()
            and i < len(skeleton_arr)
            and skeleton_arr[i].get("text", "").strip()
        ]
        if not missing:
            return result_arr

        logger.warning(
            "[%s][en_refined] %d/%d entries empty after scatter — targeted fill retry",
            ep_id, len(missing), len(result_arr),
        )

        retry_input = [
            {"idx": i, "text": skeleton_arr[i]["text"]}
            for i in missing
        ]
        system = "You are a US English subtitle polisher for short drama streaming content."
        user = (
            f"Episode {ep_id}: Polish these English subtitle entries. "
            f"Each input must produce exactly ONE polished output — never merge entries. "
            f"Max 3 lines, 40 chars/line, 140 chars total. "
            f"Return ONLY a JSON array of {{\"idx\": <int>, \"text\": \"<polished>\"}} objects.\n\n"
            f"{json.dumps(retry_input, ensure_ascii=False, indent=2)}"
        )

        try:
            from src.utils.llm_client import extract_json_array
            raw = self.llm_claude.complete(
                system=system, user=user, module_name="Translation_EN_Refine_Fill"
            )
            arr = extract_json_array(raw)
            for item in arr:
                if not isinstance(item, dict):
                    continue
                idx = int(item.get("idx", -1))
                text = str(item.get("text", "")).strip()
                if text and 0 <= idx < len(result_arr):
                    result_arr[idx]["text"] = text
        except Exception as exc:
            logger.warning(
                "[%s] Refined fill retry failed: %s — will fall back to skeleton",
                ep_id, exc,
            )

        return result_arr

    def _step2b_minor_lang(
        self, input_arr: list[dict], ep_id: str, lang: str
    ) -> list[str]:
        """Track B: minor language translation via DeepSeek."""
        lang_name = _LANG_NAMES.get(lang, lang.upper())
        tmpl = self._jinja.get_template("translate_minor_lang.j2")
        user = tmpl.render(
            episode_number=ep_id,
            target_language=lang_name,
            char_map_json=json.dumps(self._char_map, ensure_ascii=False, indent=2),
            segments_json=json.dumps(input_arr, ensure_ascii=False, indent=2),
        )
        system = f"You are a professional subtitle translator specializing in {lang_name}."
        raw = self.llm_ds.complete(
            system=system, user=user, module_name=f"Translation_{lang.upper()}"
        )
        result_arr = self._parse_translation_array(raw, len(input_arr), ep_id, lang)
        result_arr = self._validate_and_correct(
            result_arr, ep_id, lang, self.llm_ds,
            lambda violations: self._render_screen_correction(ep_id, input_arr, violations, result_arr, lang_name),
        )
        return [item.get("text", "") for item in result_arr]

    # ------------------------------------------------------------------ #
    #  Validation + correction retry                                        #
    # ------------------------------------------------------------------ #

    def _validate_and_correct(
        self,
        result_arr: list[dict],
        ep_id: str,
        lang_label: str,
        client,
        make_correction_prompt,
    ) -> list[dict]:
        """
        Check each element against screen safety constraints.
        Trigger one correction retry for violating entries.
        Fallback: truncate if correction still fails.
        """
        violations = [
            i for i, item in enumerate(result_arr)
            if not _check_screen_limits(item.get("text", ""))
        ]
        if not violations:
            return result_arr

        logger.warning(
            "[%s][%s] %d/%d entries violate screen constraints — correction retry",
            ep_id, lang_label, len(violations), len(result_arr),
        )

        try:
            system, user = make_correction_prompt(violations)
            raw = client.complete(
                system=system, user=user, module_name="Translation_Correction"
            )
            # Parse correction response directly by idx value.
            # Do NOT use _parse_translation_array (which enforces expected_count
            # positional scatter) — the LLM returns original position idx values,
            # not 0-based indices into the violations subset.
            from src.utils.llm_client import extract_json_array
            arr = extract_json_array(raw)
            for i, item in enumerate(arr):
                if isinstance(item, dict):
                    idx = int(item.get("idx", violations[i] if i < len(violations) else -1))
                    text = str(item.get("text", "")).strip()
                else:
                    idx = violations[i] if i < len(violations) else -1
                    text = str(item).strip()
                if text and 0 <= idx < len(result_arr):
                    result_arr[idx]["text"] = text
        except Exception as exc:
            logger.warning(
                "[%s][%s] Correction retry failed (%s) — falling back to truncation",
                ep_id, lang_label, exc,
            )

        # Final pass: truncate anything still over limit
        still_bad = [
            i for i in violations
            if not _check_screen_limits(result_arr[i].get("text", ""))
        ]
        for i in still_bad:
            result_arr[i]["text"] = self._truncate_to_limits(result_arr[i].get("text", ""))

        return result_arr

    @staticmethod
    def _render_screen_correction(
        ep_id: str,
        input_arr: list[dict],
        violations: list[int],
        prev_result: list[dict],
        lang_name: str,
    ) -> tuple[str, str]:
        bad = [
            {
                "idx": v,
                "source": input_arr[v]["text"] if v < len(input_arr) else "",
                "bad_translation": prev_result[v].get("text", "") if v < len(prev_result) else "",
                "violation": (
                    f"lines={len(prev_result[v].get('text','').split(chr(10)))}, "
                    f"max_line={max((len(l) for l in prev_result[v].get('text','').split(chr(10))), default=0)}, "
                    f"total={len(prev_result[v].get('text',''))}"
                    if v < len(prev_result) else "unknown"
                ),
            }
            for v in violations
        ]
        user = (
            f"Episode {ep_id} — {lang_name} subtitle correction.\n"
            f"The following entries exceed screen limits (max 3 lines, max 40 chars/line, max 140 chars total).\n"
            f"Rewrite ONLY these entries to fit within the limits. Preserve the meaning.\n"
            f"Return a JSON array of ONLY the fixed entries with their original idx values.\n\n"
            f"{json.dumps(bad, ensure_ascii=False, indent=2)}"
        )
        system = f"You are a professional subtitle editor ensuring screen safety for {lang_name} subtitles."
        return system, user

    @staticmethod
    def _truncate_to_limits(text: str) -> str:
        """Emergency truncation fallback when LLM correction still fails."""
        lines = text.split("\n")[:_MAX_LINES]
        lines = [line[:_MAX_LINE_LEN] for line in lines]
        result = "\n".join(lines)
        return result[:_MAX_TOTAL]

    # ------------------------------------------------------------------ #
    #  JSON parsing with count validation                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_translation_array(
        raw: str, expected_count: int, ep_id: str, step: str
    ) -> list[dict]:
        """
        Parse LLM response as a JSON array of {idx, text} dicts.

        Each item is scattered into its correct position using the idx field —
        gaps caused by LLM merging or skipping entries remain at the right slot
        rather than bunching up at the end.
        """
        from src.utils.llm_client import extract_json_array
        try:
            arr = extract_json_array(raw)
        except ValueError as exc:
            logger.error(
                "[%s][%s] JSON array parse failed: %s — using empty fallback",
                ep_id, step, exc,
            )
            return [{"idx": i, "text": ""} for i in range(expected_count)]

        # Build a positional result array, then scatter each item by idx
        result = [{"idx": i, "text": ""} for i in range(expected_count)]
        placed = 0
        out_of_range = 0
        for i, item in enumerate(arr):
            if isinstance(item, dict):
                idx = int(item.get("idx", i))
                text = str(item.get("text", ""))
            else:
                idx = i
                text = str(item)
            if 0 <= idx < expected_count:
                result[idx]["text"] = text
                placed += 1
            else:
                out_of_range += 1

        n_returned = len(arr)
        if n_returned != expected_count or out_of_range:
            logger.warning(
                "[%s][%s] LLM returned %d elements (expected %d); "
                "placed=%d  out-of-range=%d",
                ep_id, step, n_returned, expected_count, placed, out_of_range,
            )

        return result

    def _fill_missing_skeletons(
        self,
        result_arr: list[dict],
        segs: list[dict],
        ep_id: str,
    ) -> list[dict]:
        """
        Targeted retry for skeleton entries that are empty after idx-scatter.

        Sends ONLY the missing (idx, source_zh) pairs back to DeepSeek and
        patches the result in-place.  Handles the common case where the LLM
        merged consecutive short lines, leaving gaps at the original positions.
        """
        missing = [
            i for i, item in enumerate(result_arr)
            if not item.get("text", "").strip() and i < len(segs)
        ]
        if not missing:
            return result_arr

        logger.warning(
            "[%s][skeleton] %d/%d entries empty after scatter — targeted fill retry",
            ep_id, len(missing), len(result_arr),
        )

        retry_input = [
            {"idx": i, "text": segs[i]["source_zh"]}
            for i in missing
        ]
        system = "You are a professional subtitle translator."
        user = (
            f"Episode {ep_id}: Translate these Chinese subtitle segments to English. "
            f"Each input segment MUST produce exactly ONE output — never merge them. "
            f"Return ONLY a JSON array of {{\"idx\": <int>, \"text\": \"<translation>\"}} objects.\n\n"
            f"{json.dumps(retry_input, ensure_ascii=False, indent=2)}"
        )

        try:
            from src.utils.llm_client import extract_json_array
            raw = self.llm_ds.complete(
                system=system, user=user, module_name="Translation_Fill"
            )
            arr = extract_json_array(raw)
            for item in arr:
                if not isinstance(item, dict):
                    continue
                idx = int(item.get("idx", -1))
                text = str(item.get("text", "")).strip()
                if 0 <= idx < len(result_arr) and text:
                    result_arr[idx]["text"] = text
        except Exception as exc:
            logger.warning(
                "[%s] Skeleton fill retry failed: %s — leaving entries empty",
                ep_id, exc,
            )

        return result_arr

    # ------------------------------------------------------------------ #
    #  Cache & SRT output                                                   #
    # ------------------------------------------------------------------ #

    def _assemble_cache(
        self,
        ep_id: str,
        segs: list[dict],
        skeleton: list[str],
        en_refined: list[str],
        lang_results: dict[str, list[str]],
    ) -> dict:
        all_langs = ["en"] + self._target_langs
        segments_out = []
        for i, seg in enumerate(segs):
            entry: dict = {
                "segment_id":  seg["segment_id"],
                "start":       seg["start"],
                "end":         seg["end"],
                "source_zh":   seg["source_zh"],
                "en_skeleton": skeleton[i] if i < len(skeleton) else "",
                "en_refined":  en_refined[i] if i < len(en_refined) else "",
            }
            for lang in self._target_langs:
                texts = lang_results.get(lang, [])
                entry[lang] = texts[i] if i < len(texts) else ""
            segments_out.append(entry)

        return {
            "episode":          ep_id,
            "segment_count":    len(segs),
            "target_languages": all_langs,
            "segments":         segments_out,
        }

    @staticmethod
    def _load_cache(path: Path) -> dict:
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _emit_srts(self, ep_id: str, cache_data: dict) -> None:
        """Write one SRT file per output language to data/output/translations/."""
        segs = cache_data.get("segments", [])
        en_texts = [s.get("en_refined") or s.get("en_skeleton", "") for s in segs]

        # Cross-validate: warn if any EN segments are empty
        empty_en = sum(1 for t in en_texts if not t.strip())
        if empty_en:
            logger.warning(
                "[%s] EN coverage: %d/%d segments are empty",
                ep_id, empty_en, len(segs),
            )

        self._write_srt(ep_id, "en", en_texts, segs)
        for lang in self._target_langs:
            texts = [s.get(lang, "") for s in segs]
            self._write_srt(ep_id, lang, texts, segs)

    def _write_srt(
        self,
        ep_id: str,
        lang: str,
        texts: list[str],
        segs: list[dict],
    ) -> None:
        srt_path = self._trans_out / f"{ep_id}_{lang}.srt"
        blocks: list[str] = []
        counter = 1
        for text, seg in zip(texts, segs):
            if not text.strip():
                continue
            ts_start = _secs_to_srt_ts(seg["start"])
            ts_end   = _secs_to_srt_ts(seg["end"])
            blocks.append(f"{counter}\n{ts_start} --> {ts_end}\n{text}")
            counter += 1

        content = "\n\n".join(blocks) + "\n"
        tmp = srt_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(srt_path)
        logger.info("SRT written — [%s][%s]  %d entries  → %s", ep_id, lang, counter - 1, srt_path.name)

    # ------------------------------------------------------------------ #
    #  Coverage report                                                      #
    # ------------------------------------------------------------------ #

    def _log_coverage_report(self, episode_ids: list[str]) -> None:
        """
        Summarise EN translation coverage across all episodes after a run.
        Logs one WARNING line per episode with gaps, then a totals summary.
        """
        issues: list[str] = []
        total_segs = 0
        total_empty = 0

        for ep_id in episode_ids:
            cache_path = self._trans_cache / f"{ep_id}_translation.json"
            if not cache_path.exists():
                issues.append(f"  [{ep_id}] no translation cache")
                continue
            try:
                data = self._load_cache(cache_path)
                segs = data.get("segments", [])
                empty = sum(
                    1 for s in segs
                    if not (s.get("en_refined") or s.get("en_skeleton", "")).strip()
                )
                total_segs  += len(segs)
                total_empty += empty
                if empty:
                    issues.append(f"  [{ep_id}] {empty}/{len(segs)} EN segments empty")
            except Exception as exc:
                issues.append(f"  [{ep_id}] cache read error: {exc}")

        if issues:
            logger.warning(
                "EN coverage gaps in %d/%d episode(s)  (%d/%d segments empty):\n%s",
                len(issues), len(episode_ids),
                total_empty, total_segs,
                "\n".join(issues),
            )
        else:
            logger.info(
                "EN coverage OK — all %d episodes, %d segments, 0 empty",
                len(episode_ids), total_segs,
            )

    # ------------------------------------------------------------------ #
    #  Aligned segment loading                                              #
    # ------------------------------------------------------------------ #

    def _load_aligned_segs(self, ep_id: str) -> list[dict]:
        """
        Read data/cache/aligned/{ep_id}_aligned.json.
        Returns list of {segment_id, start, end, source_zh}.
        source_zh = master_text from AlignedSegment.
        """
        aligned_path = self._aligned_dir / f"{ep_id}_aligned.json"
        if not aligned_path.exists():
            logger.warning(
                "[%s] Aligned JSON not found at %s", ep_id, aligned_path
            )
            return []

        with aligned_path.open(encoding="utf-8") as f:
            data = json.load(f)

        result = []
        for raw_seg in data.get("segments", []):
            text = raw_seg.get("master_text", "").strip()
            if not text:
                continue
            result.append({
                "segment_id": raw_seg.get("segment_id", ""),
                "start":      float(raw_seg.get("start", 0.0)),
                "end":        float(raw_seg.get("end",   0.0)),
                "source_zh":  text,
            })
        return result

    # ------------------------------------------------------------------ #
    #  Character map + glossary                                             #
    # ------------------------------------------------------------------ #

    def _build_char_map(self) -> dict[str, str]:
        """
        Build zh-name → English-name mapping.

        Priority: char_name_en_override.json (Western names) > meta.json canonical_en (pinyin).
        Only CJK aliases are added to the map — English aliases like "Ethan" are skipped
        to prevent them from being remapped to a different canonical name.
        """
        meta_path = self._meta_dir / "meta.json"
        if not meta_path.exists():
            logger.warning(
                "meta.json not found at %s — translation will proceed without "
                "character name constraints",
                meta_path,
            )
            return {}

        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        # Load Western-name overrides if available
        override: dict[str, str] = {}
        override_path = self._meta_dir / "char_name_en_override.json"
        if override_path.exists():
            try:
                with override_path.open(encoding="utf-8") as f:
                    override = json.load(f)
                logger.info(
                    "Loaded %d Western-name overrides from char_name_en_override.json",
                    len(override),
                )
            except Exception as exc:
                logger.warning("Failed to load char_name_en_override.json: %s", exc)

        char_map: dict[str, str] = {}
        for canonical_zh, entry in meta.get("characters", {}).items():
            # Prefer Western-name override; fall back to pinyin canonical_en
            en_name = override.get(canonical_zh) or entry.get("canonical_en")
            if not en_name:
                continue
            char_map[canonical_zh] = en_name
            for alias in entry.get("aliases", []):
                # Skip English aliases (e.g. "Ethan") — they must not be remapped
                if _is_cjk_text(alias):
                    char_map[alias] = en_name

        logger.info(
            "Character map built — %d zh→en name entries",
            len(char_map),
        )
        return char_map

    def _ensure_name_override(self) -> None:
        """
        Assign Western-style names to every character in meta.json.
        Result is cached in data/meta/char_name_en_override.json.

        Incremental: if the file already exists, only processes characters that
        are new in meta.json since the last run.  Merges new names into the
        existing override so previously assigned names are preserved.
        """
        override_path = self._meta_dir / "char_name_en_override.json"
        meta_path = self._meta_dir / "meta.json"
        if not meta_path.exists():
            logger.warning("meta.json not found — cannot run name localization")
            return

        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        all_chars = set(meta.get("characters", {}).keys())

        # Load existing override (may be empty on first run)
        existing_override: dict[str, str] = {}
        if override_path.exists():
            try:
                with override_path.open(encoding="utf-8") as f:
                    existing_override = json.load(f)
            except Exception as exc:
                logger.warning(
                    "Failed to load existing char_name_en_override.json: %s — will rebuild",
                    exc,
                )

        new_chars = all_chars - set(existing_override.keys())
        if not new_chars:
            logger.info(
                "char_name_en_override.json covers all %d characters — skipping name-localization LLM call",
                len(existing_override),
            )
            return

        # Build per-character descriptor for just the new characters
        characters: list[dict] = []
        for zh_canonical in sorted(new_chars):
            entry = meta["characters"][zh_canonical]
            aliases = entry.get("aliases", [])
            en_aliases = [a for a in aliases if not _is_cjk_text(a)]
            item: dict = {
                "zh_canonical": zh_canonical,
                "current_romanization": entry.get("canonical_en") or "",
            }
            if en_aliases:
                item["note"] = f"established English first name from show: {en_aliases[0]} ★"
            characters.append(item)

        drama_context = (
            "Title: Win the Love War (胜爱情战争). Genre: workplace romance / short drama. "
            "Setting: modern corporate environment in China. "
            "Main characters include a fashion director (林展虹), a physician (闻誉施), "
            "a CEO (陆子谦, known on-screen as Ethan)."
        )

        tmpl = self._jinja.get_template("localize_char_names.j2")
        prompt = tmpl.render(
            drama_context=drama_context,
            characters_json=json.dumps(characters, ensure_ascii=False, indent=2),
        )
        system = "You are a professional drama localization specialist."

        logger.info(
            "Calling Claude for Western name assignment (%d new character(s))…",
            len(characters),
        )
        try:
            from src.utils.llm_client import extract_json
            raw = self.llm_claude.complete(
                system=system, user=prompt, module_name="NameLocalization"
            )
            result = extract_json(raw)
            char_map = result.get("character_map", {})
            if not char_map:
                raise ValueError("Empty character_map in LLM response")

            merged = {**existing_override, **char_map}
            _atomic_json_write(override_path, merged)
            logger.info(
                "Western-name override updated — %d new character(s) added → %s",
                len(char_map), override_path.name,
            )
        except Exception as exc:
            logger.error(
                "Name localization failed: %s — pinyin names will be used as fallback",
                exc,
            )

    def _load_glossary(self, path_str: str) -> dict[str, str]:
        """
        Load genre-specific glossary from *path_str*.
        Returns empty dict (silently) if the file does not exist.
        """
        glossary_path = Path(path_str)
        if not glossary_path.exists():
            logger.debug(
                "Genre glossary not found at %s — translation proceeds without it",
                glossary_path,
            )
            return {}

        try:
            with glossary_path.open(encoding="utf-8") as f:
                data = json.load(f)
            terms = data if isinstance(data, dict) else data.get("terms", {})
            logger.info("Genre glossary loaded — %d term(s) from %s", len(terms), glossary_path)
            return terms
        except Exception as exc:
            logger.warning(
                "Failed to load genre glossary at %s: %s — proceeding without it",
                glossary_path, exc,
            )
            return {}
