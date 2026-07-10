"""
Token budget estimator (dry-run mode).

Scans all aligned JSON files, estimates input token counts per pipeline stage
using tiktoken (char/3 fallback when unavailable), and prints a budget report
before any LLM API call is made.

Usage
─────
    from src.utils.token_counter import TokenCounter
    counter = TokenCounter(cfg)
    result  = counter.estimate()   # prints table + writes token_estimate.json

Stages estimated
────────────────
  Stage 3  MapReduce    — Map batches (20 eps/batch) + one Reduce call
  Stage 4  Refine       — one LLM call per episode
  Stage 5  Drama        — Map call per episode + one Reduce (if drama_analysis enabled)
  Stage 6  Translation  — skeleton + EN refine + minor-lang calls (if ZH source)
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Per-call prompt overhead estimates (tokens) ───────────────────────────────
_OH_MAP_BATCH     = 450   # map_extract.j2 template + system
_OH_MAP_REDUCE    = 350   # reduce_canonicalize.j2 + system
_OH_REFINE        = 650   # refine prompt + system + meta JSON
_OH_DRAMA_MAP     = 550   # map_conflict.j2 + system
_OH_DRAMA_REDUCE  = 900   # reduce_rhythm.j2 + system (large prompt)
_OH_SKEL          = 450   # translate_en_skeleton.j2 + system
_OH_CLAUDE        = 350   # refine_en_claude.j2 + system
_OH_MINOR         = 400   # translate_minor_lang.j2 + system


def _make_encoder():
    """Return tiktoken encoder or None (graceful fallback)."""
    try:
        import tiktoken
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        return None


def _tok(text: str, enc) -> int:
    """Count tokens; fall back to len(text)//3 when no encoder."""
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 3)


# ══��═══════════════════════════════════════════════════════════════════════════
#  TokenCounter
# ═���════════════════════════════════════════════════════════════════════════════


class TokenCounter:
    """
    Pre-run token budget estimator.

    Parameters
    ----------
    cfg:
        Parsed config/settings.yaml dict (same object used by the pipeline).
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        paths = cfg.get("paths", {})
        self._cache_dir  = Path(paths.get("cache_dir",  "data/cache"))
        self._output_dir = Path(paths.get("output_dir", "data/output"))

        pipe = cfg.get("pipeline", {})
        self._mode            = pipe.get("mode", "same_lang")
        self._source_language = pipe.get("source_language", "zh")

        meta_cfg = cfg.get("metadata", {})
        self._batch_size: int = int(meta_cfg.get("map_batch_size", 20))

        trans_cfg = cfg.get("intelligence", {}).get("translation", {})
        self._translate = (
            trans_cfg.get("enabled", True)
            and self._source_language != "en"
        )
        self._target_langs: list[str] = trans_cfg.get("target_languages", [])

        drama_cfg = cfg.get("intelligence", {}).get("drama_analysis", {})
        self._drama         = drama_cfg.get("enabled", True)
        self._srt_char_limit: int = int(drama_cfg.get("srt_char_limit", 15000))

        self._enc = _make_encoder()

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def estimate(self) -> dict:
        """
        Run token budget estimation.

        Returns a dict with per-stage breakdowns and total.
        Writes ``token_estimate.json`` to output_dir.
        Logs a formatted report to INFO.
        """
        aligned_dir = self._cache_dir / "aligned"
        episodes = self._load_aligned(aligned_dir)
        if not episodes:
            return {}

        ep_ids = sorted(episodes)
        n = len(ep_ids)

        stage3 = self._estimate_stage3(episodes, ep_ids)
        stage4 = self._estimate_stage4(episodes, ep_ids)
        stage5 = self._estimate_stage5(ep_ids)
        stage6 = self._estimate_stage6(episodes, ep_ids)

        total_input = (
            stage3["input_tokens"]
            + stage4["input_tokens"]
            + stage5["input_tokens"]
            + stage6["input_tokens"]
        )

        result: dict = {
            "episodes_found": n,
            "encoder": "tiktoken" if self._enc is not None else "char/3 fallback",
            "stages": {
                "stage3_mapreduce": stage3,
                "stage4_refine":    stage4,
                "stage5_drama":     stage5,
                "stage6_translate": stage6,
            },
            "total_input_tokens": total_input,
        }

        # Cost estimate using configured pricing
        pricing = self._cfg.get("pricing", {})
        model   = self._cfg.get("execution", {}).get("llm", {}).get("model", "")
        price   = pricing.get(model, {})
        inp_rate   = float(price.get("input_cost_per_m",  0))
        out_rate   = float(price.get("output_cost_per_m", 0))
        if inp_rate > 0:
            # Rough output estimate: 25% of input tokens
            est_output = int(total_input * 0.25)
            cost = (total_input / 1_000_000) * inp_rate + (est_output / 1_000_000) * out_rate
            result["estimated_cost_cny"] = round(cost, 4)
            result["pricing_model"] = model

        self._print_report(result)
        self._write(result)
        return result

    # ------------------------------------------------------------------ #
    #  Stage estimators                                                    #
    # ------------------------------------------------------------------ #

    def _estimate_stage3(self, episodes: dict, ep_ids: list[str]) -> dict:
        """Map batches + Reduce."""
        n_batches = math.ceil(len(ep_ids) / self._batch_size)
        map_tokens = 0

        for i in range(0, len(ep_ids), self._batch_size):
            batch = ep_ids[i : i + self._batch_size]
            text  = self._format_batch_text(episodes, batch)
            map_tokens += _tok(text, self._enc) + _OH_MAP_BATCH

        # Reduce input: all batch outputs merged; ~300 tokens/character entity
        n_chars_est = max(10, len(ep_ids) // 3)
        reduce_tokens = n_chars_est * 300 + _OH_MAP_REDUCE

        return {
            "input_tokens": map_tokens + reduce_tokens,
            "map_calls":    n_batches,
            "reduce_calls": 1,
            "api_calls":    n_batches + 1,
        }

    def _estimate_stage4(self, episodes: dict, ep_ids: list[str]) -> dict:
        """One LLM call per episode."""
        total = 0
        per_ep: dict[str, int] = {}

        for ep_id in ep_ids:
            segs = episodes[ep_id]
            lines = [
                f"{s.get('master_text', '')} | {s.get('context_text', '')}"
                for s in segs
            ]
            tok = _tok("\n".join(lines), self._enc) + _OH_REFINE
            total += tok
            per_ep[ep_id] = tok

        max_tok = max(per_ep.values()) if per_ep else 0
        avg_tok = total // len(ep_ids) if ep_ids else 0

        # Flag episodes that exceed the configured warn threshold
        warn = int(self._cfg.get("execution", {})
                   .get("token_budget", {})
                   .get("warn_tokens", 50_000))
        over_budget = [ep for ep, t in per_ep.items() if t > warn]

        result: dict = {
            "input_tokens":       total,
            "api_calls":          len(ep_ids),
            "avg_tokens_per_ep":  avg_tok,
            "max_tokens_per_ep":  max_tok,
        }
        if over_budget:
            result["over_budget_eps"] = over_budget

        return result

    def _estimate_stage5(self, ep_ids: list[str]) -> dict:
        """Drama rhythm analysis — Map per episode + Reduce."""
        if not self._drama:
            return {"input_tokens": 0, "api_calls": 0, "note": "disabled"}

        subdir  = "cn" if self._source_language == "zh" else self._source_language
        srt_dir = self._output_dir / subdir

        map_tokens = 0
        missing_srts = 0

        for ep_id in ep_ids:
            srt_path = srt_dir / f"{ep_id}.srt"
            if srt_path.exists():
                srt_text = srt_path.read_text(encoding="utf-8")
                if len(srt_text) > self._srt_char_limit:
                    srt_text = srt_text[: self._srt_char_limit]
                tok = _tok(srt_text, self._enc) + _OH_DRAMA_MAP
            else:
                tok = 5_000 + _OH_DRAMA_MAP   # rough per-episode estimate
                missing_srts += 1
            map_tokens += tok

        # Reduce: conflict chain ≈ 1/3 of Map total (dialogues stripped)
        reduce_tokens = map_tokens // 3 + _OH_DRAMA_REDUCE

        result: dict = {
            "input_tokens": map_tokens + reduce_tokens,
            "map_calls":    len(ep_ids),
            "reduce_calls": 1,
            "api_calls":    len(ep_ids) + 1,
        }
        if missing_srts:
            result["note"] = (
                f"{missing_srts}/{len(ep_ids)} SRT(s) not found — "
                "those episodes use a 5k-token estimate"
            )
        return result

    def _estimate_stage6(self, episodes: dict, ep_ids: list[str]) -> dict:
        """Translation matrix — skeleton + EN refine + minor langs per episode."""
        if not self._translate:
            lang = self._source_language
            return {
                "input_tokens": 0,
                "api_calls":    0,
                "note":         f"skipped (source_language='{lang}')",
            }

        n_minor = len(self._target_langs)
        total   = 0

        for ep_id in ep_ids:
            segs    = episodes[ep_id]
            zh_text = " ".join(s.get("master_text", "") for s in segs)
            zh_tok  = _tok(zh_text, self._enc)

            skeleton = zh_tok + _OH_SKEL
            en_refine = int(zh_tok * 0.9) + _OH_CLAUDE
            minor     = (zh_tok + _OH_MINOR) * n_minor
            total    += skeleton + en_refine + minor

        calls_per_ep = 2 + n_minor
        return {
            "input_tokens": total,
            "api_calls":    len(ep_ids) * calls_per_ep,
            "tracks":       ["skeleton", "en_refined"] + self._target_langs,
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _load_aligned(self, aligned_dir: Path) -> dict[str, list]:
        """Load all *_aligned.json files; return {ep_id: segments_list}."""
        if not aligned_dir.exists():
            logger.warning(
                "TokenCounter: aligned cache not found at %s — "
                "run Stages 1+2 before estimating token budgets",
                aligned_dir,
            )
            return {}

        episodes: dict[str, list] = {}
        for fp in sorted(aligned_dir.glob("*_aligned.json")):
            ep_id = fp.stem.replace("_aligned", "")
            try:
                with fp.open(encoding="utf-8") as f:
                    data = json.load(f)
                segs = data if isinstance(data, list) else data.get("segments", [])
                if isinstance(segs, list):
                    episodes[ep_id] = segs
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("TokenCounter: skip %s — %s", fp.name, exc)

        if not episodes:
            logger.warning("TokenCounter: no valid aligned JSON files found in %s", aligned_dir)
        return episodes

    def _format_batch_text(self, episodes: dict, ep_ids: list[str]) -> str:
        """Format segments for a Map batch the way map_phase.py does."""
        lines: list[str] = []
        for ep_id in ep_ids:
            segs = episodes.get(ep_id, [])
            lines.append(f"=== {ep_id} ===")
            for seg in segs:
                m = seg.get("master_text", "")
                c = seg.get("context_text", "")
                if self._mode == "cross_lang" and c:
                    lines.append(f"[OCR] {c}")
                    lines.append(f"[ASR] {m}")
                else:
                    lines.append(m)
        return "\n".join(lines)

    def _write(self, result: dict) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out = self._output_dir / "token_estimate.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)
        logger.info("TokenCounter: budget report written → %s", out)

    def _print_report(self, result: dict) -> None:
        n      = result["episodes_found"]
        enc    = result["encoder"]
        stages = result["stages"]
        total  = result["total_input_tokens"]

        sep = "=" * 68
        print(sep)
        print(f"  Token Budget Estimate  |  {n} episodes  |  {enc}")
        print(sep)
        print(f"  {'Stage':<30} | {'Input Tokens':>13} | {'API Calls':>9}")
        print("-" * 68)

        labels = {
            "stage3_mapreduce": "Stage 3  MapReduce",
            "stage4_refine":    "Stage 4  Refine",
            "stage5_drama":     "Stage 5  Drama Analysis",
            "stage6_translate": "Stage 6  Translation",
        }
        for key, info in stages.items():
            label = labels.get(key, key)
            inp   = info.get("input_tokens", 0)
            calls = info.get("api_calls", 0)
            note  = info.get("note", "")
            note_str = f"  [{note}]" if note else ""
            print(f"  {label:<30} | {inp:>13,} | {calls:>9,}{note_str}")

            # Per-episode detail for Stage 4
            if key == "stage4_refine":
                avg = info.get("avg_tokens_per_ep", 0)
                mx  = info.get("max_tokens_per_ep", 0)
                over = info.get("over_budget_eps", [])
                print(f"    avg/ep={avg:,}  max/ep={mx:,}", end="")
                if over:
                    print(f"  ⚠ over-budget: {over}", end="")
                print()

        print("-" * 68)
        print(f"  {'TOTAL (input only)':<30} | {total:>13,} |")

        if "estimated_cost_cny" in result:
            cost  = result["estimated_cost_cny"]
            model = result.get("pricing_model", "?")
            print(f"  {'Est. cost (≈25% output)':<30} | {'¥' + str(cost):>13} |  {model}")

        print(sep)
