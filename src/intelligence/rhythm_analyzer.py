"""
Layer 5 Intelligence — Drama Rhythm Analyzer.

Two-phase MapReduce pipeline for pure-semantic Chinese short-drama analysis.

Map phase  : per-episode scene extraction (location, actions, unresolved debt,
             pivot signals, speaker-attributed dialogues).
Reduce phase: global macro blueprint — causal debt narrative, two structural
             pinch points, commercial flow type A/B, marketing clip picks.

Design constraints
──────────────────
• No numeric scoring, percentages, or intensity ratings anywhere in the pipeline.
• Map phase runs in parallel (ThreadPoolExecutor); each result is cached atomically
  so partial runs are resumable.
• Reduce phase receives a condensed conflict chain (scene facts only, no
  structured_dialogues) to stay within token budgets for long series.
• The LLMClient ledger is shared with the rest of the pipeline for unified FinOps.

Outputs
───────
data/cache/drama_map/{ep_id}_conflict.json  — per-episode cache (auto-generated)
data/output/drama_structure_graph.json      — final assembled result
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _natural_sort_key(ep_id: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", ep_id)]


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


# ══════════════════════════════════════════════════════════════════════════════
#  RhythmAnalyzer
# ══════════════════════════════════════════════════════════════════════════════


class RhythmAnalyzer:
    """
    Two-phase drama rhythm analysis engine.

    Parameters
    ----------
    cfg:
        Parsed ``config/settings.yaml`` dict.
    llm_client:
        Shared ``LLMClient`` instance (ledger is cumulative across all pipeline stages).
    output_dir:
        Directory where ``drama_structure_graph.json`` is written.
    cache_dir:
        Pipeline cache root; drama_map/ sub-directory is created here.
    meta:
        Parsed ``meta.json`` dict.  Used to inject the character list into the
        map prompt for speaker attribution.  Pass ``{}`` if unavailable.
    """

    def __init__(
        self,
        cfg: dict,
        llm_client,
        output_dir: Path,
        cache_dir: Path,
        meta: Optional[dict] = None,
    ) -> None:
        da_cfg = cfg.get("intelligence", {}).get("drama_analysis", {})
        self._map_workers:      int = int(da_cfg.get("map_workers",      4))
        self._map_max_tokens:   int = int(da_cfg.get("map_max_tokens",   6000))
        self._reduce_max_tokens:int = int(da_cfg.get("reduce_max_tokens", 8000))
        self._srt_char_limit:   int = int(da_cfg.get("srt_char_limit",   15000))

        self._llm = llm_client
        self._output_dir = output_dir
        self._drama_map_dir = cache_dir / "drama_map"
        self._drama_map_dir.mkdir(parents=True, exist_ok=True)

        self._characters: dict = (meta or {}).get("characters", {})
        self._jinja = _build_jinja_env()

        # SRT files live in the language subdir under output_dir (cn/ or en/ etc.)
        source_language = cfg.get("pipeline", {}).get("source_language", "zh")
        _subdir = "cn" if source_language == "zh" else source_language
        self._srt_dir = output_dir / _subdir

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run(self, episode_ids: list[str]) -> dict:
        """
        Run the full two-phase analysis.

        Returns the assembled drama structure graph dict (also written to disk).
        Raises RuntimeError if the map phase produces zero valid episode results.
        """
        if not episode_ids:
            raise ValueError("episode_ids must not be empty")

        logger.info(
            "RhythmAnalyzer: map phase — %d episode(s)  workers=%d",
            len(episode_ids), self._map_workers,
        )
        conflict_map = self._map_episodes(episode_ids)

        if not conflict_map:
            raise RuntimeError(
                "RhythmAnalyzer: map phase produced no valid results — "
                "check that SRT files exist in %s and the LLM is reachable" % self._srt_dir
            )

        logger.info(
            "RhythmAnalyzer: reduce phase — condensing %d episode(s)",
            len(conflict_map),
        )
        blueprint = self._reduce_to_blueprint(conflict_map)

        return self._assemble_and_write(conflict_map, blueprint, len(episode_ids))

    # ------------------------------------------------------------------ #
    #  Map phase                                                           #
    # ------------------------------------------------------------------ #

    def _map_episodes(self, episode_ids: list[str]) -> dict[str, dict]:
        """
        Analyse each episode in parallel.  Episodes that fail are logged and
        excluded from the conflict map (pipeline does not stall).
        """
        conflict_map: dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=self._map_workers) as pool:
            future_to_ep = {
                pool.submit(self._analyze_one_episode, ep_id): ep_id
                for ep_id in episode_ids
            }
            for future in as_completed(future_to_ep):
                ep_id = future_to_ep[future]
                try:
                    result = future.result()
                    conflict_map[ep_id] = result
                except Exception as exc:
                    logger.error(
                        "RhythmAnalyzer: map failed for [%s] — skipped.  Error: %s",
                        ep_id, exc,
                    )

        return conflict_map

    def _analyze_one_episode(self, ep_id: str) -> dict:
        """
        Analyse a single episode.  Returns the cached result if it exists.

        Cache file: drama_map/{ep_id}_conflict.json
        """
        from src.utils.llm_client import LLMCallError, extract_json

        cache_path = self._drama_map_dir / f"{ep_id}_conflict.json"
        if cache_path.exists():
            try:
                with cache_path.open(encoding="utf-8") as f:
                    cached = json.load(f)
                if "scenes" in cached and isinstance(cached["scenes"], list):
                    logger.debug("RhythmAnalyzer: cache hit — %s", ep_id)
                    return cached
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "RhythmAnalyzer: corrupt cache for [%s] — re-analysing.  %s",
                    ep_id, exc,
                )

        srt_content = self._load_srt(ep_id)

        template = self._jinja.get_template("map_conflict.j2")
        user_prompt = template.render(
            episode_id=ep_id,
            characters=self._characters,
            srt_content=srt_content,
        )

        raw = self._llm.complete(
            system=(
                "You are a precise Chinese drama script analyst. "
                "Your response must be valid JSON only — no prose, no markdown."
            ),
            user=user_prompt,
            max_tokens=self._map_max_tokens,
            json_mode=True,
            module_name="Drama_Analysis",
        )

        try:
            data = extract_json(raw)
        except ValueError as exc:
            raise ValueError(
                f"[{ep_id}] LLM returned non-JSON: {exc}"
            ) from exc

        if not isinstance(data.get("scenes"), list):
            raise ValueError(
                f"[{ep_id}] LLM response missing 'scenes' list"
            )

        # Ensure episode_id is present in the cached result
        data["episode_id"] = ep_id
        _atomic_json_write(cache_path, data)
        logger.info("RhythmAnalyzer: [%s] analysed — %d scene(s)", ep_id, len(data["scenes"]))
        return data

    # ------------------------------------------------------------------ #
    #  SRT loading                                                         #
    # ------------------------------------------------------------------ #

    def _load_srt(self, ep_id: str) -> str:
        """
        Load the SRT file for ep_id, truncated to srt_char_limit.

        Raises FileNotFoundError if no SRT file can be found.
        """
        srt_path = self._srt_dir / f"{ep_id}.srt"
        if not srt_path.exists():
            raise FileNotFoundError(
                f"SRT file not found for episode [{ep_id}]: expected {srt_path}"
            )
        content = srt_path.read_text(encoding="utf-8")
        if len(content) > self._srt_char_limit:
            content = content[: self._srt_char_limit]
            logger.debug(
                "RhythmAnalyzer: [%s] SRT truncated to %d chars",
                ep_id, self._srt_char_limit,
            )
        return content

    # ------------------------------------------------------------------ #
    #  Reduce phase                                                        #
    # ------------------------------------------------------------------ #

    def _reduce_to_blueprint(self, conflict_map: dict[str, dict]) -> dict:
        """
        Build the global macro blueprint from the condensed conflict chain.

        Falls back to an empty blueprint dict on LLM failure so the pipeline
        can still persist partial results.
        """
        from src.utils.llm_client import LLMCallError, extract_json

        chain_text = self._build_conflict_chain(conflict_map)
        n_episodes = len(conflict_map)

        template = self._jinja.get_template("reduce_rhythm.j2")
        user_prompt = template.render(
            n_episodes=n_episodes,
            conflict_chain=chain_text,
        )

        logger.info(
            "RhythmAnalyzer: reduce LLM call — %d eps  chain_chars=%d",
            n_episodes, len(chain_text),
        )

        try:
            raw = self._llm.complete(
                system=(
                    "You are a precise Chinese drama story architect. "
                    "Your response must be valid JSON only — no prose, no markdown."
                ),
                user=user_prompt,
                max_tokens=self._reduce_max_tokens,
                json_mode=True,
                module_name="Drama_Blueprint",
            )
        except LLMCallError as exc:
            logger.warning(
                "RhythmAnalyzer: reduce LLM call failed — blueprint will be empty.  %s", exc
            )
            return {}

        try:
            blueprint = extract_json(raw)
        except ValueError as exc:
            logger.warning(
                "RhythmAnalyzer: reduce LLM returned non-JSON — blueprint empty.  %s", exc
            )
            return {}

        # Schema validation — warn on missing keys but return partial blueprint
        _REQUIRED = ("debt_chain_narrative", "first_pinch", "second_pinch",
                     "post_first_pinch_flow", "marketing_clips")
        missing = [k for k in _REQUIRED if k not in blueprint]
        if missing:
            logger.warning(
                "RhythmAnalyzer: Reduce blueprint missing key(s): %s — "
                "paywall report and clip recommendations may be incomplete.",
                missing,
            )
        return blueprint

    def _build_conflict_chain(self, conflict_map: dict[str, dict]) -> str:
        """
        Build a condensed plain-text conflict chain from the map results.

        structured_dialogues are intentionally excluded to keep the reduce
        prompt within token budgets for long series (80+ episodes).
        """
        lines: list[str] = []
        for ep_id in sorted(conflict_map, key=_natural_sort_key):
            ep_data = conflict_map[ep_id]
            scenes = ep_data.get("scenes", [])
            lines.append(f"=== {ep_id} ===")
            for scene in scenes:
                scene_id    = scene.get("scene_id", "?")
                location    = scene.get("location", "")
                time_marker = scene.get("time", "")
                actions     = scene.get("scene_actions", [])
                debt        = scene.get("unresolved_debt") or ""
                pivots      = scene.get("pivot_signals", [])

                lines.append(f"  [{scene_id}]  {location}  /  {time_marker}")
                for act in actions:
                    lines.append(f"    • {act}")
                if debt:
                    lines.append(f"    DEBT: {debt}")
                for pivot in pivots:
                    lines.append(f"    PIVOT: {pivot}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Assembly + persistence                                              #
    # ------------------------------------------------------------------ #

    def _assemble_and_write(
        self,
        conflict_map: dict[str, dict],
        blueprint: dict,
        total_episodes: int,
    ) -> dict:
        """
        Assemble the full drama structure graph and write it atomically.

        The episode_conflicts section contains the complete per-episode data
        (including structured_dialogues from cache), so the output file is
        the single authoritative artefact for downstream consumers.
        """
        # Sort episode conflicts by natural episode order
        episode_conflicts = {
            ep_id: conflict_map[ep_id]
            for ep_id in sorted(conflict_map, key=_natural_sort_key)
        }

        graph = {
            "total_episodes_analysed": len(conflict_map),
            "total_episodes_in_series": total_episodes,
            "macro_blueprint": blueprint,
            "episode_conflicts": episode_conflicts,
        }

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "drama_structure_graph.json"
        _atomic_json_write(out_path, graph)
        logger.info(
            "RhythmAnalyzer: drama_structure_graph.json written → %s  "
            "(%d episodes analysed)",
            out_path, len(conflict_map),
        )
        return graph
