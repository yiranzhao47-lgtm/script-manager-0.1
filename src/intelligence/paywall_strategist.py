"""
Paywall Strategy Report Generator.

Reads drama_structure_graph.json, extracts the macro blueprint and a
compressed slice of episode scene data, then calls Claude to produce a
Markdown paywall & marketing strategy report.

Entry point
───────────
    python pipeline.py --paywall-report

Output
──────
    data/output/paywall_strategy_report.md
    data/output/cost_report_paywall.json   (FinOps)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# Episodes to include in the compressed scene feed (paywall zone).
_SCENE_EP_RANGE = range(1, 26)  # ep01-25


class PaywallStrategist:
    """
    Generates an AI-driven paywall and marketing strategy report for a drama
    series by feeding drama_structure_graph.json to Claude.

    Parameters
    ----------
    cfg:
        Parsed config/settings.yaml dict.
    """

    def __init__(self, cfg: dict) -> None:
        paths = cfg.get("paths", {})
        self._output_dir = Path(paths.get("output_dir", "data/output"))
        self._meta_dir   = Path(paths.get("meta_dir",   "data/meta"))

        prompts_dir = Path("config/prompts")
        self._jinja = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=False,
            keep_trailing_newline=True,
        )

        from src.utils.llm_client import LLMClient
        self.llm_claude = LLMClient.from_cfg_key(cfg, "llm_claude")

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """
        Generate the paywall strategy report.

        Returns the path to the written Markdown file.
        Raises FileNotFoundError if drama_structure_graph.json is missing.
        """
        graph_path = self._output_dir / "drama_structure_graph.json"
        if not graph_path.exists():
            raise FileNotFoundError(
                f"drama_structure_graph.json not found at {graph_path}. "
                "Run the full pipeline with intelligence.drama_analysis.enabled: true first."
            )

        with graph_path.open(encoding="utf-8") as f:
            graph = json.load(f)

        drama_title  = self._resolve_title()
        macro        = graph.get("macro_blueprint", {})
        debt_chain   = macro.get("debt_chain_narrative", "")
        macro_ref    = {k: v for k, v in macro.items() if k != "debt_chain_narrative"}
        episode_scenes = self._extract_compressed_scenes(graph)

        prompt = self._render_prompt(drama_title, debt_chain, macro_ref, episode_scenes)
        system = (
            "You are a senior international short-drama executive producer "
            "and global growth director with deep expertise in narrative arc "
            "analysis and paywall monetization strategy."
        )

        logger.info(
            "PaywallStrategist — calling Claude  "
            "(~%d chars of context)", len(prompt)
        )
        raw = self.llm_claude.complete(
            system=system,
            user=prompt,
            module_name="Paywall_Strategy",
        )

        out_path = self._output_dir / "paywall_strategy_report.md"
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(raw.strip(), encoding="utf-8")
        tmp.replace(out_path)
        logger.info("Paywall strategy report written → %s", out_path)
        return out_path

    # ------------------------------------------------------------------ #
    #  Data extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_compressed_scenes(self, graph: dict) -> list[dict]:
        """
        Return a compressed representation of episode_conflicts for ep01-25.

        Keeps only scene_id, scene_actions, unresolved_debt, pivot_signals —
        drops location / time / structured_dialogues to stay within ~10K tokens.
        """
        ep_conflicts = graph.get("episode_conflicts", {})
        result = []
        for ep_num in _SCENE_EP_RANGE:
            ep_id = str(ep_num).zfill(2)
            if ep_id not in ep_conflicts:
                continue
            scenes = []
            for sc in ep_conflicts[ep_id].get("scenes", []):
                scenes.append({
                    "scene_id":       sc.get("scene_id", ""),
                    "scene_actions":  sc.get("scene_actions", []),
                    "unresolved_debt": sc.get("unresolved_debt", ""),
                    "pivot_signals":  sc.get("pivot_signals", []),
                })
            if scenes:
                result.append({"episode": ep_id, "scenes": scenes})
        return result

    def _resolve_title(self) -> str:
        """Read drama title from meta.json; fall back to generic label."""
        meta_path = self._meta_dir / "meta.json"
        if meta_path.exists():
            try:
                with meta_path.open(encoding="utf-8") as f:
                    meta = json.load(f)
                title = meta.get("title") or meta.get("drama_title") or ""
                if title:
                    return title
            except Exception:
                pass
        return "胜爱情战争"

    # ------------------------------------------------------------------ #
    #  Prompt rendering                                                    #
    # ------------------------------------------------------------------ #

    def _render_prompt(
        self,
        drama_title: str,
        debt_chain: str,
        macro_ref: dict,
        episode_scenes: list[dict],
    ) -> str:
        tmpl = self._jinja.get_template("paywall_strategy.j2")
        return tmpl.render(
            drama_title=drama_title,
            debt_chain_narrative=debt_chain,
            macro_blueprint_json=json.dumps(macro_ref, ensure_ascii=False, indent=2),
            episode_scenes_json=json.dumps(episode_scenes, ensure_ascii=False, indent=2),
        )
