"""
Generate Meta ad copy for assembled marketing clips.

Usage:
    python scripts/write_copy.py <drama_name>

Reads:   data/output/<drama>/clip_plans.json
         data/output/<drama>/drama_structure_graph.json
Outputs: data/output/<drama>/creatives/ad_copy.csv
           columns: clip_filename, primary_text, headline, description

One DeepSeek batch call for all clips — no per-clip API round-trips.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROMPTS_DIR = _ROOT / "config" / "prompts"

# Meta ad field hard limits (characters)
_MAX_PRIMARY  = 125
_MAX_HEADLINE = 27
_MAX_DESC     = 27


def _load_cfg() -> dict:
    with (_ROOT / "config" / "settings.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_name(s: str) -> str:
    """Mirror of assemble_clips._safe_name — must stay in sync."""
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in s).strip()[:35]


def _clip_filename(clip_id: int, clip_title: str) -> str:
    return f"ext_{str(clip_id).zfill(2)}_{_safe_name(clip_title)}.mp4"


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        logger.warning("  Field truncated from %d to %d chars: %r", len(text), limit, text[:30])
        return text[:limit].rstrip()
    return text


def _validate_copy(entries: list[dict]) -> list[dict]:
    """Enforce field length limits and strip line breaks from primary_text."""
    cleaned: list[dict] = []
    for entry in entries:
        primary = entry.get("primary_text", "").replace("\n", " ").strip()
        headline = entry.get("headline", "").strip()
        description = entry.get("description", "").strip()
        cleaned.append({
            "clip_id":     entry.get("clip_id"),
            "primary_text": _truncate(primary,  _MAX_PRIMARY),
            "headline":     _truncate(headline,  _MAX_HEADLINE),
            "description":  _truncate(description, _MAX_DESC),
        })
    return cleaned


def main(drama_name: str) -> int:
    cfg = _load_cfg()

    plans_path = _ROOT / "data" / "output" / drama_name / "clip_plans.json"
    graph_path = _ROOT / "data" / "output" / drama_name / "drama_structure_graph.json"
    out_dir    = _ROOT / "data" / "output" / drama_name / "creatives"
    csv_path   = out_dir / "ad_copy.csv"

    if not plans_path.exists():
        logger.error("clip_plans.json not found: %s", plans_path)
        return 1
    if not graph_path.exists():
        logger.error("drama_structure_graph.json not found: %s", graph_path)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    with plans_path.open(encoding="utf-8") as f:
        plans_data = json.load(f)
    with graph_path.open(encoding="utf-8") as f:
        graph = json.load(f)

    clip_plans: list[dict] = plans_data.get("clip_plans", [])
    if not clip_plans:
        logger.error("clip_plans.json contains no clip_plans entries")
        return 1

    synopsis_en: str = (
        graph.get("macro_blueprint", {}).get("synopsis_en", "")
        or "A Chinese short drama series."
    )

    # Build clip context list for the prompt
    clips_for_prompt: list[dict] = []
    for plan in clip_plans:
        src = plan.get("_source_clip", {})
        clips_for_prompt.append({
            "clip_id":            plan.get("clip_id"),
            "clip_title":         plan.get("clip_title", ""),
            "clip_zone":          src.get("clip_zone", ""),
            "mix_strategy":       src.get("mix_strategy", ""),
            "action_start_focus": src.get("action_start_focus", ""),
            "action_end_focus":   src.get("action_end_focus", ""),
            "contrast_rationale": src.get("contrast_rationale", ""),
        })

    logger.info(
        "Writing Meta ad copy — drama=%s  clips=%d",
        drama_name, len(clips_for_prompt),
    )

    from jinja2 import Environment, FileSystemLoader
    from src.utils.llm_client import LLMClient, LLMCallError, extract_json_array

    jinja = Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    user_prompt = jinja.get_template("meta_ad_copy.j2").render(
        synopsis_en=synopsis_en,
        clips=clips_for_prompt,
    )

    llm = LLMClient.from_cfg_key(cfg, "llm")  # DeepSeek

    da_cfg = cfg.get("intelligence", {}).get("drama_analysis", {})
    max_tokens = int(da_cfg.get("reduce_max_tokens", 8000))

    try:
        raw = llm.complete(
            system="You are a Meta advertising specialist for short drama streaming content.",
            user=user_prompt,
            max_tokens=max_tokens,
            json_mode=True,
            module_name="MetaAdCopy",
        )
    except LLMCallError as exc:
        logger.error("LLM call failed: %s", exc)
        return 1

    try:
        entries = extract_json_array(raw)
    except ValueError as exc:
        logger.error("LLM returned non-JSON-array: %s\nRaw: %s", exc, raw[:300])
        return 1

    if len(entries) != len(clip_plans):
        logger.warning(
            "LLM returned %d entries but expected %d — will match by clip_id",
            len(entries), len(clip_plans),
        )

    # Validate field lengths
    entries = _validate_copy(entries)

    # Index by clip_id for safe lookup
    copy_by_id: dict[int, dict] = {}
    for entry in entries:
        cid = entry.get("clip_id")
        if cid is not None:
            copy_by_id[int(cid)] = entry

    # Write CSV
    rows: list[dict] = []
    missing_ids: list[int] = []
    for plan in clip_plans:
        cid   = int(plan.get("clip_id", 0))
        title = plan.get("clip_title", "")
        fname = _clip_filename(cid, title)
        copy  = copy_by_id.get(cid)
        if copy is None:
            logger.warning("No ad copy returned for clip_id=%d (%s) — leaving empty", cid, fname)
            missing_ids.append(cid)
            rows.append({
                "clip_filename": fname,
                "primary_text":  "",
                "headline":      "",
                "description":   "",
            })
        else:
            rows.append({
                "clip_filename": fname,
                "primary_text":  copy["primary_text"],
                "headline":      copy["headline"],
                "description":   copy["description"],
            })

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_filename", "primary_text", "headline", "description"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("ad_copy.csv written: %s  (%d rows)", csv_path, len(rows))
    if missing_ids:
        logger.warning("Missing copy for clip_id(s): %s", missing_ids)

    # Print preview table
    print(f"\n{'Filename':<45}  {'Primary Text':<50}  Headline")
    print("─" * 110)
    for row in rows:
        fn  = row["clip_filename"][:45]
        pt  = row["primary_text"][:50]
        hl  = row["headline"]
        print(f"{fn:<45}  {pt:<50}  {hl}")

    return 0 if not missing_ids else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/write_copy.py <drama_name>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
