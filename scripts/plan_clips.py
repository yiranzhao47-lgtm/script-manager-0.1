"""
Generate extended ~3-minute clip plans for marketing using LLM.

Usage:
    python scripts/plan_clips.py <drama_name>

Example:
    python scripts/plan_clips.py "dollar baby"

Reads:   data/output/<drama>/drama_structure_graph.json
Outputs: data/output/<drama>/clip_plans.json
"""
from __future__ import annotations

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


def _load_cfg() -> dict:
    with (_ROOT / "config" / "settings.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ts_to_sec(ts: str) -> float:
    """Convert HH:MM:SS,mmm to total seconds (best-effort)."""
    try:
        ts = ts.replace(",", ".")
        h, m, rest = ts.split(":")
        s = float(rest)
        return int(h) * 3600 + int(m) * 60 + s
    except Exception:
        return 0.0


def _build_scene_inventory(episode_conflicts: dict) -> str:
    """Compact scene inventory with explicit duration so LLM can sum correctly."""
    lines: list[str] = []
    for ep_id in sorted(episode_conflicts, key=lambda x: int(x) if x.isdigit() else x):
        scenes = episode_conflicts[ep_id].get("scenes", [])
        lines.append(f"=== EP{ep_id} ({len(scenes)} scenes) ===")
        for scene in scenes:
            sid   = scene.get("scene_id", "?")
            start = scene.get("scene_start_time", "?")
            end   = scene.get("scene_end_time", "?")
            dur   = max(0.0, _ts_to_sec(end) - _ts_to_sec(start))
            actions = scene.get("scene_actions", [])
            pivots  = scene.get("pivot_signals", [])
            lines.append(f"[{sid}] {start}→{end} ({dur:.0f}s)")
            for act in actions[:3]:
                lines.append(f"  • {act}")
            for pv in pivots[:2]:
                lines.append(f"  PIVOT: {pv}")
        lines.append("")
    return "\n".join(lines)


# ─── Post-processing: auto-extend short plans ─────────────────────────────────

def _build_ordered_scenes(episode_conflicts: dict) -> list[dict]:
    """Flat list of all scenes in chronological order with episode_id and duration."""
    ordered: list[dict] = []
    for ep_id in sorted(episode_conflicts, key=lambda x: int(x) if x.isdigit() else x):
        for scene in episode_conflicts[ep_id].get("scenes", []):
            start = scene.get("scene_start_time", "")
            end   = scene.get("scene_end_time", "")
            ordered.append({
                **scene,
                "episode_id":   ep_id,
                "duration_sec": max(0.0, _ts_to_sec(end) - _ts_to_sec(start)),
            })
    return ordered


def _plan_actual_sec(segments: list[dict]) -> float:
    return sum(
        max(0.0, _ts_to_sec(s.get("end_time", "")) - _ts_to_sec(s.get("start_time", "")))
        for s in segments
    )


def _extend_plan(plan: dict, ordered_scenes: list[dict], target_sec: int = 165) -> None:
    """
    Extend plan in-place by appending scenes until total video duration ≥ target_sec.

    Tracks actual VIDEO time (segment spans including inter-scene gaps).
    A scene is added even when it pushes total past target — the clip is not cut
    in the middle of a scene.  Only the hard ceiling (target + 35s) stops a scene
    from being included.  After reaching target, the next available pivot scene
    is used as the cliffhanger.
    """
    segments = plan.get("segments", [])
    if not segments:
        return

    committed_sec: float = _plan_actual_sec(segments)
    if committed_sec >= target_sec:
        return

    max_sec = target_sec + 50  # hard ceiling: never include a scene that would pass this

    # Anchor: last scene_id in the last segment
    last_seg  = segments[-1]
    last_sids = last_seg.get("scene_ids", [])
    last_sid  = last_sids[-1] if last_sids else ""

    start_pos = next(
        (i for i, sc in enumerate(ordered_scenes) if sc.get("scene_id") == last_sid),
        -1,
    )
    if start_pos == -1:
        last_ep  = last_seg.get("episode_id", "")
        last_end = _ts_to_sec(last_seg.get("end_time", ""))
        start_pos = next(
            (i for i, sc in enumerate(ordered_scenes)
             if sc["episode_id"] == last_ep
             and abs(_ts_to_sec(sc.get("scene_end_time", "")) - last_end) < 0.5),
            -1,
        )
    if start_pos == -1:
        return

    pending_ep:    str | None = None
    pending_start: str | None = None
    pending_end:   str | None = None
    pending_ids:   list[str]  = []
    cliffhanger:   str | None = None

    def _pending_span() -> float:
        if pending_start and pending_end:
            return max(0.0, _ts_to_sec(pending_end) - _ts_to_sec(pending_start))
        return 0.0

    following = ordered_scenes[start_pos + 1:]
    target_reached = False

    for i, sc in enumerate(following):
        ep_id    = sc["episode_id"]
        sc_start = sc.get("scene_start_time", "")
        sc_end   = sc.get("scene_end_time", "")
        sc_id    = sc.get("scene_id", "")

        # If target already reached, scan for the first cliffhanger pivot
        if target_reached:
            if sc.get("pivot_signals"):
                cliffhanger = sc_id
            break

        # Compute total video time if we include this scene
        if ep_id == pending_ep:
            new_span     = max(0.0, _ts_to_sec(sc_end) - _ts_to_sec(pending_start))
            total_if_add = committed_sec + new_span
        else:
            sc_span      = max(0.0, _ts_to_sec(sc_end) - _ts_to_sec(sc_start))
            total_if_add = committed_sec + _pending_span() + sc_span

        # Hard ceiling: skip this scene and stop
        if total_if_add > max_sec:
            cliffhanger = sc_id
            break

        # Flush pending when switching episodes
        if pending_ep is not None and ep_id != pending_ep:
            committed_sec += _pending_span()
            segments.append({
                "episode_id": pending_ep,
                "start_time": pending_start,
                "end_time":   pending_end,
                "scene_ids":  pending_ids[:],
                "note":       "[auto-extended]",
            })
            pending_ep = pending_start = pending_end = None
            pending_ids = []

        # Add scene to pending
        if pending_ep is None:
            pending_ep    = ep_id
            pending_start = sc_start
        pending_end = sc_end
        pending_ids.append(sc_id)

        # Check if target is now reached after adding this scene
        current_total = committed_sec + _pending_span()
        if current_total >= target_sec:
            target_reached = True  # next iteration scans for cliffhanger

    # Flush remaining pending
    if pending_ep is not None:
        committed_sec += _pending_span()
        segments.append({
            "episode_id": pending_ep,
            "start_time": pending_start,
            "end_time":   pending_end,
            "scene_ids":  pending_ids,
            "note":       "[auto-extended]",
        })

    if cliffhanger:
        plan["cliffhanger_scene_id"] = cliffhanger
        plan["cliffhanger_reason"]   = "[auto-extended: stops before next pivot scene]"

    plan["estimated_total_sec"] = round(committed_sec)
    plan["_auto_extended"]      = True


def _extend_short_plans(plans: dict, episode_conflicts: dict, target_sec: int = 165) -> int:
    """
    Extend all plans below target_sec. Returns count of plans extended.
    Strips any previously auto-extended segments before re-extending, so
    this function is safe to call multiple times on the same plans dict.
    """
    ordered_scenes = _build_ordered_scenes(episode_conflicts)
    extended = 0
    for plan in plans.get("clip_plans", []):
        # Strip segments that were added by a previous auto-extension run
        plan["segments"] = [
            s for s in plan.get("segments", [])
            if s.get("note") != "[auto-extended]"
        ]
        plan.pop("_auto_extended", None)

        actual = _plan_actual_sec(plan.get("segments", []))
        if actual < target_sec:
            _extend_plan(plan, ordered_scenes, target_sec)
            extended += 1
    return extended


def _build_clips_text(marketing_clips: list) -> str:
    """Compact text describing the marketing clip starting points."""
    lines: list[str] = []
    for i, clip in enumerate(marketing_clips, 1):
        ep      = clip.get("episode_id", "?")
        start   = clip.get("clip_start_time", "?")
        end     = clip.get("clip_end_time", "?")
        strategy = clip.get("mix_strategy", "?")
        a_start  = clip.get("action_start_focus", "")
        a_end    = clip.get("action_end_focus", "")
        lines.append(f"CLIP {i}: episode_id={ep}  start_time={start}  end_time={end}  [{strategy}]")
        lines.append(f"  Opening action : {a_start}")
        lines.append(f"  Closing action : {a_end}")
        lines.append("")
    return "\n".join(lines)


def _validate_plans(plans: dict, episode_conflicts: dict) -> list[str]:
    """Lightweight validation — return list of warning strings."""
    warnings: list[str] = []
    # Build scene timecode index for quick lookup
    valid_times: set[str] = set()
    for ep in episode_conflicts.values():
        for sc in ep.get("scenes", []):
            valid_times.add(sc.get("scene_start_time", ""))
            valid_times.add(sc.get("scene_end_time", ""))

    for plan in plans.get("clip_plans", []):
        cid = plan.get("clip_id", "?")
        prev_ep  = -1
        prev_end = -1.0
        for j, seg in enumerate(plan.get("segments", []), 1):
            ep_id = seg.get("episode_id", "")
            start = seg.get("start_time", "")
            end   = seg.get("end_time", "")
            ep_num = int(ep_id) if ep_id.isdigit() else -1

            if ep_num < prev_ep:
                warnings.append(
                    f"clip {cid} seg {j}: episode goes backward "
                    f"(ep{ep_id} after ep{prev_ep:02d})"
                )
            if start not in valid_times:
                warnings.append(
                    f"clip {cid} seg {j}: start_time '{start}' not in scene inventory"
                )
            if end not in valid_times:
                warnings.append(
                    f"clip {cid} seg {j}: end_time '{end}' not in scene inventory"
                )
            if _ts_to_sec(start) >= _ts_to_sec(end):
                warnings.append(
                    f"clip {cid} seg {j}: start >= end ({start} >= {end})"
                )
            prev_ep  = ep_num
            prev_end = _ts_to_sec(end)
    return warnings


def main(drama_name: str) -> int:
    cfg = _load_cfg()

    from src.utils.llm_client import LLMCallError, LLMClient, extract_json
    from jinja2 import Environment, FileSystemLoader

    llm = LLMClient(cfg)

    graph_path = _ROOT / "data" / "output" / drama_name / "drama_structure_graph.json"
    out_path   = _ROOT / "data" / "output" / drama_name / "clip_plans.json"

    if not graph_path.exists():
        logger.error("drama_structure_graph.json not found: %s", graph_path)
        return 1

    with graph_path.open(encoding="utf-8") as f:
        graph = json.load(f)

    marketing_clips   = graph.get("macro_blueprint", {}).get("marketing_clips", [])
    episode_conflicts = graph.get("episode_conflicts", {})

    scene_inventory = _build_scene_inventory(episode_conflicts)
    clips_text      = _build_clips_text(marketing_clips)

    logger.info("Clips: %d  |  Scene inventory: %d chars", len(marketing_clips), len(scene_inventory))

    jinja = Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    user_prompt = jinja.get_template("clip_planner.j2").render(
        drama_name=drama_name,
        n_clips=len(marketing_clips),
        clips_text=clips_text,
        scene_inventory=scene_inventory,
    )

    logger.info("Calling LLM for clip planning (~%d chars prompt)...", len(user_prompt))

    da_cfg = cfg.get("intelligence", {}).get("drama_analysis", {})
    max_tokens = int(da_cfg.get("reduce_max_tokens", 8000))

    try:
        raw = llm.complete(
            system=(
                "You are a professional Chinese short-drama marketing clip planner. "
                "Your response must be valid JSON only — no prose, no markdown fences."
            ),
            user=user_prompt,
            max_tokens=max_tokens,
            json_mode=True,
            module_name="Clip_Planning",
        )
    except LLMCallError as exc:
        logger.error("LLM call failed: %s", exc)
        return 1

    try:
        plans = extract_json(raw)
    except ValueError as exc:
        logger.error("LLM returned non-JSON: %s", exc)
        logger.debug("Raw response: %s", raw[:500])
        return 1

    # Validate
    warnings = _validate_plans(plans, episode_conflicts)
    if warnings:
        logger.warning("Validation warnings (%d):", len(warnings))
        for w in warnings:
            logger.warning("  • %s", w)

    # Auto-extend plans that are below the 165-second target
    n_extended = _extend_short_plans(plans, episode_conflicts, target_sec=165)
    if n_extended:
        logger.info("Auto-extended %d plan(s) to reach ~165s target.", n_extended)

    # Attach source clip metadata
    for i, plan in enumerate(plans.get("clip_plans", [])):
        if i < len(marketing_clips):
            plan["_source_clip"] = marketing_clips[i]

    out_path.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Clip plans saved → %s", out_path)

    # Summary table (show actual computed duration, not LLM estimate)
    clip_list = plans.get("clip_plans", [])
    print(f"\n{'#':>3}  {'Title':<35}  {'Segs':>4}  {'actual s':>8}  {'ext?':>4}  Cliffhanger")
    print("─" * 90)
    for plan in clip_list:
        cid   = plan.get("clip_id", "?")
        title = str(plan.get("clip_title", ""))[:35]
        segs  = len(plan.get("segments", []))
        actual = round(_plan_actual_sec(plan.get("segments", [])))
        ext   = "yes" if plan.get("_auto_extended") else "no"
        cliff = plan.get("cliffhanger_scene_id", "?")
        print(f"{str(cid):>3}  {title:<35}  {segs:>4}  {actual:>8}  {ext:>4}  {cliff}")
    print(f"\n{len(clip_list)} plans written. Review then run:")
    print(f"  python scripts/assemble_clips.py \"{drama_name}\"")
    return 0


def extend_only(drama_name: str) -> int:
    """Re-apply auto-extension to existing clip_plans.json without calling LLM."""
    graph_path = _ROOT / "data" / "output" / drama_name / "drama_structure_graph.json"
    out_path   = _ROOT / "data" / "output" / drama_name / "clip_plans.json"

    if not out_path.exists():
        logger.error("clip_plans.json not found. Run without --extend-only first.")
        return 1

    with graph_path.open(encoding="utf-8") as f:
        graph = json.load(f)
    with out_path.open(encoding="utf-8") as f:
        plans = json.load(f)

    episode_conflicts = graph.get("episode_conflicts", {})
    n_extended = _extend_short_plans(plans, episode_conflicts, target_sec=165)
    logger.info("Extended %d plan(s).", n_extended)

    out_path.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")

    clip_list = plans.get("clip_plans", [])
    print(f"\n{'#':>3}  {'Title':<35}  {'Segs':>4}  {'actual s':>8}  {'ext?':>4}  Cliffhanger")
    print("─" * 90)
    for plan in clip_list:
        cid   = plan.get("clip_id", "?")
        title = str(plan.get("clip_title", ""))[:35]
        segs  = len(plan.get("segments", []))
        actual = round(_plan_actual_sec(plan.get("segments", [])))
        ext   = "yes" if plan.get("_auto_extended") else "no"
        cliff = plan.get("cliffhanger_scene_id", "?")
        print(f"{str(cid):>3}  {title:<35}  {segs:>4}  {actual:>8}  {ext:>4}  {cliff}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plan_clips.py <drama_name> [--extend-only]")
        sys.exit(1)
    if "--extend-only" in sys.argv:
        sys.exit(extend_only(sys.argv[1]))
    sys.exit(main(sys.argv[1]))
