"""
Safely reset one episode's checkpoint state.

Usage:
    python scripts/reset_checkpoint.py <drama_name> <episode_id> <state>

Example:
    python scripts/reset_checkpoint.py "dollar baby" 01 ocr_done
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

VALID_STATES = ("pending", "asr_done", "ocr_done", "aligned", "refined", "complete")

def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <drama_name> <episode_id> <state>")
        print(f"Valid states: {', '.join(VALID_STATES)}")
        sys.exit(1)

    drama_name = sys.argv[1].strip()
    ep_id      = sys.argv[2].strip()
    state      = sys.argv[3].strip()

    if state not in VALID_STATES:
        print(f"Error: unknown state '{state}'. Valid: {', '.join(VALID_STATES)}")
        sys.exit(1)

    path = _ROOT / "data" / "cache" / drama_name / "checkpoint.json"
    if not path.exists():
        print(f"Error: checkpoint file not found at {path}")
        sys.exit(1)

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    old_state = data.get("episodes", {}).get(ep_id, "pending")
    data.setdefault("episodes", {})[ep_id] = state

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

    print(f"[{ep_id}]  {old_state}  →  {state}")

if __name__ == "__main__":
    main()
