"""
Safely reset one episode's checkpoint state.

Usage:
    python scripts/reset_checkpoint.py <episode_id> <state>

Example:
    python scripts/reset_checkpoint.py 01 ocr_done
"""
import json
import sys
from pathlib import Path

VALID_STATES = ("pending", "asr_done", "ocr_done", "aligned", "refined", "complete")

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <episode_id> <state>")
        print(f"Valid states: {', '.join(VALID_STATES)}")
        sys.exit(1)

    ep_id = sys.argv[1].strip()
    state = sys.argv[2].strip()

    if state not in VALID_STATES:
        print(f"Error: unknown state '{state}'. Valid: {', '.join(VALID_STATES)}")
        sys.exit(1)

    path = Path("data/cache/checkpoint.json")
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
