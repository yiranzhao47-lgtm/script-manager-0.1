"""
Dry-run utility — pre-flight checks without processing any video.

Reports:
  • Language detection result across first N episodes
  • Estimated token budget per episode and total cost
  • GPU VRAM headroom vs. configured models
"""
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    with open("config/settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from src.utils.lang_detector import run_preflight

    video_dir = Path(cfg["paths"]["raw_video_dir"])
    report = run_preflight(cfg, video_dir)

    print(f"\n{'─'*60}")
    print(f"  Mode              : {report.mode_configured}")
    print(f"  Detected dominant : {report.detected_dominant}")
    print(f"  Overall CJK ratio : {report.overall_cjk_ratio:.1%}")
    print(f"  Match             : {'✓ OK' if report.match else '✗ MISMATCH'}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
