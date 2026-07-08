"""
FinOps Cost Auditor — token usage accounting and financial reporting.

Reads the in-memory ledger from a shared LLMClient instance, converts raw
token counts into CNY using the per-model pricing table in config/settings.yaml,
prints a formatted terminal report, and atomically writes cost_report.json to
data/output/.

Usage
─────
    from src.intelligence.cost_auditor import CostAuditor

    auditor = CostAuditor(llm_client, output_dir=Path("data/output"))
    auditor.emit_financial_report(cfg)
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Column widths for the terminal table
_W_MODULE = 22
_W_TOKENS = 14
_W_COST   = 12


class CostAuditor:
    """
    Single-responsibility financial auditor for one pipeline run.

    Parameters
    ----------
    llm_client:
        The shared ``LLMClient`` instance whose ledger holds all token usage
        accumulated during this run.
    output_dir:
        Directory where ``cost_report.json`` is written.
        Defaults to ``data/output``.
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        output_dir: Path = Path("data/output"),
    ) -> None:
        self._llm = llm_client
        self._output_dir = output_dir

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def emit_financial_report(self, config: dict) -> None:
        """
        Pull usage data from the LLMClient ledger, compute costs, print the
        terminal dashboard, and persist ``cost_report.json``.

        Parameters
        ----------
        config:
            Parsed ``config/settings.yaml`` dict.  Must contain a ``pricing``
            top-level key and an ``execution.llm.model`` key.
        """
        ledger  = self._llm.get_ledger_data()
        model   = config.get("execution", {}).get("llm", {}).get("model", "unknown")
        pricing = config.get("pricing", {}).get(model, {})

        in_rate  = float(pricing.get("input_cost_per_m",  0.0))
        out_rate = float(pricing.get("output_cost_per_m", 0.0))

        if not pricing:
            logger.warning(
                "CostAuditor: no pricing entry for model '%s' — costs will show ¥0.0000",
                model,
            )

        rows = self._build_rows(ledger, in_rate, out_rate)
        total_usage = ledger["total"]
        total_cost  = _calc_cost(total_usage, in_rate, out_rate)

        self._print_dashboard(model, rows, total_usage, total_cost)
        self._write_json_report(config, model, rows, total_usage, total_cost)

    # ------------------------------------------------------------------ #
    #  Row construction                                                    #
    # ------------------------------------------------------------------ #

    def _build_rows(
        self,
        ledger: dict,
        in_rate: float,
        out_rate: float,
    ) -> list[dict]:
        """Return per-module rows sorted by cost descending."""
        rows = []
        for module, usage in ledger["by_module"].items():
            rows.append({
                "module":        module,
                "input_tokens":  usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "calls":         usage["calls"],
                "cost_cny":      _calc_cost(usage, in_rate, out_rate),
            })
        rows.sort(key=lambda r: r["cost_cny"], reverse=True)
        return rows

    # ------------------------------------------------------------------ #
    #  Terminal dashboard                                                   #
    # ------------------------------------------------------------------ #

    def _print_dashboard(
        self,
        model: str,
        rows: list[dict],
        total: dict,
        total_cost: float,
    ) -> None:
        ts          = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_line = f"  FinOps Cost Report  |  {model}  |  {ts}"

        W = _W_MODULE + _W_TOKENS * 2 + _W_COST + 7
        bar     = "=" * W
        mid_bar = "-" * W

        col_hdr = (
            f"  {'Module':<{_W_MODULE - 2}}"
            f"| {'Input Tokens':>{_W_TOKENS - 1}}"
            f"| {'Output Tokens':>{_W_TOKENS - 1}}"
            f"| {'Cost (CNY)':>{_W_COST - 1}}"
        )

        lines = [bar, header_line, bar, col_hdr, mid_bar]

        if rows:
            for r in rows:
                lines.append(
                    f"  {r['module']:<{_W_MODULE - 2}}"
                    f"| {r['input_tokens']:>{_W_TOKENS - 1},}"
                    f"| {r['output_tokens']:>{_W_TOKENS - 1},}"
                    f"| {r['cost_cny']:>{_W_COST - 1}.4f}"
                )
        else:
            lines.append(f"  {'(no API calls recorded)':<{W - 2}}")

        lines += [
            mid_bar,
            f"  {'TOTAL':<{_W_MODULE - 2}}"
            f"| {total['input_tokens']:>{_W_TOKENS - 1},}"
            f"| {total['output_tokens']:>{_W_TOKENS - 1},}"
            f"| {total_cost:>{_W_COST - 1}.4f}",
            bar,
        ]

        report_str = "\n".join(lines)
        logger.info("\n%s", report_str)
        try:
            print(report_str, flush=True)
        except UnicodeEncodeError:
            print(report_str.encode("ascii", errors="replace").decode("ascii"), flush=True)

    # ------------------------------------------------------------------ #
    #  JSON persistence                                                    #
    # ------------------------------------------------------------------ #

    def _write_json_report(
        self,
        config: dict,
        model: str,
        rows: list[dict],
        total: dict,
        total_cost: float,
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "cost_report.json"

        payload = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "pricing_used": config.get("pricing", {}).get(model, {}),
            "by_module": rows,
            "total": {
                "input_tokens":  total["input_tokens"],
                "output_tokens": total["output_tokens"],
                "calls":         total["calls"],
                "cost_cny":      round(total_cost, 4),
            },
        }

        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        logger.info("CostAuditor: cost_report.json written → %s", out_path)


# ── Module-level helper ───────────────────────────────────────────────────────


def _calc_cost(usage: dict, in_rate: float, out_rate: float) -> float:
    """Compute CNY cost from token counts and per-million-token rates."""
    return round(
        usage["input_tokens"]  / 1_000_000 * in_rate
        + usage["output_tokens"] / 1_000_000 * out_rate,
        6,
    )
