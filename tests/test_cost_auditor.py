"""
Unit tests for the FinOps monitoring module:
  • LLMClient._record_usage / get_ledger_data
  • CostAuditor.emit_financial_report (cost math, JSON output, print)

All tests are self-contained and use temporary directories; no live API calls.
"""
from __future__ import annotations

import json
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal config used across all tests
# ---------------------------------------------------------------------------
_CFG = {
    "pipeline": {"mode": "same_lang"},
    "paths": {
        "raw_video_dir": "data/raw",
        "cache_dir":     "data/cache",
        "meta_dir":      "data/meta",
        "output_dir":    "data/output",
    },
    "execution": {
        "llm": {
            "model":       "deepseek-chat",
            "base_url":    "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "max_tokens":  8000,
        },
        "retry": {
            "max_attempts":   1,
            "wait_min_sec":   0,
            "wait_max_sec":   0,
            "no_retry_codes": [400, 401],
        },
    },
    "pricing": {
        "deepseek-chat": {
            "input_cost_per_m":  1.0,
            "output_cost_per_m": 2.0,
        }
    },
}


def _make_llm_client():
    from src.utils.llm_client import LLMClient
    return LLMClient(_CFG)


# ===========================================================================
# Test: LLMClient ledger initialisation
# ===========================================================================

class TestLedgerInit(unittest.TestCase):

    def test_initial_ledger_structure(self):
        client = _make_llm_client()
        data = client.get_ledger_data()
        self.assertIn("total", data)
        self.assertIn("by_module", data)
        self.assertEqual(data["total"]["input_tokens"], 0)
        self.assertEqual(data["total"]["output_tokens"], 0)
        self.assertEqual(data["total"]["calls"], 0)
        self.assertEqual(data["by_module"], {})

    def test_get_ledger_returns_deep_copy(self):
        client = _make_llm_client()
        snap1 = client.get_ledger_data()
        snap1["total"]["input_tokens"] = 99999  # mutate the copy
        snap2 = client.get_ledger_data()
        self.assertEqual(snap2["total"]["input_tokens"], 0)  # original untouched


# ===========================================================================
# Test: _record_usage internal accumulation
# ===========================================================================

class TestRecordUsage(unittest.TestCase):

    def test_single_call_updates_total(self):
        client = _make_llm_client()
        client._record_usage("TestModule", 100, 50)
        data = client.get_ledger_data()
        self.assertEqual(data["total"]["input_tokens"], 100)
        self.assertEqual(data["total"]["output_tokens"], 50)
        self.assertEqual(data["total"]["calls"], 1)

    def test_single_call_creates_module_entry(self):
        client = _make_llm_client()
        client._record_usage("ROI_Auto_Heal", 200, 80)
        data = client.get_ledger_data()
        self.assertIn("ROI_Auto_Heal", data["by_module"])
        m = data["by_module"]["ROI_Auto_Heal"]
        self.assertEqual(m["input_tokens"], 200)
        self.assertEqual(m["output_tokens"], 80)
        self.assertEqual(m["calls"], 1)

    def test_multiple_calls_same_module_accumulate(self):
        client = _make_llm_client()
        client._record_usage("Subtitle_Refine", 1000, 400)
        client._record_usage("Subtitle_Refine", 2000, 600)
        data = client.get_ledger_data()
        m = data["by_module"]["Subtitle_Refine"]
        self.assertEqual(m["input_tokens"], 3000)
        self.assertEqual(m["output_tokens"], 1000)
        self.assertEqual(m["calls"], 2)

    def test_multiple_modules_tracked_independently(self):
        client = _make_llm_client()
        client._record_usage("Map_Extract",        5000, 1000)
        client._record_usage("Reduce_Canonicalize", 3000, 800)
        data = client.get_ledger_data()
        self.assertEqual(data["by_module"]["Map_Extract"]["input_tokens"], 5000)
        self.assertEqual(data["by_module"]["Reduce_Canonicalize"]["input_tokens"], 3000)
        self.assertEqual(data["total"]["input_tokens"], 8000)
        self.assertEqual(data["total"]["calls"], 2)

    def test_zero_tokens_still_increments_call_count(self):
        client = _make_llm_client()
        client._record_usage("default", 0, 0)
        data = client.get_ledger_data()
        self.assertEqual(data["total"]["calls"], 1)
        self.assertEqual(data["by_module"]["default"]["calls"], 1)


# ===========================================================================
# Test: complete() routes module_name into ledger via mocked API call
# ===========================================================================

class TestCompleteModuleNameWiring(unittest.TestCase):

    def _mock_resp(self, in_tok: int, out_tok: int, content: str = "ok"):
        resp = MagicMock()
        resp.choices[0].message.content = content
        resp.usage.prompt_tokens     = in_tok
        resp.usage.completion_tokens = out_tok
        return resp

    def test_module_name_default(self):
        client = _make_llm_client()
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = self._mock_resp(100, 40)
        client._openai_client = mock_openai

        client.complete(system="s", user="u")
        data = client.get_ledger_data()
        self.assertIn("default", data["by_module"])

    def test_module_name_custom(self):
        client = _make_llm_client()
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = self._mock_resp(500, 200)
        client._openai_client = mock_openai

        client.complete(system="s", user="u", module_name="Drama_Analysis")
        data = client.get_ledger_data()
        self.assertIn("Drama_Analysis", data["by_module"])
        self.assertEqual(data["by_module"]["Drama_Analysis"]["input_tokens"], 500)
        self.assertEqual(data["by_module"]["Drama_Analysis"]["output_tokens"], 200)

    def test_tokens_extracted_from_usage(self):
        client = _make_llm_client()
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = self._mock_resp(1234, 567)
        client._openai_client = mock_openai

        client.complete(system="s", user="u", module_name="Subtitle_Refine")
        data = client.get_ledger_data()
        self.assertEqual(data["total"]["input_tokens"], 1234)
        self.assertEqual(data["total"]["output_tokens"], 567)


# ===========================================================================
# Test: CostAuditor — cost calculation helper
# ===========================================================================

class TestCostCalculation(unittest.TestCase):

    def test_zero_tokens_zero_cost(self):
        from src.intelligence.cost_auditor import _calc_cost
        usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        self.assertEqual(_calc_cost(usage, 1.0, 2.0), 0.0)

    def test_one_million_input_tokens(self):
        from src.intelligence.cost_auditor import _calc_cost
        usage = {"input_tokens": 1_000_000, "output_tokens": 0, "calls": 1}
        self.assertAlmostEqual(_calc_cost(usage, 1.0, 2.0), 1.0, places=4)

    def test_one_million_output_tokens(self):
        from src.intelligence.cost_auditor import _calc_cost
        usage = {"input_tokens": 0, "output_tokens": 1_000_000, "calls": 1}
        self.assertAlmostEqual(_calc_cost(usage, 1.0, 2.0), 2.0, places=4)

    def test_mixed_tokens_correct_sum(self):
        from src.intelligence.cost_auditor import _calc_cost
        # 500k input @ ¥1/M = ¥0.5 ; 250k output @ ¥2/M = ¥0.5 ; total = ¥1.0
        usage = {"input_tokens": 500_000, "output_tokens": 250_000, "calls": 5}
        self.assertAlmostEqual(_calc_cost(usage, 1.0, 2.0), 1.0, places=4)

    def test_zero_rates_zero_cost(self):
        from src.intelligence.cost_auditor import _calc_cost
        usage = {"input_tokens": 999_999, "output_tokens": 888_888, "calls": 10}
        self.assertEqual(_calc_cost(usage, 0.0, 0.0), 0.0)


# ===========================================================================
# Test: CostAuditor.emit_financial_report — integration
# ===========================================================================

class TestCostAuditorReport(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _make_populated_client(self):
        client = _make_llm_client()
        client._record_usage("Subtitle_Refine",     345_678, 89_012)
        client._record_usage("Map_Extract",           23_456,  4_567)
        client._record_usage("Reduce_Canonicalize",    5_678,  1_234)
        client._record_usage("ROI_Auto_Heal",          1_234,    567)
        return client

    def test_json_report_written(self):
        from src.intelligence.cost_auditor import CostAuditor
        client = self._make_populated_client()
        with patch("builtins.print"):  # suppress terminal output in tests
            CostAuditor(client, output_dir=self.tmp).emit_financial_report(_CFG)
        self.assertTrue((self.tmp / "cost_report.json").exists())

    def test_json_report_structure(self):
        from src.intelligence.cost_auditor import CostAuditor
        client = self._make_populated_client()
        with patch("builtins.print"):
            CostAuditor(client, output_dir=self.tmp).emit_financial_report(_CFG)
        data = json.loads((self.tmp / "cost_report.json").read_text(encoding="utf-8"))
        self.assertIn("generated_at", data)
        self.assertIn("model", data)
        self.assertIn("by_module", data)
        self.assertIn("total", data)
        self.assertIn("cost_cny", data["total"])

    def test_total_cost_is_sum_of_modules(self):
        from src.intelligence.cost_auditor import CostAuditor
        client = self._make_populated_client()
        with patch("builtins.print"):
            CostAuditor(client, output_dir=self.tmp).emit_financial_report(_CFG)
        data = json.loads((self.tmp / "cost_report.json").read_text(encoding="utf-8"))
        module_cost_sum = sum(r["cost_cny"] for r in data["by_module"])
        self.assertAlmostEqual(module_cost_sum, data["total"]["cost_cny"], places=3)

    def test_rows_sorted_by_cost_descending(self):
        from src.intelligence.cost_auditor import CostAuditor
        client = self._make_populated_client()
        with patch("builtins.print"):
            CostAuditor(client, output_dir=self.tmp).emit_financial_report(_CFG)
        data = json.loads((self.tmp / "cost_report.json").read_text(encoding="utf-8"))
        costs = [r["cost_cny"] for r in data["by_module"]]
        self.assertEqual(costs, sorted(costs, reverse=True))

    def test_report_handles_missing_pricing_gracefully(self):
        from src.intelligence.cost_auditor import CostAuditor
        cfg_no_pricing = {**_CFG, "pricing": {}}
        client = _make_llm_client()
        client._record_usage("Subtitle_Refine", 1000, 500)
        with patch("builtins.print"):
            CostAuditor(client, output_dir=self.tmp).emit_financial_report(cfg_no_pricing)
        data = json.loads((self.tmp / "cost_report.json").read_text(encoding="utf-8"))
        self.assertEqual(data["total"]["cost_cny"], 0.0)

    def test_zero_usage_report_still_written(self):
        from src.intelligence.cost_auditor import CostAuditor
        client = _make_llm_client()  # no calls recorded
        with patch("builtins.print"):
            CostAuditor(client, output_dir=self.tmp).emit_financial_report(_CFG)
        data = json.loads((self.tmp / "cost_report.json").read_text(encoding="utf-8"))
        self.assertEqual(data["total"]["input_tokens"], 0)
        self.assertEqual(data["total"]["cost_cny"], 0.0)
        self.assertEqual(data["by_module"], [])

    def test_terminal_output_contains_model_name(self):
        from src.intelligence.cost_auditor import CostAuditor
        client = _make_llm_client()
        client._record_usage("Subtitle_Refine", 100, 50)
        buf = StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: buf.write(str(a[0]) + "\n")):
            CostAuditor(client, output_dir=self.tmp).emit_financial_report(_CFG)
        output = buf.getvalue()
        self.assertIn("deepseek-chat", output)

    def test_terminal_output_contains_module_name(self):
        from src.intelligence.cost_auditor import CostAuditor
        client = _make_llm_client()
        client._record_usage("Drama_Analysis", 9_000_000, 1_000_000)
        buf = StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: buf.write(str(a[0]) + "\n")):
            CostAuditor(client, output_dir=self.tmp).emit_financial_report(_CFG)
        output = buf.getvalue()
        self.assertIn("Drama_Analysis", output)

    def test_json_report_is_valid_utf8(self):
        from src.intelligence.cost_auditor import CostAuditor
        client = _make_llm_client()
        with patch("builtins.print"):
            CostAuditor(client, output_dir=self.tmp).emit_financial_report(_CFG)
        raw = (self.tmp / "cost_report.json").read_bytes()
        raw.decode("utf-8")  # must not raise


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
