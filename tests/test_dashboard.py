import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import update_risk as risk
import refresh_dashboard as refresh


class DashboardTests(unittest.TestCase):
    def test_cushion_current_and_changes_share_the_observation_date(self):
        dates = pd.date_range("2024-01-01", "2025-08-01", freq="MS")
        pe = pd.DataFrame({"Date": dates, "Value": [20 + i / 10 for i in range(len(dates))]})
        yields = pd.DataFrame({"Date": dates, "Value": [1 + i / 100 for i in range(len(dates))]})
        # A newer yield must not change a Cushion whose P/E is still from August.
        yields.loc[len(yields)] = [pd.Timestamp("2025-09-23"), 9.0]
        quarterly = pd.DataFrame({"Date": pd.date_range("2022-01-01", periods=12, freq="QS"),
                                  "Value": [2 + i / 10 for i in range(12)]})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dashboard.csv"
            with patch.multiple(risk, DATA_DIR=Path(temp), DASHBOARD_FILE=output,
                                HISTORY_FILE=Path(temp) / "history.csv"), \
                 patch.object(risk, "fetch_forward_pe", return_value=pe), \
                 patch.object(risk, "fetch_bis_credit_gap", return_value=quarterly), \
                 patch.object(risk, "fetch_fred", side_effect=lambda name: yields if name == "DFII10" else quarterly), \
                 contextlib.redirect_stdout(io.StringIO()):
                risk.main()
            rows = pd.read_csv(output).set_index("Indicator")
        current = 100 / 21.9 - 1.19
        prior_month = 100 / 21.8 - 1.18
        for name in ["Nasdaq Cushion", "S&P 500 Cushion"]:
            row = rows.loc[name]
            self.assertEqual(row["CurrentValue"], f"{current:.2f}%")
            self.assertIn("PE 2025-08-01; DFII10 2025-08-01", row["DataDate"])
            self.assertIn(f"1M {current - prior_month:+.2f}pp", row["RecentChange"])

    def test_cushion_never_uses_future_or_excessively_old_yield(self):
        pe = pd.DataFrame({"Date": pd.to_datetime(["2025-05-01", "2025-06-01"]), "Value": [20., 20.]})
        yields = pd.DataFrame({"Date": pd.to_datetime(["2025-04-30", "2025-05-02"]), "Value": [2., 9.]})
        result = risk.cushion_history(pe, yields)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Cushion"], 3.)

    def test_failed_module_restores_complete_previous_output(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "old.csv").write_text("valid previous data")
            groups = {"index": {"commands": ["broken.py"], "outputs": ["old.csv", "new.csv"]}}
            def broken(*args, **kwargs):
                (data / "old.csv").write_text("partial replacement")
                (data / "new.csv").write_text("partial new file")
                return subprocess.CompletedProcess(args[0], 1, "provider failed")
            with patch.multiple(refresh, DATA_DIR=data, GROUPS=groups), contextlib.redirect_stdout(io.StringIO()):
                result = refresh.run_module("index", runner=broken)
            self.assertEqual(result["state"], "failed")
            self.assertEqual((data / "old.csv").read_text(), "valid previous data")
            self.assertFalse((data / "new.csv").exists())

    def test_backup_retries_risk_failure_even_if_market_dates_are_current(self):
        report = {"modules": {name: {"state": "success", "freshness": "current"} for name in refresh.GROUPS}}
        self.assertFalse(refresh.needs_attention(report))
        report["modules"]["risk"]["freshness"] = "stale"
        self.assertFalse(refresh.needs_attention(report))
        report["modules"]["risk"]["state"] = "cached"
        self.assertTrue(refresh.needs_attention(report))
        report["modules"]["risk"]["state"] = "failed"
        self.assertTrue(refresh.needs_attention(report))

    def test_quarterly_release_lag_is_not_treated_as_daily_staleness(self):
        rows = [{"Indicator": "Credit-to-GDP Gap", "DataDate": "2026-03-31"},
                {"Indicator": "C&I SLOOS", "DataDate": "2026-07-01"},
                {"Indicator": "S&P 500 Forward P/E", "DataDate": "2026-08-05"}]
        now = datetime(2026, 9, 24, 2, tzinfo=timezone.utc)
        with patch.object(refresh, "read_csv", return_value=rows):
            status = refresh.describe_module("risk", {"state": "success"}, now, {})
        self.assertEqual([r["stale"] for r in status["indicators"]], [False, False, True])
        self.assertEqual(status["indicators"][-1]["age_days"], 49)

    def test_market_calendar_handles_holidays_early_close_and_dst(self):
        cases = [
            ("2026-07-04T22:00:00+00:00", "2026-07-02"),
            ("2026-11-27T18:29:00+00:00", "2026-11-25"),
            ("2026-11-27T18:31:00+00:00", "2026-11-27"),
            ("2026-03-09T20:29:00+00:00", "2026-03-06"),
            ("2026-03-09T20:31:00+00:00", "2026-03-09"),
        ]
        for instant, expected in cases:
            with self.subTest(instant=instant):
                self.assertEqual(refresh.market_clock(datetime.fromisoformat(instant))["expected_market_date"], expected)


if __name__ == "__main__":
    unittest.main()
