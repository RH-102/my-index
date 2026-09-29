import contextlib
import csv
import io
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import refresh_dashboard as refresh
import update_valuation as valuation


class ValuationTests(unittest.TestCase):
    def test_factset_parses_report_date_and_forward_not_trailing(self):
        text = ("EARNINGS INSIGHT September 25, 2026 "
                "Valuation: The forward 12 -month P/E ratio for the S&P 500 is 19.2. "
                "Trailing P/E 25.8")
        row = valuation.parse_factset(text, "https://example.com/report.pdf", date(2026, 9, 28))
        self.assertEqual(row["Date"], "2026-09-25")
        self.assertEqual(row["DateType"], "报告日期")
        self.assertEqual(row["ForwardPE"], 19.2)
        with self.assertRaises(ValueError):
            valuation.parse_factset(text.replace("19.2", "nan"), "", date(2026, 9, 28))
        with self.assertRaises(ValueError):
            valuation.parse_factset(text, "", date(2026, 9, 24))

    def ndx_payload(self):
        return {"label": "Nasdaq 100", "source": {"forwardOwnMethod": "FY1/FY2 consensus; ETF weights"},
                "forward": [{"date": "2026-08-05", "value": 22.37}],
                "updated": "2026-09-28", "current": {"forward": 22.37},
                "forwardOwn": [{"date": "2026-09-18", "value": 21.35, "basis": "dl-blended-fy1fy2"}]}

    def test_ndx_estimate_uses_observation_date_not_download_date_or_old_series(self):
        rows = valuation.parse_ndx(self.ndx_payload(), date(2026, 9, 28))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ForwardPE"], 21.35)
        self.assertEqual(rows[0]["Date"], "2026-09-18")
        self.assertEqual(rows[0]["Kind"], "ETF权重估算")

    def test_ndx_rejects_method_changes_and_future_or_invalid_values(self):
        for key, value in [("basis", "different-method"), ("date", "2026-09-29"),
                           ("value", float("inf")), ("value", 0)]:
            payload = self.ndx_payload()
            payload["forwardOwn"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                valuation.parse_ndx(payload, date(2026, 9, 28))

    def test_sources_never_overwrite_each_other_and_old_responses_do_not_regress_latest(self):
        cached = [{"SourceId": "a", "Date": "2026-09-25", "ForwardPE": 19.2},
                  {"SourceId": "b", "Date": "2026-09-25", "ForwardPE": 21.35}]
        fresh = [{"SourceId": "a", "Date": "2026-09-18", "ForwardPE": 19.1}]
        rows = valuation.merge_rows(cached, fresh)
        self.assertEqual(len(rows), 3)
        latest = max((row for row in rows if row["SourceId"] == "a"), key=lambda row: row["Date"])
        self.assertEqual(latest["ForwardPE"], 19.2)
        self.assertEqual([row["ForwardPE"] for row in rows if row["SourceId"] == "b"], [21.35])

    def test_failed_source_uses_its_own_cache_without_touching_model_history(self):
        ndx = valuation.parse_ndx(self.ndx_payload(), date(2026, 9, 28))
        sp = dict(ndx[0], SourceId=valuation.SOURCE_IDS[0], Index="S&P 500", ForwardPE=19.2,
                  Date="2026-09-25", Source="FactSet Earnings Insight", DateType="报告日期")
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            history, latest = data / "history.csv", data / "latest.csv"
            valuation.write_rows(history, [sp] + ndx)
            model = data / "risk_source_sp500_forward_pe.csv"
            model.write_text("immutable model history")
            with patch.multiple(valuation, DATA_DIR=data, HISTORY_FILE=history, LATEST_FILE=latest), \
                 patch.object(valuation, "fetch_factset", side_effect=RuntimeError("network error")), \
                 patch.object(valuation, "fetch_ndx", return_value=ndx), \
                 contextlib.redirect_stdout(io.StringIO()) as log:
                valuation.main()
            rows = list(csv.DictReader(io.StringIO(latest.read_text())))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["ForwardPE"], "19.2")
            self.assertIn("using local cache", log.getvalue())
            self.assertEqual(model.read_text(), "immutable model history")

    def test_stale_reference_remains_visible_in_status(self):
        rows = [{"Index": "Nasdaq-100", "Date": "2026-08-05"},
                {"Index": "S&P 500", "Date": "2026-09-25"}]
        with patch.object(refresh, "read_csv", return_value=rows):
            report = refresh.describe_module("valuation", {"state": "success"},
                                             datetime(2026, 9, 29, 1, tzinfo=timezone.utc), {})
        self.assertEqual(report["freshness"], "stale")
        self.assertEqual([row["stale"] for row in report["indicators"]], [True, False])


if __name__ == "__main__":
    unittest.main()
