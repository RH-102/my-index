import contextlib
import io
import runpy
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import price_download as prices


class PriceDownloadTests(unittest.TestCase):
    def test_exchange_calendar_excludes_unfinished_sessions_and_holidays(self):
        self.assertEqual(prices.completed_market_dates(
            "2026-07-01", datetime.fromisoformat("2026-07-04T22:00:00+00:00")),
            ["2026-07-01", "2026-07-02"])
        for instant, expected in [("2026-11-27T18:29:00+00:00", "2026-11-25"),
                                  ("2026-11-27T18:31:00+00:00", "2026-11-27")]:
            self.assertEqual(prices.completed_market_dates(
                "2026-11-25", datetime.fromisoformat(instant))[-1], expected)

    def test_rebalance_close_requires_both_baskets_but_not_removed_stock_after(self):
        required = prices.required_price_dates(
            ["2026-09-21", "2026-09-22", "2026-09-23"],
            {"2026-09-21": ["NVDA", "AAPL"], "2026-09-22": ["NVDA", "MSFT"]})
        self.assertEqual(required["AAPL"], {"2026-09-21", "2026-09-22"})
        self.assertEqual(required["MSFT"], {"2026-09-22", "2026-09-23"})

    def test_invalid_prices_and_column_layouts(self):
        raw = pd.DataFrame({"Close": [100., float("nan"), float("inf"), 0., -1.]},
                           index=pd.date_range("2026-09-21", periods=5))
        expected = {"2026-09-21": 100.}
        self.assertEqual(prices.extract_closes(raw, "NVDA"), expected)
        ticker_first = pd.concat({"NVDA": raw}, axis=1)
        self.assertEqual(prices.extract_closes(ticker_first, "NVDA"), expected)
        self.assertEqual(prices.extract_closes(ticker_first.swaplevel(axis=1), "NVDA"), expected)

    def test_retries_only_missing_dates_and_preserves_existing_closes(self):
        history = {"NVDA": {"2026-09-25": 100.}, "AAPL": {"2026-09-28": 200.}}
        required = {"NVDA": {"2026-09-25", "2026-09-28"}, "AAPL": {"2026-09-28"}}
        loader = Mock(side_effect=[RuntimeError("temporary timeout"),
                                  {"2026-09-25": 999., "2026-09-28": 105.}])
        with contextlib.redirect_stdout(io.StringIO()):
            prices.fill_missing_prices(history, required, loader=loader, sleep=lambda _: None)
        self.assertEqual(history["NVDA"], {"2026-09-25": 100., "2026-09-28": 105.})
        self.assertEqual(loader.call_count, 2)
        for call in loader.call_args_list:
            self.assertEqual(call.args[:2], ("NVDA", ["2026-09-28"]))

    def test_stale_retries_fail_instead_of_carrying_forward_price(self):
        history = {"NVDA": {"2026-09-25": 100.}}
        loader = Mock(return_value={"2026-09-25": 100.})
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                RuntimeError, "NVDA: 2026-09-28"):
            prices.fill_missing_prices(history, {"NVDA": {"2026-09-28"}},
                                       loader=loader, sleep=lambda _: None)
        self.assertEqual(loader.call_count, 3)
        self.assertNotIn("2026-09-28", history["NVDA"])

    def run_index(self, recovered):
        # Reproduce the production bug: only NVDA's latest batch close is missing.
        dates = ["2026-08-12", "2026-09-22", "2026-09-25", "2026-09-28"]
        raw = pd.concat({
            "NVDA": pd.DataFrame({"Close": [100., 110., 120., float("nan")]}, index=pd.to_datetime(dates)),
            "AAPL": pd.DataFrame({"Close": [200., 220., 240., 250.]}, index=pd.to_datetime(dates)),
            "MSFT": pd.DataFrame({"Close": [400., 440., 480., 500.]}, index=pd.to_datetime(dates)),
        }, axis=1)
        original_fill = prices.fill_missing_prices
        loader = Mock(return_value=recovered)
        def fill(history, required):
            original_fill(history, required, loader=loader, sleep=lambda _: None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            data = root / "data"
            data.mkdir()
            shutil.copy(ROOT / "scripts/update_prices.py", root / "scripts/update_prices.py")
            (data / "rebalances.csv").write_text(
                "EffectiveDate,Symbol,Name,Quantity\n"
                "2026-08-12,NVDA,Nvidia,1\n2026-08-12,AAPL,Apple,1\n"
                "2026-09-22,NVDA,Nvidia,1\n2026-09-22,MSFT,Microsoft,1\n")
            names = ["holdings.csv", "latest.csv", "index_history.csv",
                     "holdings_history.csv", "divisor_history.csv"]
            for name in names:
                (data / name).write_text("PreviousCompleteData\n1\n")
            before = {name: (data / name).read_bytes() for name in names}
            with patch.object(prices, "completed_market_dates", return_value=dates), \
                 patch.object(prices.yf, "download", return_value=raw), \
                 patch.object(prices, "fill_missing_prices", side_effect=fill), \
                 contextlib.redirect_stdout(io.StringIO()):
                if not recovered:
                    with self.assertRaisesRegex(RuntimeError, "NVDA: 2026-09-28"):
                        runpy.run_path(str(root / "scripts/update_prices.py"))
                    self.assertEqual(before, {name: (data / name).read_bytes() for name in names})
                    return
                runpy.run_path(str(root / "scripts/update_prices.py"))
            history = pd.read_csv(data / "index_history.csv")
            self.assertEqual(history["Date"].tolist(), dates)
            # 330 / 3 = 110 before rebalance; new divisor = 550 / 110 = 5.
            self.assertAlmostEqual(history.iloc[1]["IndexLevel"], 110.)
            self.assertAlmostEqual(history.iloc[-1]["IndexLevel"], 125.)
            self.assertAlmostEqual(history.iloc[-1]["DailyReturn"], 125 / 120 - 1)
            latest = pd.read_csv(data / "latest.csv")
            self.assertEqual(set(latest["PriceDate"]), {"2026-09-28"})
            self.assertAlmostEqual(latest["Weight"].sum(), 1.)
            loader.assert_called_once_with("NVDA", ["2026-09-28"], 1)

    def test_missing_anchor_close_is_recovered_and_today_is_saved(self):
        self.run_index({"2026-09-28": 125.})

    def test_unavailable_close_leaves_all_previous_files_unchanged(self):
        self.run_index({})


if __name__ == "__main__":
    unittest.main()
