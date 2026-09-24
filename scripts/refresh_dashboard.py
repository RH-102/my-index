"""Refresh independent dashboard modules and publish honest update status.

A failed module restores its last complete output set. Other modules and the
status banner can still be deployed. The workflow calls --check AFTER deployment
so a failed/stale refresh is visible both on the site and in GitHub Actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATUS_FILE = DATA_DIR / "update_status.json"
ET = ZoneInfo("America/New_York")
GROUPS = {
    "index": {
        "commands": ["validate_official_rebalances.py", "update_prices.py"],
        "outputs": ["holdings.csv", "latest.csv", "index_history.csv",
                    "holdings_history.csv", "divisor_history.csv"],
    },
    "benchmarks": {
        "commands": ["update_benchmarks.py"],
        "outputs": ["benchmark_history.csv"],
    },
    "risk": {
        "commands": ["update_risk_v3.py", "add_hy_oas.py", "add_ci_sloos.py",
                     "run_allocation_card.py", "snapshot_risk_dashboard.py"],
        "outputs": ["risk_dashboard.csv", "risk_history.csv",
                    "sp500_allocation_history.csv", "sp500_allocation_state.csv",
                    "risk_dashboard_history.csv"],
    },
}

# Monitoring thresholds, not claims about a source's publication schedule.
# Quarterly series deliberately allow their normal reporting lag.
RISK_MAX_AGE_DAYS = {
    "Nasdaq-100 Forward P/E": 14, "S&P 500 Forward P/E": 14,
    "Nasdaq Cushion": 14, "S&P 500 Cushion": 14,
    "DFII10 (10Y Real Yield)": 7, "HY OAS 3M Change": 7,
    "C&I SLOOS": 150, "Credit-to-GDP Gap": 210,
}


def read_csv(name: str) -> list[dict]:
    with (DATA_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def market_clock(now: datetime) -> dict:
    import pandas_market_calendars as mcal

    today = now.astimezone(ET).date()
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=today - timedelta(days=20), end_date=today + timedelta(days=20)
    )
    closes = [value.to_pydatetime() for value in schedule["market_close"]]
    # Allow 30 minutes for daily closes to become available. The calendar
    # handles weekends, holidays, early closes and daylight-saving changes.
    completed = [close for close in closes if close + timedelta(minutes=30) <= now]
    expected = completed[-1].astimezone(ET).date().isoformat()
    next_close = next(close for close in closes if close + timedelta(minutes=30) > now)
    return {
        "expected_market_date": expected,
        "next_update_due_at": (next_close + timedelta(hours=3)).isoformat(),
    }


def dated_rows(name: str, columns: list[str]) -> list[dict]:
    rows = read_csv(name)
    if not rows:
        raise ValueError(f"{name} has no rows")
    dates = [row["Date"] for row in rows]
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError(f"{name} has duplicate or unordered dates")
    for row in rows:
        datetime.strptime(row["Date"], "%Y-%m-%d")
        for column in columns:
            if not math.isfinite(float(row[column])):
                raise ValueError(f"{name} has a non-finite {column}")
    return rows


def validate_module(name: str) -> None:
    if name == "index":
        history = dated_rows("index_history.csv", ["IndexLevel", "DailyReturn", "Divisor"])
        latest = read_csv("latest.csv")
        if not latest or {row["PriceDate"] for row in latest} != {history[-1]["Date"]}:
            raise ValueError("Holdings and index dates do not agree")
        weights = [float(row["Weight"]) for row in latest]
        if not all(math.isfinite(v) and v > 0 for v in weights) or abs(sum(weights) - 1) > 1e-6:
            raise ValueError("Holdings weights are invalid")
        value = sum(float(row["MarketValue"]) for row in latest)
        level = value / float(history[-1]["Divisor"])
        if not math.isfinite(level) or not math.isclose(level, float(history[-1]["IndexLevel"]), rel_tol=1e-9):
            raise ValueError("Holdings value does not reproduce Index Level")
    elif name == "benchmarks":
        dated_rows("benchmark_history.csv", ["Nasdaq100Close", "SP500Close",
                                            "Nasdaq100ReturnPct", "SP500ReturnPct"])
    else:
        rows = read_csv("risk_dashboard.csv")
        required = set(RISK_MAX_AGE_DAYS) | {"S&P 500建议持仓"}
        if required - {row["Indicator"] for row in rows}:
            raise ValueError("The risk dashboard is missing required indicators")
        for row in rows:
            if row["Indicator"] in required and not row["CurrentValue"].strip():
                raise ValueError(f"Missing risk value: {row['Indicator']}")


def run_module(name: str, runner=subprocess.run) -> dict:
    group = GROUPS[name]
    backup = {file: (DATA_DIR / file).read_bytes() if (DATA_DIR / file).exists() else None
              for file in group["outputs"]}
    logs = []
    result = {"state": "success", "message": "更新完成"}
    try:
        for script in group["commands"]:
            print(f"Running {name}: {script}", flush=True)
            process = runner([sys.executable, str(ROOT / "scripts" / script)],
                             cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=600)
            output = process.stdout or ""
            logs.append(f"--- {script} (exit {process.returncode}) ---\n{output}")
            print(output, flush=True)
            if process.returncode:
                raise RuntimeError(f"{script} exited with code {process.returncode}")
        validate_module(name)
        fallback = [line.strip() for line in "\n".join(logs).splitlines()
                    if "using local cache" in line.lower()
                    or "preserving existing" in line.lower()]
        if fallback:
            result.update(state="cached", message="来源刷新未完成，部分数据使用缓存", details=fallback)
    except Exception as exc:
        for file, content in backup.items():
            path = DATA_DIR / file
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        result.update(state="failed", message="本次更新失败，已保留上一份完整数据", error=str(exc))
        print(f"::error::{name}: {exc}", flush=True)
        logs.append(str(exc))
    log_name = "risk_update.log" if name == "risk" else f"{name}_update.log"
    (DATA_DIR / log_name).write_text("\n".join(logs), encoding="utf-8")
    return result


def describe_module(name: str, result: dict, now: datetime, clock: dict) -> dict:
    result = dict(result)
    try:
        if name in {"index", "benchmarks"}:
            file = "index_history.csv" if name == "index" else "benchmark_history.csv"
            latest = max(row["Date"] for row in read_csv(file))
            result["data_date"] = latest
            result["freshness"] = "stale" if latest < clock["expected_market_date"] else "current"
        else:
            indicators = []
            for row in read_csv("risk_dashboard.csv"):
                label = row["Indicator"]
                if label not in RISK_MAX_AGE_DAYS:
                    continue
                dates = re.findall(r"\d{4}-\d{2}-\d{2}", row["DataDate"])
                as_of = min(dates) if dates else None
                age = (now.astimezone(ET).date() - datetime.strptime(as_of, "%Y-%m-%d").date()).days if as_of else None
                limit = RISK_MAX_AGE_DAYS[label]
                indicators.append({"indicator": label, "data_date": as_of,
                                   "age_days": age, "max_age_days": limit,
                                   "stale": age is None or age > limit})
            if not indicators:
                raise ValueError("No dated risk indicators")
            result["indicators"] = indicators
            result["freshness"] = "stale" if any(row["stale"] for row in indicators) else "current"
    except Exception as exc:
        result.update(freshness="unknown", message="数据状态无法核验", error=str(exc))
    return result


def needs_attention(report: dict) -> bool:
    for name in GROUPS:
        module = report.get("modules", {}).get(name, {})
        if module.get("state") != "success" or module.get("freshness") in {None, "unknown"}:
            return True
        if name != "risk" and module.get("freshness") != "current":
            return True
    # Old but successfully retrieved risk observations remain a visible warning,
    # not a fictitious download failure. They are refreshed on the next daily run.
    return False


def backup_needed(now: datetime) -> bool:
    try:
        report = json.loads(STATUS_FILE.read_text())
        return needs_attention(report) or report["expected_market_date"] != market_clock(now)["expected_market_date"]
    except (OSError, ValueError, KeyError):
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--backup-needed", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if args.backup_needed:
        print("true" if backup_needed(now) else "false")
        return
    if args.check:
        report = json.loads(STATUS_FILE.read_text())
        if needs_attention(report):
            raise SystemExit("Dashboard deployed with update failures or stale market data; see update_status.json and the site banner.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    market_clock(now)  # Check calendar availability before changing any outputs.
    results = {name: run_module(name) for name in GROUPS}
    now = datetime.now(timezone.utc)
    clock = market_clock(now)
    modules = {name: describe_module(name, results[name], now, clock) for name in GROUPS}
    report = {"generated_at": now.isoformat(), **clock, "modules": modules}
    temp = STATUS_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATUS_FILE)
    summary = ["## Dashboard update", f"Expected market close: {clock['expected_market_date']}"]
    for name, module in modules.items():
        summary.append(f"- {name}: {module['state']}; data {module['freshness']}")
        if module["freshness"] == "stale":
            print(f"::warning::{name}: older source observations; see dashboard status")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
