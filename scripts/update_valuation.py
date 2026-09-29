"""Refresh recent P/E references without splicing incompatible histories.

FactSet's dated S&P 500 consensus report and History of Market's Nasdaq-100
ETF-weighted estimate have independent series IDs. Neither replaces the older
terminal series used by the historical risk/portfolio model.
"""

import csv
import math
import re
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from pypdf import PdfReader

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HISTORY_FILE = DATA_DIR / "valuation_reference_history.csv"
LATEST_FILE = DATA_DIR / "valuation_latest.csv"
ET = ZoneInfo("America/New_York")
FACTSET_ROOT = ("https://advantage.factset.com/hubfs/Website/Resources%20Section/"
                "Research%20Desk/Earnings%20Insight/")
NDX_URL = "https://historyofmarket.com/api/ndx/forward-pe.json"
SOURCE_IDS = ("factset-sp500-ntm", "hom-ndx-etf-fy1fy2")
COLUMNS = ["SourceId", "Index", "Date", "DateType", "ForwardPE", "Kind",
           "Source", "SourceURL", "Method", "FetchedAt"]
MONTH_DATE = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}"


def parse_factset(text, url, today):
    text = re.sub(r"\s+", " ", text).replace("–", "-").replace("−", "-")
    date_match = re.search(r"EARNINGS INSIGHT\s+(" + MONTH_DATE + r")", text)
    value_match = re.search(
        r"Valuation:\s+The forward\s+12\s*-\s*month P/E ratio for the S&P 500 is\s+(\d+(?:\.\d+)?)",
        text, re.IGNORECASE)
    if not date_match or not value_match:
        raise ValueError("FactSet report date or S&P 500 forward 12-month P/E not found")
    report_date = datetime.strptime(date_match[1], "%B %d, %Y").date()
    value = float(value_match[1])
    if report_date > today or not 1 < value < 100:
        raise ValueError("Invalid or future-dated FactSet observation")
    return {"SourceId": SOURCE_IDS[0], "Index": "S&P 500", "Date": report_date.isoformat(),
            "DateType": "报告日期", "ForwardPE": value, "Kind": "未来12个月一致预期（周报）",
            "Source": "FactSet Earnings Insight", "SourceURL": url,
            "Method": "FactSet 周报原值；日期为报告发布日期，不代表当天收盘后的实时估值。"}


def fetch_factset(today, session):
    friday = today - timedelta(days=(today.weekday() - 4) % 7)
    errors = []
    # Thursday covers holiday publication shifts; dated PDFs also avoid stale
    # redirects from a generic latest-report URL. Do not infer a report date.
    for week in range(4):
        for offset in (0, 1):
            date = friday - timedelta(days=7 * week + offset)
            url = FACTSET_ROOT + f"EarningsInsight_{date:%m%d%y}.pdf"
            for attempt in range(2):
                try:
                    response = session.get(url, timeout=(8, 25))
                    if response.status_code == 404:
                        break
                    response.raise_for_status()
                    if not response.content.startswith(b"%PDF"):
                        raise ValueError("Expected a PDF report")
                    reader = PdfReader(BytesIO(response.content))
                    row = parse_factset(reader.pages[0].extract_text(), url, today)
                    if row["Date"] != date.isoformat():
                        raise ValueError("PDF publication date differs from requested archive date")
                    return [row]
                except Exception as exc:
                    errors.append(str(exc))
                    print(f"FactSet {date} attempt {attempt + 1}: {exc}", flush=True)
                    if attempt == 0:
                        time.sleep(1)
    raise RuntimeError("No valid recent FactSet report: " + "; ".join(errors[-2:]))


def parse_ndx(payload, today):
    if "Nasdaq 100" not in str(payload.get("label", "")):
        raise ValueError("Unexpected index in Nasdaq P/E response")
    method = str(payload.get("source", {}).get("forwardOwnMethod", ""))
    if "FY1/FY2" not in method or "ETF" not in method:
        raise ValueError("Nasdaq estimate methodology changed or is missing")
    points = payload.get("forwardOwn", [])
    if not points:
        raise ValueError("No independent Nasdaq forward estimate series")
    rows = []
    for point in points:
        date = datetime.strptime(point["date"], "%Y-%m-%d").date()
        value = float(point["value"])
        if point.get("basis") != "dl-blended-fy1fy2":
            raise ValueError("Unknown Nasdaq estimate basis")
        if date > today or not math.isfinite(value) or not 1 < value < 100:
            raise ValueError("Invalid or future-dated Nasdaq estimate")
        rows.append({"SourceId": SOURCE_IDS[1], "Index": "Nasdaq-100", "Date": date.isoformat(),
                     "DateType": "观测日期", "ForwardPE": value, "Kind": "ETF权重估算",
                     "Source": "History of Market · 美股编年史", "SourceURL": NDX_URL,
                     "Method": "使用ETF公开权重及FY1/FY2盈利一致预期估算未来12个月P/E；与终端历史序列分开记录。"})
    return rows


def fetch_ndx(today, session):
    response = session.get(NDX_URL, timeout=(8, 20))
    response.raise_for_status()
    return parse_ndx(response.json(), today)


def merge_rows(cached, fresh):
    # Source identity is part of the key: a newer proxy never overwrites the
    # other source's observation or the model's risk_source_* files.
    by_key = {(row["SourceId"], row["Date"]): row for row in cached}
    by_key.update({(row["SourceId"], row["Date"]): row for row in fresh})
    return [by_key[key] for key in sorted(by_key)]


def write_rows(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    now = datetime.now(ET)
    cached = []
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open(newline="", encoding="utf-8") as handle:
            cached = list(csv.DictReader(handle))
    fresh = []
    with requests.Session() as session:
        session.headers["User-Agent"] = "Mozilla/5.0 (compatible; my-index/1.0)"
        for source_id, fetcher in zip(SOURCE_IDS, (fetch_factset, fetch_ndx)):
            try:
                rows = fetcher(now.date(), session)
                for row in rows:
                    row["FetchedAt"] = now.isoformat(timespec="seconds")
                fresh.extend(rows)
            except Exception as exc:
                previous = [row for row in cached if row["SourceId"] == source_id]
                if not previous:
                    raise RuntimeError(f"{source_id}: no valid cached reference: {exc}") from exc
                print(f"{source_id}: {exc}; using local cache through "
                      f"{max(row['Date'] for row in previous)}", flush=True)
    history = merge_rows(cached, fresh)
    latest = [max((row for row in history if row["SourceId"] == source_id),
                  key=lambda row: row["Date"]) for source_id in SOURCE_IDS]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(HISTORY_FILE, history)
    write_rows(LATEST_FILE, latest)
    for row in latest:
        print(f"{row['Index']}: {float(row['ForwardPE']):.2f}x; {row['DateType']} "
              f"{row['Date']}; {row['Source']} ({row['Kind']})")


if __name__ == "__main__":
    main()
