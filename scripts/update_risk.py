from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DASHBOARD_FILE = DATA_DIR / "risk_dashboard.csv"
HISTORY_FILE = DATA_DIR / "risk_history.csv"
TIMEZONE = ZoneInfo("America/New_York")

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
BIS_GAP_URL = (
    "https://stats.bis.org/api/v1/data/WS_CREDIT_GAP/"
    "Q.US.P.A.C/all?detail=full"
)
KOYFIN_URL = "https://app.koyfin.com/api/v3p/data/graph?schema=packed"
KOYFIN_HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://app.koyfin.com/",
    "Origin": "https://app.koyfin.com",
    "User-Agent": "Mozilla/5.0",
}

# ETF proxies for the two indices. Their holdings closely track the indices.
# Koyfin's f_pe field is 12-month forward P/E.
KOYFIN_QQQ_KID = "et-gpvivq"
KOYFIN_SPY_KID = "et-n5kqqt"

RISK_LABELS = {
    0: "🟢安全",
    1: "🟡注意",
    2: "🟠高风险",
    3: "🔴极高风险",
}

OUTPUT_COLUMNS = [
    "RowType",
    "Indicator",
    "CurrentValue",
    "HistoryPercentile",
    "RecentChange",
    "Rating",
    "RatingLevel",
    "DataDate",
    "Source",
    "Note",
    "UpdatedAt",
]


def get_json(url: str, payload: dict) -> dict:
    last_error = None
    for _ in range(3):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=KOYFIN_HEADERS,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Koyfin request failed: {last_error}")


def parse_koyfin_dates(values) -> pd.Series:
    raw = pd.Series(values)
    numeric = pd.to_numeric(raw, errors="coerce")

    if numeric.notna().mean() > 0.9:
        median = numeric.dropna().abs().median()
        if median > 1e12:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        if median > 1e9:
            return pd.to_datetime(numeric, unit="s", errors="coerce")

    return pd.to_datetime(raw, errors="coerce")


def fetch_koyfin_forward_pe(kid: str) -> pd.DataFrame:
    today = datetime.now(TIMEZONE).date().isoformat()
    payload = {
        "id": kid,
        "key": "f_pe",
        "currency": "USD",
        "financialPeriodType": "LTM",
        "priceFormat": "standard",
        "dateFrom": "1995-01-01",
        "dateTo": today,
    }

    data = get_json(KOYFIN_URL, payload)
    graph = data.get("graph", {})
    dates = graph.get("date", [])
    values = graph.get("value", [])

    if not dates or not values:
        raise RuntimeError(f"No forward P/E data returned for {kid}")

    frame = pd.DataFrame(
        {
            "Date": parse_koyfin_dates(dates),
            "Value": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna()

    frame = frame[(frame["Value"] > 1) & (frame["Value"] < 100)]
    frame = frame.sort_values("Date").drop_duplicates("Date", keep="last")

    if len(frame) < 100:
        raise RuntimeError(f"Too little forward P/E history for {kid}: {len(frame)} rows")

    return frame.reset_index(drop=True)


def fetch_fred(series_id: str) -> pd.DataFrame:
    response = requests.get(
        FRED_BASE.format(series_id),
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    if frame.shape[1] < 2:
        raise RuntimeError(f"Unexpected FRED response for {series_id}")

    frame = frame.iloc[:, :2].copy()
    frame.columns = ["Date", "Value"]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame.dropna().sort_values("Date")
    return frame.reset_index(drop=True)


def fetch_bis_credit_gap() -> pd.DataFrame:
    response = requests.get(
        BIS_GAP_URL,
        headers={
            "Accept": "application/vnd.sdmx.data+csv;version=1.0.0;labels=id",
            "User-Agent": "Mozilla/5.0",
        },
        timeout=60,
    )
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    upper = {str(c).upper(): c for c in frame.columns}

    time_col = upper.get("TIME_PERIOD") or upper.get("TIME_PERIOD_START")
    value_col = upper.get("OBS_VALUE")
    if time_col is None or value_col is None:
        raise RuntimeError(
            "Unexpected BIS columns: " + ", ".join(map(str, frame.columns))
        )

    def parse_quarter(value):
        text = str(value).strip()
        try:
            return pd.Period(text, freq="Q").end_time.normalize()
        except Exception:
            return pd.to_datetime(text, errors="coerce")

    out = pd.DataFrame(
        {
            "Date": frame[time_col].map(parse_quarter),
            "Value": pd.to_numeric(frame[value_col], errors="coerce"),
        }
    ).dropna()

    return out.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def percentile_rank(values: pd.Series, current: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty or pd.isna(current):
        return np.nan
    return float((clean <= current).mean() * 100)


def asof_value(frame: pd.DataFrame, target: pd.Timestamp) -> float:
    subset = frame[frame["Date"] <= target]
    if subset.empty:
        return np.nan
    return float(subset.iloc[-1]["Value"])


def changes_by_month(frame: pd.DataFrame, months=(1, 3, 6)) -> dict[int, float]:
    current_date = frame.iloc[-1]["Date"]
    current = float(frame.iloc[-1]["Value"])
    changes = {}
    for month in months:
        old = asof_value(frame, current_date - pd.DateOffset(months=month))
        changes[month] = current - old if not pd.isna(old) else np.nan
    return changes


def rolling_month_change(frame: pd.DataFrame, months: int) -> pd.Series:
    left = frame[["Date", "Value"]].copy()
    left["Lookup"] = left["Date"] - pd.DateOffset(months=months)
    left = left.sort_values("Lookup")

    right = frame[["Date", "Value"]].copy()
    right.columns = ["PastDate", "PastValue"]
    right = right.sort_values("PastDate")

    merged = pd.merge_asof(
        left,
        right,
        left_on="Lookup",
        right_on="PastDate",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    )
    return merged["Value"] - merged["PastValue"]


def cushion_history(pe: pd.DataFrame, real_yield: pd.DataFrame) -> pd.DataFrame:
    pe2 = pe[["Date", "Value"]].copy().sort_values("Date")
    pe2.columns = ["Date", "ForwardPE"]

    ry = real_yield[["Date", "Value"]].copy().sort_values("Date")
    ry.columns = ["RealYieldDate", "DFII10"]

    merged = pd.merge_asof(
        pe2,
        ry,
        left_on="Date",
        right_on="RealYieldDate",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    ).dropna()

    merged["EarningsYield"] = 100.0 / merged["ForwardPE"]
    merged["Cushion"] = merged["EarningsYield"] - merged["DFII10"]
    return merged


def risk_high_percentile(p: float) -> int:
    if pd.isna(p):
        return 1
    if p >= 95:
        return 3
    if p >= 85:
        return 2
    if p >= 65:
        return 1
    return 0


def risk_low_percentile(p: float) -> int:
    if pd.isna(p):
        return 1
    if p <= 5:
        return 3
    if p <= 15:
        return 2
    if p <= 35:
        return 1
    return 0


def cushion_absolute_risk(value: float) -> int:
    if value < 1:
        return 3
    if value < 2:
        return 2
    if value < 3:
        return 1
    return 0


def credit_gap_risk(value: float) -> int:
    if value > 10:
        return 3
    if value >= 5:
        return 2
    if value >= 2:
        return 1
    return 0


def trend3(values: list[float]) -> str:
    if len(values) < 3:
        return "数据不足"
    a, b, c = values[0], values[1], values[2]
    if a > b > c:
        return "连续上升"
    if a < b < c:
        return "连续下降"
    return "波动/未连续"


def mortgage_risk(values: pd.Series, yoy_pct: float, current_pct: float) -> int:
    if len(values) < 3:
        return 1
    current, prev, prev2 = map(float, values.iloc[-3:][::-1])
    rising = current > prev > prev2

    if rising and yoy_pct >= 95:
        return 3
    if rising and yoy_pct >= 85:
        return 2
    if rising or yoy_pct >= 75 or (current_pct >= 90 and current > prev):
        return 1
    return 0


def fmt_num(value: float, suffix="", digits=2, sign=False) -> str:
    if pd.isna(value):
        return "--"
    prefix = "+" if sign and value > 0 else ""
    return f"{prefix}{value:.{digits}f}{suffix}"


def fmt_percentile(value: float, qualifier="") -> str:
    if pd.isna(value):
        return "--"
    base = f"{value:.1f}%"
    return f"{qualifier}{base}" if qualifier else base


def metric_row(
    indicator: str,
    current: str,
    percentile: str,
    change: str,
    level: int,
    data_date: str,
    source: str,
    note: str,
    updated_at: str,
) -> dict:
    return {
        "RowType": "Metric",
        "Indicator": indicator,
        "CurrentValue": current,
        "HistoryPercentile": percentile,
        "RecentChange": change,
        "Rating": RISK_LABELS[level],
        "RatingLevel": level,
        "DataDate": data_date,
        "Source": source,
        "Note": note,
        "UpdatedAt": updated_at,
    }


def summary_row(indicator: str, level: int, note: str, updated_at: str) -> dict:
    return {
        "RowType": "Summary",
        "Indicator": indicator,
        "CurrentValue": RISK_LABELS[level],
        "HistoryPercentile": "",
        "RecentChange": "",
        "Rating": RISK_LABELS[level],
        "RatingLevel": level,
        "DataDate": "",
        "Source": "Composite",
        "Note": note,
        "UpdatedAt": updated_at,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    updated_at = datetime.now(TIMEZONE).isoformat(timespec="seconds")

    print("Fetching forward P/E history from Koyfin...")
    nasdaq_pe = fetch_koyfin_forward_pe(KOYFIN_QQQ_KID)
    sp_pe = fetch_koyfin_forward_pe(KOYFIN_SPY_KID)

    print("Fetching FRED DFII10 and mortgage delinquency...")
    real_yield = fetch_fred("DFII10")
    mortgage = fetch_fred("DRSFRMACBS")

    print("Fetching BIS US credit-to-GDP gap...")
    credit_gap = fetch_bis_credit_gap()

    ndx_current = float(nasdaq_pe.iloc[-1]["Value"])
    sp_current = float(sp_pe.iloc[-1]["Value"])
    ndx_pe_pct = percentile_rank(nasdaq_pe["Value"], ndx_current)
    sp_pe_pct = percentile_rank(sp_pe["Value"], sp_current)
    ndx_pe_changes = changes_by_month(nasdaq_pe, (1, 3))
    sp_pe_changes = changes_by_month(sp_pe, (1, 3))
    ndx_pe_risk = risk_high_percentile(ndx_pe_pct)
    sp_pe_risk = risk_high_percentile(sp_pe_pct)

    real_current = float(real_yield.iloc[-1]["Value"])
    real_changes = changes_by_month(real_yield, (1, 3, 6))
    real_3m_history = rolling_month_change(real_yield, 3)
    real_3m_pct = percentile_rank(real_3m_history, real_changes[3])
    real_risk = risk_high_percentile(real_3m_pct)

    ndx_cush = cushion_history(nasdaq_pe, real_yield)
    sp_cush = cushion_history(sp_pe, real_yield)

    ndx_cush_current = 100.0 / ndx_current - real_current
    sp_cush_current = 100.0 / sp_current - real_current
    ndx_cush_pct = percentile_rank(ndx_cush["Cushion"], ndx_cush_current)
    sp_cush_pct = percentile_rank(sp_cush["Cushion"], sp_cush_current)
    ndx_cush_risk = max(
        risk_low_percentile(ndx_cush_pct),
        cushion_absolute_risk(ndx_cush_current),
    )
    sp_cush_risk = max(
        risk_low_percentile(sp_cush_pct),
        cushion_absolute_risk(sp_cush_current),
    )

    ndx_cush_changes = changes_by_month(
        ndx_cush[["Date", "Cushion"]].rename(columns={"Cushion": "Value"}),
        (1, 3),
    )
    sp_cush_changes = changes_by_month(
        sp_cush[["Date", "Cushion"]].rename(columns={"Cushion": "Value"}),
        (1, 3),
    )

    cg_current = float(credit_gap.iloc[-1]["Value"])
    cg_prev = float(credit_gap.iloc[-2]["Value"]) if len(credit_gap) >= 2 else np.nan
    cg_year = float(credit_gap.iloc[-5]["Value"]) if len(credit_gap) >= 5 else np.nan
    cg_pct = percentile_rank(credit_gap["Value"], cg_current)
    cg_trend = trend3(
        [float(v) for v in credit_gap["Value"].tail(3).iloc[::-1].tolist()]
    )
    cg_risk = credit_gap_risk(cg_current)

    mort_current = float(mortgage.iloc[-1]["Value"])
    mort_prev = float(mortgage.iloc[-2]["Value"]) if len(mortgage) >= 2 else np.nan
    mort_prev2 = float(mortgage.iloc[-3]["Value"]) if len(mortgage) >= 3 else np.nan
    mort_year = float(mortgage.iloc[-5]["Value"]) if len(mortgage) >= 5 else np.nan
    mort_q1 = mort_current - mort_prev
    mort_q2 = mort_prev - mort_prev2
    mort_yoy = mort_current - mort_year
    mort_current_pct = percentile_rank(mortgage["Value"], mort_current)
    mort_yoy_history = mortgage["Value"].diff(4)
    mort_yoy_pct = percentile_rank(mort_yoy_history, mort_yoy)
    mort_trend = trend3([mort_current, mort_prev, mort_prev2])
    mort_risk_level = mortgage_risk(mortgage["Value"], mort_yoy_pct, mort_current_pct)

    tech_risk = max(ndx_pe_risk, ndx_cush_risk, real_risk)
    overall_risk = max(sp_pe_risk, sp_cush_risk, real_risk)

    ndx_date = nasdaq_pe.iloc[-1]["Date"].date().isoformat()
    sp_date = sp_pe.iloc[-1]["Date"].date().isoformat()
    real_date = real_yield.iloc[-1]["Date"].date().isoformat()
    cg_date = credit_gap.iloc[-1]["Date"].date().isoformat()
    mort_date = mortgage.iloc[-1]["Date"].date().isoformat()

    rows = [
        summary_row(
            "科技/AI估值风险",
            tech_risk,
            "综合 Nasdaq-100 Forward P/E、Nasdaq Cushion 与 DFII10 三个月变化。",
            updated_at,
        ),
        summary_row(
            "整体美股估值风险",
            overall_risk,
            "综合 S&P 500 Forward P/E、S&P 500 Cushion 与 DFII10 三个月变化。",
            updated_at,
        ),
        metric_row(
            "Nasdaq-100 Forward P/E",
            fmt_num(ndx_current, "x", 2),
            fmt_percentile(ndx_pe_pct),
            f"1M {fmt_num(ndx_pe_changes[1], 'x', 2, True)} | 3M {fmt_num(ndx_pe_changes[3], 'x', 2, True)}",
            ndx_pe_risk,
            ndx_date,
            "Koyfin QQQ proxy",
            "QQQ 12-month forward P/E is used as a Nasdaq-100 proxy.",
            updated_at,
        ),
        metric_row(
            "Nasdaq Cushion",
            fmt_num(ndx_cush_current, "%", 2),
            fmt_percentile(ndx_cush_pct),
            f"1M {fmt_num(ndx_cush_changes[1], 'pp', 2, True)} | 3M {fmt_num(ndx_cush_changes[3], 'pp', 2, True)}",
            ndx_cush_risk,
            f"PE {ndx_date}; DFII10 {real_date}",
            "Koyfin + FRED DFII10",
            "Earnings Yield = 100 / Forward P/E; Cushion = Earnings Yield - DFII10. Lower percentile means thinner cushion and more valuation risk.",
            updated_at,
        ),
        metric_row(
            "S&P 500 Forward P/E",
            fmt_num(sp_current, "x", 2),
            fmt_percentile(sp_pe_pct),
            f"1M {fmt_num(sp_pe_changes[1], 'x', 2, True)} | 3M {fmt_num(sp_pe_changes[3], 'x', 2, True)}",
            sp_pe_risk,
            sp_date,
            "Koyfin SPY proxy",
            "SPY 12-month forward P/E is used as an S&P 500 proxy.",
            updated_at,
        ),
        metric_row(
            "S&P 500 Cushion",
            fmt_num(sp_cush_current, "%", 2),
            fmt_percentile(sp_cush_pct),
            f"1M {fmt_num(sp_cush_changes[1], 'pp', 2, True)} | 3M {fmt_num(sp_cush_changes[3], 'pp', 2, True)}",
            sp_cush_risk,
            f"PE {sp_date}; DFII10 {real_date}",
            "Koyfin + FRED DFII10",
            "Absolute cushion guide: ≥4.5% wide/safe; 3–4.5% reasonable; 2–3% elevated; 1–2% high; <1% extreme. Historical percentile is primary.",
            updated_at,
        ),
        metric_row(
            "DFII10 (10Y Real Yield)",
            fmt_num(real_current, "%", 2),
            fmt_percentile(real_3m_pct, "3M变化 "),
            " | ".join(
                [
                    f"1M {fmt_num(real_changes[1] * 100, 'bp', 0, True)}",
                    f"3M {fmt_num(real_changes[3] * 100, 'bp', 0, True)}",
                    f"6M {fmt_num(real_changes[6] * 100, 'bp', 0, True)}",
                ]
            ),
            real_risk,
            real_date,
            "FRED DFII10",
            "评级主要看过去3个月实际收益率上升速度的历史百分位。",
            updated_at,
        ),
        metric_row(
            "Credit-to-GDP Gap",
            fmt_num(cg_current, "pp", 2),
            fmt_percentile(cg_pct),
            f"上季 {fmt_num(cg_prev, 'pp', 2)} | 1年前 {fmt_num(cg_year, 'pp', 2)} | {cg_trend}",
            cg_risk,
            cg_date,
            "BIS Q.US.P.A.C",
            "Total credit to the US private non-financial sector. Guide: <2 safe; 2–5 watch; 5–10 high; >10 very high.",
            updated_at,
        ),
        metric_row(
            "Mortgage Delinquency",
            fmt_num(mort_current, "%", 2),
            f"当前 {fmt_percentile(mort_current_pct)} | YoY速度 {fmt_percentile(mort_yoy_pct)}",
            " | ".join(
                [
                    f"最近Q {fmt_num(mort_q1, 'pp', 2, True)}",
                    f"前一Q {fmt_num(mort_q2, 'pp', 2, True)}",
                    f"YoY {fmt_num(mort_yoy, 'pp', 2, True)}",
                    mort_trend,
                ]
            ),
            mort_risk_level,
            mort_date,
            "FRED DRSFRMACBS",
            "重点看是否连续上升，以及YoY上升速度是否进入历史异常区间，而不是只看绝对逾期率。",
            updated_at,
        ),
    ]

    dashboard = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    dashboard.to_csv(DASHBOARD_FILE, index=False)

    ndx_hist = ndx_cush[["Date", "ForwardPE", "DFII10", "EarningsYield", "Cushion"]].copy()
    ndx_hist = ndx_hist.rename(
        columns={
            "ForwardPE": "NasdaqForwardPE",
            "EarningsYield": "NasdaqEarningsYield",
            "Cushion": "NasdaqCushion",
        }
    )
    sp_hist = sp_cush[["Date", "ForwardPE", "EarningsYield", "Cushion"]].copy()
    sp_hist = sp_hist.rename(
        columns={
            "ForwardPE": "SP500ForwardPE",
            "EarningsYield": "SP500EarningsYield",
            "Cushion": "SP500Cushion",
        }
    )

    history = pd.merge(ndx_hist, sp_hist, on="Date", how="outer").sort_values("Date")
    real_change_frame = real_yield[["Date"]].copy()
    real_change_frame["DFII10_3M_Change_bp"] = real_3m_history.values * 100
    history = pd.merge(history, real_change_frame, on="Date", how="outer")
    history = history.sort_values("Date")
    history["Date"] = history["Date"].dt.date.astype(str)
    history.to_csv(HISTORY_FILE, index=False)

    print("SUCCESS")
    print(f"Risk dashboard rows: {len(dashboard)}")
    print(f"Risk history rows: {len(history)}")
    print(f"Tech risk: {RISK_LABELS[tech_risk]}")
    print(f"Overall valuation risk: {RISK_LABELS[overall_risk]}")


if __name__ == "__main__":
    main()
