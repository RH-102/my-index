"""Daily V3.1-D S&P 500 suggested stock-allocation model.

The model follows the user's V3.1-D rules and writes one Allocation row into
risk_dashboard.csv so the homepage can display it as a third summary card.

Important implementation detail:
- S&P 500 consensus Forward P/E updates less frequently than the index price.
- On each consensus P/E observation date, implied Forward EPS = SPX / Forward P/E.
- Between consensus updates, current estimated Forward P/E = latest SPX close /
  latest implied consensus Forward EPS. This lets valuation move daily with the
  market without inventing new earnings estimates.

All slower-moving series use the latest published observation and are cached in
this repository. No 5% rounding is applied to the final allocation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TIMEZONE = ZoneInfo("America/New_York")

DASHBOARD_FILE = DATA_DIR / "risk_dashboard.csv"
SP_PE_FILE = DATA_DIR / "risk_source_sp500_forward_pe.csv"
REAL_YIELD_FILE = DATA_DIR / "risk_source_dfii10.csv"
HY_OAS_FILE = DATA_DIR / "risk_source_hy_oas.csv"
SLOOS_FILE = DATA_DIR / "risk_source_ci_sloos.csv"
CREDIT_GAP_FILE = DATA_DIR / "risk_source_credit_gap.csv"
SPX_PRICE_FILE = DATA_DIR / "risk_source_sp500_price.csv"
STATE_FILE = DATA_DIR / "sp500_allocation_state.csv"
HISTORY_FILE = DATA_DIR / "sp500_allocation_history.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; my-index/1.0)"}
STOOQ_URL = "https://stooq.com/q/d/l/?s=%5Espx&i=d"

OUTPUT_COLUMNS = [
    "RowType", "Indicator", "CurrentValue", "HistoryPercentile",
    "RecentChange", "Rating", "RatingLevel", "DataDate", "Source",
    "Note", "UpdatedAt",
]


def clip(value: float, low=0.0, high=1.0) -> float:
    return float(max(low, min(high, value)))


def load_series(path: Path, min_rows: int = 1) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Required local data file is missing: {path.name}")
    frame = pd.read_csv(path)
    if len(frame.columns) < 2:
        raise RuntimeError(f"Unexpected format in {path.name}")
    out = frame.iloc[:, :2].copy()
    out.columns = ["Date", "Value"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna().sort_values("Date").drop_duplicates("Date", keep="last")
    out = out.reset_index(drop=True)
    if len(out) < min_rows:
        raise RuntimeError(f"Not enough observations in {path.name}")
    return out


def save_series(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["Date", "Value"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna().sort_values("Date").drop_duplicates("Date", keep="last")
    disk = out.copy()
    disk["Date"] = disk["Date"].dt.strftime("%Y-%m-%d")
    disk.to_csv(path, index=False)
    return out.reset_index(drop=True)


def extract_yfinance_close(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "Value"])

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame(columns=["Date", "Value"])
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame(columns=["Date", "Value"])
        close = raw["Close"]

    out = pd.DataFrame({
        "Date": pd.to_datetime(close.index, errors="coerce").tz_localize(None),
        "Value": pd.to_numeric(close.values, errors="coerce"),
    })
    return out.dropna().sort_values("Date").reset_index(drop=True)


def fetch_spx_yahoo(start: str) -> pd.DataFrame:
    raw = yf.download(
        "^GSPC",
        start=start,
        end=(datetime.now().date() + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    out = extract_yfinance_close(raw)
    if out.empty:
        raise RuntimeError("Yahoo returned no S&P 500 prices")
    return out


def fetch_spx_stooq() -> pd.DataFrame:
    response = requests.get(STOOQ_URL, headers=HEADERS, timeout=(6, 20))
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    cols = {str(c).lower(): c for c in raw.columns}
    if "date" not in cols or "close" not in cols:
        raise RuntimeError("Unexpected Stooq S&P 500 CSV format")
    out = raw[[cols["date"], cols["close"]]].copy()
    out.columns = ["Date", "Value"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    return out.dropna().sort_values("Date").reset_index(drop=True)


def refresh_spx_prices() -> pd.DataFrame:
    cached = None
    if SPX_PRICE_FILE.exists():
        try:
            cached = load_series(SPX_PRICE_FILE, 50)
        except Exception:
            cached = None

    start = "1989-01-01" if cached is None else (
        cached.iloc[-1]["Date"] - pd.Timedelta(days=15)
    ).date().isoformat()

    fresh = None
    errors = []
    try:
        fresh = fetch_spx_yahoo(start)
    except Exception as exc:
        errors.append(f"Yahoo: {exc}")

    if fresh is None or fresh.empty:
        try:
            full = fetch_spx_stooq()
            fresh = full[full["Date"] >= pd.Timestamp(start)].copy()
            if fresh.empty:
                fresh = full
        except Exception as exc:
            errors.append(f"Stooq: {exc}")

    if fresh is None or fresh.empty:
        if cached is not None:
            print("SPX refresh failed; using local cache. " + "; ".join(errors))
            return cached
        raise RuntimeError("Could not obtain S&P 500 prices: " + "; ".join(errors))

    combined = fresh if cached is None else pd.concat([cached, fresh], ignore_index=True)
    out = save_series(SPX_PRICE_FILE, combined)
    print(
        f"Saved {SPX_PRICE_FILE.name}: {len(out)} observations through "
        f"{out.iloc[-1]['Date'].date()}"
    )
    return out


def asof_value(frame: pd.DataFrame, target: pd.Timestamp) -> float:
    subset = frame[frame["Date"] <= target]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[-1]["Value"])


def implied_forward_eps(pe: pd.DataFrame, spx: pd.DataFrame) -> pd.DataFrame:
    left = pe[["Date", "Value"]].copy().sort_values("Date")
    left.columns = ["Date", "ForwardPE"]
    right = spx[["Date", "Value"]].copy().sort_values("Date")
    right.columns = ["PriceDate", "SPX"]

    merged = pd.merge_asof(
        left,
        right,
        left_on="Date",
        right_on="PriceDate",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    ).dropna()
    merged["ForwardEPS"] = merged["SPX"] / merged["ForwardPE"]
    return merged


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"InFreeze": False, "R15": np.nan, "FreezeStartDate": ""}
    try:
        row = pd.read_csv(STATE_FILE).iloc[-1]
        return {
            "InFreeze": str(row.get("InFreeze", "False")).lower() in {"true", "1"},
            "R15": pd.to_numeric(row.get("R15", np.nan), errors="coerce"),
            "FreezeStartDate": str(row.get("FreezeStartDate", "") or ""),
        }
    except Exception:
        return {"InFreeze": False, "R15": np.nan, "FreezeStartDate": ""}


def save_state(in_freeze: bool, r15: float, freeze_start: str, updated_at: str) -> None:
    pd.DataFrame([{
        "InFreeze": bool(in_freeze),
        "R15": "" if pd.isna(r15) else float(r15),
        "FreezeStartDate": freeze_start,
        "UpdatedAt": updated_at,
    }]).to_csv(STATE_FILE, index=False)


def update_history(row: dict) -> None:
    new = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        old = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates("Date", keep="last").sort_values("Date")
    else:
        combined = new
    combined.to_csv(HISTORY_FILE, index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    updated_at = datetime.now(TIMEZONE).isoformat(timespec="seconds")

    spx = refresh_spx_prices()
    sp_pe = load_series(SP_PE_FILE, 20)
    real_yield = load_series(REAL_YIELD_FILE, 100)
    hy = load_series(HY_OAS_FILE, 60)
    sloos = load_series(SLOOS_FILE, 20)
    credit = load_series(CREDIT_GAP_FILE, 20)

    # --------------------------------------------------------
    # Daily S&P 500 price, ATH, drawdown
    # --------------------------------------------------------
    spx_current = float(spx.iloc[-1]["Value"])
    spx_date = pd.Timestamp(spx.iloc[-1]["Date"])
    ath = float(spx[spx["Date"] <= spx_date]["Value"].max())
    dd = max(0.0, (ath - spx_current) / ath)
    d_score = clip(dd / 0.30)

    # --------------------------------------------------------
    # Latest consensus Forward EPS; daily estimated Forward P/E
    # --------------------------------------------------------
    eps_hist = implied_forward_eps(sp_pe, spx)
    if len(eps_hist) < 5:
        raise RuntimeError("Could not derive enough S&P 500 Forward EPS history")

    latest_eps_row = eps_hist.iloc[-1]
    forward_eps = float(latest_eps_row["ForwardEPS"])
    consensus_date = pd.Timestamp(latest_eps_row["Date"])
    current_forward_pe = spx_current / forward_eps

    eps_3m_target = consensus_date - pd.DateOffset(months=3)
    eps_3m_rows = eps_hist[eps_hist["Date"] <= eps_3m_target]
    if eps_3m_rows.empty:
        eps_revision = 0.0
        eps_3m = np.nan
    else:
        eps_3m = float(eps_3m_rows.iloc[-1]["ForwardEPS"])
        eps_revision = forward_eps / eps_3m - 1.0

    earnings_penalty = clip((-eps_revision) / 0.10)

    # --------------------------------------------------------
    # Valuation score V
    # --------------------------------------------------------
    ry = float(real_yield.iloc[-1]["Value"])
    ry_date = pd.Timestamp(real_yield.iloc[-1]["Date"])
    cushion = 100.0 / current_forward_pe - ry
    v_score = clip((cushion - 1.5) / 2.5)

    # --------------------------------------------------------
    # HY OAS current, 20-trading-day change, 3-month change
    # --------------------------------------------------------
    hy_current = float(hy.iloc[-1]["Value"])
    hy_date = pd.Timestamp(hy.iloc[-1]["Date"])

    hy_business = hy[hy["Date"].dt.weekday < 5].copy()
    if len(hy_business) >= 21:
        hy_20d_old = float(hy_business.iloc[-21]["Value"])
    else:
        hy_20d_old = asof_value(hy, hy_date - pd.Timedelta(days=28))
    hy_20d_bp = (hy_current - hy_20d_old) * 100.0

    hy_3m_old = asof_value(hy, hy_date - pd.DateOffset(months=3))
    hy_3m_bp = (hy_current - hy_3m_old) * 100.0

    # --------------------------------------------------------
    # SLOOS and credit gap
    # --------------------------------------------------------
    sloos_current = float(sloos.iloc[-1]["Value"])
    sloos_prev = float(sloos.iloc[-2]["Value"]) if len(sloos) >= 2 else sloos_current
    delta_sloos = sloos_current - sloos_prev
    sloos_date = pd.Timestamp(sloos.iloc[-1]["Date"])

    gap = float(credit.iloc[-1]["Value"])
    gap_date = pd.Timestamp(credit.iloc[-1]["Date"])

    # Credit peak multiplier M
    cond_a = hy_20d_bp <= -50.0
    cond_b = delta_sloos <= 0.0
    m = (float(cond_a) + float(cond_b)) / 2.0
    d_effective = d_score * m

    # Opportunity score O
    opportunity = 0.65 * v_score + 0.35 * d_effective

    # --------------------------------------------------------
    # Crisis score R
    # --------------------------------------------------------
    r_oas = clip((hy_current - 4.0) / 4.0)
    r_oas_eff = r_oas if (sloos_current >= 15.0 or gap > 2.0) else 0.5 * r_oas
    r_sloos = clip(sloos_current / 40.0)
    r_delta_oas = clip((hy_3m_bp - 50.0) / 150.0)
    r_credit = clip((gap - 2.0) / 8.0)

    crisis = (
        0.45 * r_oas_eff
        + 0.30 * r_sloos
        + 0.15 * r_delta_oas
        + 0.10 * r_credit
    )

    bubble_gate = int(gap > 10.0 and sloos_current > 10.0 and delta_sloos > 0.0)

    # Crisis Freeze: capture R at the first observation with DD >= 15%.
    state = load_state()
    if dd < 0.15:
        in_freeze = False
        r15 = np.nan
        freeze_start = ""
        r_effective = crisis
    else:
        if not state["InFreeze"] or pd.isna(state["R15"]):
            in_freeze = True
            r15 = crisis
            freeze_start = spx_date.date().isoformat()
        else:
            in_freeze = True
            r15 = float(state["R15"])
            freeze_start = state["FreezeStartDate"]
        r_effective = min(crisis, r15)

    save_state(in_freeze, r15, freeze_start, updated_at)

    # --------------------------------------------------------
    # Final V3.1-D allocation — NO 5% rounding
    # --------------------------------------------------------
    w_raw = (
        70.0
        + 60.0 * (opportunity - 0.5)
        - 25.0 * r_effective
        - 10.0 * earnings_penalty
        - 10.0 * bubble_gate
    )
    allocation = float(max(50.0, min(100.0, w_raw)))

    history_row = {
        "Date": spx_date.date().isoformat(),
        "AllocationPct": allocation,
        "WRaw": w_raw,
        "SPX": spx_current,
        "ATH": ath,
        "DrawdownPct": dd * 100.0,
        "EstimatedForwardPE": current_forward_pe,
        "ForwardEPS": forward_eps,
        "ForwardEPSConsensusDate": consensus_date.date().isoformat(),
        "EPSRevision3M": eps_revision,
        "RealYield": ry,
        "CushionPct": cushion,
        "V": v_score,
        "D": d_score,
        "CreditPeakMultiplier": m,
        "DEffective": d_effective,
        "Opportunity": opportunity,
        "HYOAS": hy_current,
        "HYOAS20DChangeBp": hy_20d_bp,
        "HYOAS3MChangeBp": hy_3m_bp,
        "SLOOS": sloos_current,
        "SLOOSQoQ": delta_sloos,
        "CreditGap": gap,
        "EarningsPenalty": earnings_penalty,
        "CrisisScore": crisis,
        "REffective": r_effective,
        "R15": "" if pd.isna(r15) else r15,
        "BubbleGate": bubble_gate,
        "UpdatedAt": updated_at,
    }
    update_history(history_row)

    if not DASHBOARD_FILE.exists():
        raise RuntimeError("risk_dashboard.csv does not exist")

    dashboard = pd.read_csv(DASHBOARD_FILE)
    dashboard = dashboard[dashboard["RowType"] != "Allocation"].copy()

    detail = (
        f"SPX {spx_current:.0f} | DD -{dd*100:.1f}% | "
        f"O {opportunity:.2f} | R_eff {r_effective:.2f} | E {earnings_penalty:.2f}"
    )
    note = (
        "V3.1-D suggested S&P 500 stock allocation. No 5% rounding. "
        f"Daily estimated Forward P/E {current_forward_pe:.2f}x uses SPX {spx_current:.2f} "
        f"divided by latest implied consensus Forward EPS {forward_eps:.2f} "
        f"from {consensus_date.date().isoformat()}. Cushion {cushion:.2f}%. "
        f"HY OAS {hy_current:.2f}% ({hy_20d_bp:+.0f}bp/20D, {hy_3m_bp:+.0f}bp/3M); "
        f"SLOOS {sloos_current:.1f}% (QoQ {delta_sloos:+.1f}pp); "
        f"Credit Gap {gap:.2f}pp; EPS revision 3M {eps_revision*100:+.2f}%."
    )

    allocation_row = {
        "RowType": "Allocation",
        "Indicator": "S&P 500建议持仓",
        "CurrentValue": f"{allocation:.1f}%",
        "HistoryPercentile": "",
        "RecentChange": detail,
        "Rating": "",
        "RatingLevel": "",
        "DataDate": spx_date.date().isoformat(),
        "Source": "V3.1-D model / local cached market data",
        "Note": note,
        "UpdatedAt": updated_at,
    }

    dashboard = pd.concat([dashboard, pd.DataFrame([allocation_row])], ignore_index=True)
    dashboard = dashboard.reindex(columns=OUTPUT_COLUMNS)
    dashboard.to_csv(DASHBOARD_FILE, index=False)

    print("SUCCESS")
    print(f"S&P 500 date: {spx_date.date()}")
    print(f"S&P 500 close: {spx_current:.2f}; ATH: {ath:.2f}; drawdown: {dd*100:.2f}%")
    print(f"Daily estimated Forward P/E: {current_forward_pe:.2f}x")
    print(f"Cushion: {cushion:.2f}%; V={v_score:.3f}; D_eff={d_effective:.3f}; O={opportunity:.3f}")
    print(f"Crisis R={crisis:.3f}; R_eff={r_effective:.3f}; E={earnings_penalty:.3f}; Gate={bubble_gate}")
    print(f"Suggested S&P 500 allocation: {allocation:.1f}%")


if __name__ == "__main__":
    main()
