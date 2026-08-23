"""Run the V3.2-Alpha allocation model and publish it as a homepage summary card.

V3.2-Alpha keeps the existing market-data, Crisis Score, Earnings Penalty,
Bubble Gate, Crisis Freeze, valuation, drawdown, and re-entry calculations from
update_allocation.py, but changes the final stock-allocation rule to:

    W = clip(100 - 30*R_effective - 10*E - 10*BubbleGate, 60, 100)

Valuation V, Drawdown D, and Credit Peak M remain diagnostic / re-entry signals;
they do not directly reduce the normal allocation. No 5% rounding is applied.
"""

import pandas as pd

import update_allocation


def implied_forward_eps_compat(pe: pd.DataFrame, spx: pd.DataFrame) -> pd.DataFrame:
    """Force identical datetime64[ns] dtypes before pandas merge_asof."""
    left = pe[["Date", "Value"]].copy().sort_values("Date")
    left.columns = ["Date", "ForwardPE"]
    left["Date"] = pd.to_datetime(left["Date"], errors="coerce").astype("datetime64[ns]")

    right = spx[["Date", "Value"]].copy().sort_values("Date")
    right.columns = ["PriceDate", "SPX"]
    right["PriceDate"] = pd.to_datetime(
        right["PriceDate"], errors="coerce"
    ).astype("datetime64[ns]")

    left = left.dropna().sort_values("Date")
    right = right.dropna().sort_values("PriceDate")

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


def as_number(value, default=0.0):
    number = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(number) else float(number)


# Pandas can preserve different datetime resolutions when the source CSVs were
# written by different libraries. Normalize them before the as-of merge.
update_allocation.implied_forward_eps = implied_forward_eps_compat

# Run the existing data pipeline first. This updates all source inputs, Crisis
# Freeze state, and the detailed history fields. The final allocation is then
# replaced below by the V3.2-Alpha rule.
update_allocation.main()

# ---------------------------------------------------------------------------
# Replace the latest history row with the V3.2-Alpha allocation.
# ---------------------------------------------------------------------------
history_path = update_allocation.HISTORY_FILE
history = pd.read_csv(history_path)
if history.empty:
    raise RuntimeError("sp500_allocation_history.csv is empty")

history_dates = pd.to_datetime(history["Date"], errors="coerce")
if history_dates.notna().sum() == 0:
    raise RuntimeError("No valid dates in sp500_allocation_history.csv")

latest_idx = history_dates.idxmax()
latest = history.loc[latest_idx].copy()

r_effective = as_number(latest.get("REffective"))
earnings_penalty = as_number(latest.get("EarningsPenalty"))
bubble_gate = as_number(latest.get("BubbleGate"))

w_raw = (
    100.0
    - 30.0 * r_effective
    - 10.0 * earnings_penalty
    - 10.0 * bubble_gate
)
allocation = max(60.0, min(100.0, w_raw))

history.loc[latest_idx, "WRaw"] = w_raw
history.loc[latest_idx, "AllocationPct"] = allocation

# A completely blank CSV column is inferred as float64 by pandas. Force an
# object column before writing the text model label, otherwise pandas 3.x raises.
if "ModelVersion" not in history.columns:
    history["ModelVersion"] = ""
else:
    history["ModelVersion"] = history["ModelVersion"].astype("object")
history.loc[latest_idx, "ModelVersion"] = "V3.2-Alpha"

history.to_csv(history_path, index=False)
latest = history.loc[latest_idx].copy()

# ---------------------------------------------------------------------------
# Publish the Alpha result as the homepage summary card.
# ---------------------------------------------------------------------------
path = update_allocation.DASHBOARD_FILE
frame = pd.read_csv(path)

# Prefer the Allocation row just written by update_allocation.main(). Remove any
# stale Summary copy of the same card if this script is run manually twice.
stale_summary = (
    frame["Indicator"].eq("S&P 500建议持仓")
    & frame["RowType"].eq("Summary")
)
frame = frame[~stale_summary].copy()
mask = frame["RowType"].eq("Allocation") & frame["Indicator"].eq("S&P 500建议持仓")

if mask.any():
    spx = as_number(latest.get("SPX"))
    drawdown = as_number(latest.get("DrawdownPct"))
    forward_pe = as_number(latest.get("EstimatedForwardPE"))
    forward_eps = as_number(latest.get("ForwardEPS"))
    cushion = as_number(latest.get("CushionPct"))
    v_score = as_number(latest.get("V"))
    d_score = as_number(latest.get("D"))
    credit_peak_m = as_number(latest.get("CreditPeakMultiplier"))
    crisis = as_number(latest.get("CrisisScore"))
    hy_oas = as_number(latest.get("HYOAS"))
    hy_20d = as_number(latest.get("HYOAS20DChangeBp"))
    hy_3m = as_number(latest.get("HYOAS3MChangeBp"))
    sloos = as_number(latest.get("SLOOS"))
    sloos_qoq = as_number(latest.get("SLOOSQoQ"))
    credit_gap = as_number(latest.get("CreditGap"))
    eps_revision = as_number(latest.get("EPSRevision3M"))
    consensus_date = str(latest.get("ForwardEPSConsensusDate", ""))

    current_value = f"{allocation:.1f}%"
    recent_change = (
        f"SPX {spx:.0f} | DD -{drawdown:.1f}% | "
        f"R_eff {r_effective:.2f} | E {earnings_penalty:.2f} | Gate {int(bubble_gate)}"
    )
    note = (
        "V3.2-Alpha suggested S&P 500 stock allocation. No 5% rounding. "
        "Final formula: W = clip(100 - 30*R_effective - 10*E - 10*BubbleGate, 60, 100). "
        "Valuation V, Drawdown D, and Credit Peak M are retained as re-entry / confirmation signals "
        "but do not directly reduce normal allocation. "
        f"Current R={crisis:.3f}, R_effective={r_effective:.3f}, E={earnings_penalty:.3f}, "
        f"BubbleGate={int(bubble_gate)}. Daily estimated Forward P/E {forward_pe:.2f}x uses "
        f"SPX {spx:.2f} and implied consensus Forward EPS {forward_eps:.2f} from {consensus_date}. "
        f"Cushion {cushion:.2f}%, V={v_score:.3f}, D={d_score:.3f}, M={credit_peak_m:.1f}. "
        f"HY OAS {hy_oas:.2f}% ({hy_20d:+.0f}bp/20D, {hy_3m:+.0f}bp/3M); "
        f"SLOOS {sloos:.1f}% (QoQ {sloos_qoq:+.1f}pp); Credit Gap {credit_gap:.2f}pp; "
        f"EPS revision 3M {eps_revision*100:+.2f}%."
    )

    frame.loc[mask, "CurrentValue"] = current_value
    frame.loc[mask, "RecentChange"] = recent_change
    frame.loc[mask, "Source"] = "V3.2-Alpha model / local cached market data"
    frame.loc[mask, "Note"] = note
    frame.loc[mask, "RowType"] = "Summary"
    frame.loc[mask, "Rating"] = current_value
    frame.loc[mask, "RatingLevel"] = 0
    frame.to_csv(path, index=False)

    print("Published V3.2-Alpha S&P 500 suggested allocation as summary card")
    print(
        f"V3.2-Alpha allocation: {allocation:.1f}% "
        f"(R_eff={r_effective:.3f}, E={earnings_penalty:.3f}, Gate={int(bubble_gate)})"
    )
else:
    raise RuntimeError("S&P 500 allocation row was not generated")
