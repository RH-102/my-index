"""Run the V3.1-D allocation model and publish it as a homepage summary card."""

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


# Pandas can preserve different datetime resolutions when the source CSVs were
# written by different libraries. Normalize them before the as-of merge.
update_allocation.implied_forward_eps = implied_forward_eps_compat

update_allocation.main()

path = update_allocation.DASHBOARD_FILE
frame = pd.read_csv(path)
mask = frame["RowType"].eq("Allocation")
if mask.any():
    frame.loc[mask, "RowType"] = "Summary"
    frame.loc[mask, "Rating"] = frame.loc[mask, "CurrentValue"]
    # Reuse the existing compact green badge style for the allocation percentage.
    frame.loc[mask, "RatingLevel"] = 0
    frame.to_csv(path, index=False)
    print("Published S&P 500 suggested allocation as summary card")
