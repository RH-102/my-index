"""Run the V3.1-D allocation model and publish it as a homepage summary card."""

import pandas as pd

import update_allocation


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
