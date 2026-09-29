"""Validate completed market sessions and retry missing daily closing prices."""

import math
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf


def completed_market_dates(start, now=None):
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ZoneInfo("America/New_York")).date()
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=today)
    # Match the dashboard's 30-minute allowance, including early closes/DST.
    return [date.date().isoformat() for date, close in schedule["market_close"].items()
            if close.to_pydatetime() + timedelta(minutes=30) <= now]


def valid_price(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def extract_closes(raw, symbol):
    if raw is None or raw.empty:
        return {}
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol in raw.columns.get_level_values(0):
                close = raw[symbol]["Close"]
            else:
                close = raw["Close"][symbol]
        else:
            close = raw["Close"]
    except KeyError:
        return {}
    return {pd.Timestamp(date).date().isoformat(): float(value)
            for date, value in close.items()
            if not pd.isna(date) and valid_price(value)}


def required_price_dates(calendar_dates, snapshots):
    """A rebalance close needs both the outgoing and incoming baskets."""
    required = {}
    current = None
    for date in calendar_dates:
        incoming = snapshots.get(date)
        symbols = set(current or ()) | set(incoming or ())
        if not symbols:
            raise RuntimeError(f"No active holdings for {date}")
        for symbol in symbols:
            required.setdefault(symbol, set()).add(date)
        if incoming is not None:
            current = incoming
    return required


def missing_price_dates(history, required):
    return {symbol: missing for symbol, dates in sorted(required.items())
            if (missing := sorted(date for date in dates
                                  if not valid_price(history.get(symbol, {}).get(date))))}


def download_missing(symbol, dates, attempt):
    start = datetime.fromisoformat(min(dates)).date()
    end = datetime.fromisoformat(max(dates)).date() + timedelta(days=1)
    options = dict(interval="1d", auto_adjust=False, timeout=15)
    if attempt == 1:
        # A short, serial request avoids reusing the original multi-ticker batch.
        raw = yf.download(symbol, start=start.isoformat(), end=end.isoformat(),
                          threads=False, progress=False, group_by="ticker",
                          multi_level_index=True, **options)
    elif (datetime.now(timezone.utc).date() - start).days < 28:
        # Change the query shape if Yahoo's date-range response is stale.
        raw = yf.Ticker(symbol).history(period="1mo", raise_errors=True, **options)
    else:
        raw = yf.Ticker(symbol).history(
            start=(start - timedelta(days=7)).isoformat(), end=end.isoformat(),
            raise_errors=True, **options)
    return extract_closes(raw, symbol)


def fill_missing_prices(history, required, loader=download_missing,
                        sleep=time.sleep, max_attempts=3, budget_seconds=180):
    """Retry only missing symbol/date pairs; never forward-fill a close."""
    deadline = time.monotonic() + budget_seconds
    for attempt in range(1, max_attempts + 1):
        missing = missing_price_dates(history, required)
        if not missing:
            return
        print(f"Missing closing prices (retry {attempt}/{max_attempts}): " +
              "; ".join(f"{symbol}: {', '.join(dates)}"
                        for symbol, dates in missing.items()), flush=True)
        if attempt > 1:
            delay = 5 * (attempt - 1)
            if time.monotonic() + delay >= deadline:
                break
            sleep(delay)
        for symbol, dates in missing.items():
            if time.monotonic() >= deadline:
                break
            try:
                fresh = loader(symbol, dates, attempt)
                # Repair only the holes; do not change already validated closes
                # with incidental observations from a different retry window.
                recovered = {date: float(fresh[date]) for date in dates
                             if valid_price(fresh.get(date))}
                history.setdefault(symbol, {}).update(recovered)
                print(f"{symbol}: recovered {len(recovered)}/{len(dates)} closes", flush=True)
            except Exception as exc:
                print(f"{symbol}: retry {attempt} failed: {exc}", flush=True)
        if time.monotonic() >= deadline:
            break
    missing = missing_price_dates(history, required)
    if missing:
        detail = "; ".join(f"{symbol}: {', '.join(dates)}"
                           for symbol, dates in missing.items())
        raise RuntimeError("Closing prices still missing after retries; no index data "
                           f"was saved. {detail}")
