from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo
import re

import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "top100_history.csv"
OFFICIAL_CHECK_FILE = DATA_DIR / "official_share_checks.csv"

START_YEAR = 2026
QUARTER_MONTHS = [2, 5, 8, 11]
TIMEZONE = ZoneInfo("America/New_York")
LIST_SIZE = 125
RULE_VERSION = "3-official-shares-on-snapshot-day"

SOURCES = {
    "NASDAQ": "https://stockanalysis.com/list/nasdaq-stocks/",
    "NYSE": "https://stockanalysis.com/list/nyse-stocks/",
}

FINANCIALS_URL = "https://stockanalysis.com/stocks/sector/financials/"

COLUMNS = [
    "SnapshotDate",
    "DataDate",
    "Status",
    "ListType",
    "RuleVersion",
    "Rank",
    "Company",
    "Symbol",
    "Exchange",
    "MarketCap",
    "MarketCapDisplay",
    "Source",
]

LIST_TYPES = [
    "All",
    "ExFinancials",
]

OFFICIAL_CHECK_COLUMNS = [
    "SnapshotDate",
    "Symbol",
    "OfficialSharesOutstanding",
    "ADRRatio",
    "ListedShareEquivalent",
    "SharesSourceURL",
    "SharesSourceDate",
    "VerifiedAt",
    "Notes",
]


def headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        )
    }


def parse_market_cap(value):
    if pd.isna(value):
        return None

    text = str(value).strip().replace("$", "").replace(",", "")
    if not text or text in {"-", "N/A", "nan"}:
        return None

    multiplier = 1.0
    suffix = text[-1].upper()

    if suffix in {"T", "B", "M", "K"}:
        text = text[:-1]
        multiplier = {
            "T": 1e12,
            "B": 1e9,
            "M": 1e6,
            "K": 1e3,
        }[suffix]

    try:
        return float(text) * multiplier
    except ValueError:
        return None


def format_market_cap(value):
    value = float(value)

    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def normalize_company_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def read_stock_table(url, required_columns):
    response = requests.get(
        url,
        headers=headers(),
        timeout=45,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    for candidate in tables:
        if required_columns.issubset(set(candidate.columns)):
            return candidate.copy()

    raise RuntimeError(
        f"Could not find the expected stock table at {url}."
    )


def fetch_exchange(exchange, url):
    table = read_stock_table(
        url,
        {"Symbol", "Company Name", "Market Cap"},
    )

    table = table[["Symbol", "Company Name", "Market Cap"]].copy()
    table.columns = ["Symbol", "Company", "MarketCapRaw"]
    table["Exchange"] = exchange
    table["MarketCap"] = table["MarketCapRaw"].map(parse_market_cap)

    table = table.dropna(
        subset=["Symbol", "Company", "MarketCap"]
    )
    table = table[table["MarketCap"] > 0]

    return table[
        ["Symbol", "Company", "Exchange", "MarketCap"]
    ]


def fetch_financial_identifiers():
    financial_symbols = set()
    financial_company_keys = set()

    for page in range(1, 21):
        url = FINANCIALS_URL
        if page > 1:
            url = f"{FINANCIALS_URL}?page={page}"

        try:
            table = read_stock_table(
                url,
                {"Symbol", "Company Name"},
            )
        except Exception:
            if financial_symbols:
                break
            raise

        rows = table[["Symbol", "Company Name"]].dropna().copy()

        symbols = (
            rows["Symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        company_keys = rows["Company Name"].map(
            normalize_company_name
        )

        before_symbols = len(financial_symbols)
        before_companies = len(financial_company_keys)

        financial_symbols.update(symbols.tolist())
        financial_company_keys.update(company_keys.tolist())

        if (
            len(financial_symbols) == before_symbols
            and len(financial_company_keys) == before_companies
        ):
            break

    if len(financial_symbols) < 100 or len(financial_company_keys) < 100:
        raise RuntimeError(
            "Financial-sector company list looks incomplete."
        )

    print(
        f"Financial-sector symbols loaded: "
        f"{len(financial_symbols)}"
    )
    print(
        f"Financial-sector companies loaded: "
        f"{len(financial_company_keys)}"
    )

    return financial_symbols, financial_company_keys


def rank_list(frame):
    ranked = (
        frame
        .sort_values("MarketCap", ascending=False)
        .head(LIST_SIZE)
        .copy()
        .reset_index(drop=True)
    )

    if len(ranked) < LIST_SIZE:
        raise RuntimeError(
            f"Only {len(ranked)} companies were available; "
            f"{LIST_SIZE} are required."
        )

    ranked["Rank"] = range(1, LIST_SIZE + 1)
    ranked["MarketCapDisplay"] = ranked["MarketCap"].map(
        format_market_cap
    )

    return ranked[
        [
            "Rank",
            "Company",
            "Symbol",
            "Exchange",
            "MarketCap",
            "MarketCapDisplay",
        ]
    ]


def fetch_current_lists():
    frames = []

    for exchange, url in SOURCES.items():
        print(f"Downloading {exchange} market-cap list...")
        frames.append(fetch_exchange(exchange, url))

    combined = pd.concat(frames, ignore_index=True)

    combined["Symbol"] = (
        combined["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    combined["CompanyKey"] = combined["Company"].map(
        normalize_company_name
    )

    combined = combined.sort_values("MarketCap", ascending=False)
    combined = combined.drop_duplicates(
        subset=["CompanyKey"],
        keep="first",
    )

    financial_symbols, financial_company_keys = (
        fetch_financial_identifiers()
    )

    all_top125 = rank_list(combined)

    nonfinancial = combined[
        ~combined["Symbol"].isin(financial_symbols)
        & ~combined["CompanyKey"].isin(financial_company_keys)
    ].copy()

    ex_financials_top125 = rank_list(nonfinancial)

    return {
        "All": all_top125,
        "ExFinancials": ex_financials_top125,
    }


def quarter_end_dates(year):
    nyse = mcal.get_calendar("NYSE")
    dates = []

    for month in QUARTER_MONTHS:
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(1)

        schedule = nyse.schedule(
            start_date=start.date().isoformat(),
            end_date=end.date().isoformat(),
        )

        if schedule.empty:
            raise RuntimeError(
                f"No NYSE trading dates found for {year}-{month:02d}."
            )

        dates.append(schedule.index[-1].date())

    return dates


def make_snapshot(
    ranked,
    list_type,
    snapshot_date,
    data_date,
    status,
    source,
):
    snapshot = ranked.copy()
    snapshot.insert(0, "SnapshotDate", snapshot_date.isoformat())
    snapshot.insert(1, "DataDate", data_date.isoformat())
    snapshot.insert(2, "Status", status)
    snapshot.insert(3, "ListType", list_type)
    snapshot.insert(4, "RuleVersion", RULE_VERSION)
    snapshot["Source"] = source

    return snapshot[COLUMNS]


def empty_history():
    return pd.DataFrame(columns=COLUMNS)


def load_history():
    if not OUTPUT_FILE.exists():
        return empty_history()

    history = pd.read_csv(OUTPUT_FILE)

    if "ListType" not in history.columns:
        print(
            "Legacy Top-100 file detected. "
            "Rebuilding it as Top-125 with two list types."
        )
        return empty_history()

    for column in COLUMNS:
        if column not in history.columns:
            history[column] = None

    return history[COLUMNS]


def snapshot_rows(history, target_text, status=None):
    subset = history[
        history["SnapshotDate"].astype(str) == target_text
    ].copy()

    if status is not None:
        subset = subset[subset["Status"].astype(str) == status]

    return subset


def snapshot_is_complete(history, target_text):
    subset = snapshot_rows(history, target_text)

    if subset.empty:
        return False

    for list_type in LIST_TYPES:
        list_rows = subset[subset["ListType"] == list_type]

        if len(list_rows) != LIST_SIZE:
            return False

        if (list_rows["Status"] == "Temporary").any():
            versions = set(
                list_rows["RuleVersion"]
                .dropna()
                .astype(str)
            )
            if versions != {RULE_VERSION}:
                return False

    return True


def snapshot_is_official(history, target_text):
    subset = snapshot_rows(history, target_text, "Official")
    if subset.empty:
        return False

    return all(
        len(subset[subset["ListType"] == list_type]) == LIST_SIZE
        for list_type in LIST_TYPES
    )


def lists_from_snapshot_rows(rows):
    result = {}

    for list_type in LIST_TYPES:
        frame = rows[rows["ListType"].astype(str) == list_type].copy()
        if len(frame) != LIST_SIZE:
            return None

        frame["Rank"] = pd.to_numeric(frame["Rank"], errors="coerce")
        frame["MarketCap"] = pd.to_numeric(
            frame["MarketCap"], errors="coerce"
        )
        frame = frame.sort_values("Rank")

        result[list_type] = frame[
            [
                "Rank",
                "Company",
                "Symbol",
                "Exchange",
                "MarketCap",
                "MarketCapDisplay",
            ]
        ].copy()

    return result


def load_official_share_equivalents(snapshot_date, required_symbols):
    target_text = snapshot_date.isoformat()

    if not OFFICIAL_CHECK_FILE.exists():
        return {}, sorted(required_symbols), [
            f"{OFFICIAL_CHECK_FILE.name} does not exist."
        ]

    checks = pd.read_csv(OFFICIAL_CHECK_FILE)

    for column in OFFICIAL_CHECK_COLUMNS:
        if column not in checks.columns:
            checks[column] = None

    checks = checks[
        checks["SnapshotDate"].astype(str) == target_text
    ].copy()

    if checks.empty:
        return {}, sorted(required_symbols), []

    checks["Symbol"] = (
        checks["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    checks = checks.drop_duplicates(
        subset=["Symbol"],
        keep="last",
    )

    equivalents = {}
    errors = []

    for symbol in sorted(required_symbols):
        row = checks[checks["Symbol"] == symbol]
        if row.empty:
            continue

        row = row.iloc[-1]
        source_url = str(row.get("SharesSourceURL", "") or "").strip()
        source_date = str(row.get("SharesSourceDate", "") or "").strip()

        if not source_url or source_url.lower() == "nan":
            errors.append(f"{symbol}: SharesSourceURL is missing.")
            continue

        if not source_date or source_date.lower() == "nan":
            errors.append(f"{symbol}: SharesSourceDate is missing.")
            continue

        listed_equivalent = pd.to_numeric(
            pd.Series([row.get("ListedShareEquivalent")]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(listed_equivalent) and float(listed_equivalent) > 0:
            equivalents[symbol] = float(listed_equivalent)
            continue

        shares = pd.to_numeric(
            pd.Series([row.get("OfficialSharesOutstanding")]),
            errors="coerce",
        ).iloc[0]
        adr_ratio = pd.to_numeric(
            pd.Series([row.get("ADRRatio")]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(shares) or float(shares) <= 0:
            errors.append(
                f"{symbol}: OfficialSharesOutstanding is missing/invalid."
            )
            continue

        if pd.isna(adr_ratio):
            adr_ratio = 1.0

        if float(adr_ratio) <= 0:
            errors.append(f"{symbol}: ADRRatio must be positive.")
            continue

        equivalents[symbol] = float(shares) / float(adr_ratio)

    missing = sorted(set(required_symbols) - set(equivalents))
    return equivalents, missing, errors


def yf_symbol(symbol):
    return str(symbol).strip().upper().replace(".", "-")


def fetch_snapshot_close_prices(symbols, snapshot_date):
    start = snapshot_date.isoformat()
    end = (snapshot_date + timedelta(days=1)).isoformat()

    symbol_map = {
        symbol: yf_symbol(symbol)
        for symbol in sorted(symbols)
    }

    prices = {}

    try:
        downloaded = yf.download(
            tickers=list(symbol_map.values()),
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
            multi_level_index=True,
        )
    except Exception as exc:
        print(f"Batch price download failed: {exc}")
        downloaded = pd.DataFrame()

    for symbol, ticker in symbol_map.items():
        close_value = None

        try:
            if not downloaded.empty:
                if isinstance(downloaded.columns, pd.MultiIndex):
                    first_level = set(
                        downloaded.columns.get_level_values(0)
                    )
                    second_level = set(
                        downloaded.columns.get_level_values(1)
                    )

                    if ticker in first_level:
                        series = downloaded[ticker]["Close"].dropna()
                    elif ticker in second_level and "Close" in first_level:
                        series = downloaded["Close"][ticker].dropna()
                    else:
                        series = pd.Series(dtype=float)
                else:
                    series = downloaded.get("Close", pd.Series(dtype=float)).dropna()

                if not series.empty:
                    close_value = float(series.iloc[-1])
        except Exception:
            close_value = None

        if close_value is None:
            try:
                fallback = yf.Ticker(ticker).history(
                    start=start,
                    end=end,
                    auto_adjust=False,
                    actions=False,
                )
                if not fallback.empty and "Close" in fallback.columns:
                    series = pd.to_numeric(
                        fallback["Close"], errors="coerce"
                    ).dropna()
                    if not series.empty:
                        close_value = float(series.iloc[-1])
            except Exception as exc:
                print(f"Price lookup failed for {symbol}: {exc}")

        if close_value is not None and close_value > 0:
            prices[symbol] = close_value

    return prices


def build_verified_official_lists(snapshot_date, provisional_lists):
    required_symbols = set()

    for ranked in provisional_lists.values():
        required_symbols.update(
            ranked["Symbol"].astype(str).str.strip().str.upper().tolist()
        )

    equivalents, missing_shares, share_errors = (
        load_official_share_equivalents(
            snapshot_date,
            required_symbols,
        )
    )

    if share_errors:
        print("Official share-check validation errors:")
        for error in share_errors:
            print(f" - {error}")

    if missing_shares:
        print(
            f"Official share verification incomplete for "
            f"{snapshot_date.isoformat()}: "
            f"{len(missing_shares)} symbols still missing."
        )
        if len(missing_shares) <= 20:
            print("Missing symbols: " + ", ".join(missing_shares))
        return None

    prices = fetch_snapshot_close_prices(
        required_symbols,
        snapshot_date,
    )

    missing_prices = sorted(required_symbols - set(prices))
    if missing_prices:
        print(
            "Official snapshot cannot be finalized because closing prices "
            "are missing for: "
            + ", ".join(missing_prices)
        )
        return None

    verified_lists = {}

    for list_type, ranked in provisional_lists.items():
        verified = ranked.copy()
        verified["Symbol"] = (
            verified["Symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        verified["MarketCap"] = verified["Symbol"].map(
            lambda symbol: equivalents[symbol] * prices[symbol]
        )

        verified_lists[list_type] = rank_list(verified)

    return verified_lists


def replace_snapshot(history, target_text, snapshots):
    history = history[
        history["SnapshotDate"].astype(str) != target_text
    ].copy()

    if snapshots:
        history = pd.concat(
            [history] + snapshots,
            ignore_index=True,
        )

    return history


def main():
    today = datetime.now(TIMEZONE).date()
    history = load_history()

    snapshot_targets = []

    for year in range(START_YEAR, today.year + 1):
        for target in quarter_end_dates(year):
            if year <= today.year:
                snapshot_targets.append(target)

    if not snapshot_targets:
        print("No quarterly snapshot dates are available.")
        return

    current_lists = None
    changed = False

    for target in snapshot_targets:
        target_text = target.isoformat()

        if snapshot_is_official(history, target_text):
            continue

        if target > today:
            if current_lists is None:
                current_lists = fetch_current_lists()

            snapshots = [
                make_snapshot(
                    ranked,
                    list_type,
                    target,
                    today,
                    "Temporary",
                    "StockAnalysis",
                )
                for list_type, ranked in current_lists.items()
            ]

            history = replace_snapshot(
                history,
                target_text,
                snapshots,
            )
            changed = True
            print(
                f"Refreshed temporary Top 125 snapshots for {target_text} "
                f"using data from {today.isoformat()}."
            )
            continue

        if target == today:
            if current_lists is None:
                current_lists = fetch_current_lists()

            verified_lists = build_verified_official_lists(
                target,
                current_lists,
            )

            if verified_lists is not None:
                snapshots = [
                    make_snapshot(
                        ranked,
                        list_type,
                        target,
                        target,
                        "Official",
                        "Official shares outstanding x snapshot close",
                    )
                    for list_type, ranked in verified_lists.items()
                ]
                print(
                    f"Saved VERIFIED official Top 125 snapshots for "
                    f"{target_text}."
                )
            else:
                snapshots = [
                    make_snapshot(
                        ranked,
                        list_type,
                        target,
                        target,
                        "PendingOfficial",
                        "StockAnalysis - pending official shares check",
                    )
                    for list_type, ranked in current_lists.items()
                ]
                print(
                    f"Saved PendingOfficial snapshots for {target_text}; "
                    "they will not lock until every displayed company has "
                    "an official share-count verification."
                )

            history = replace_snapshot(
                history,
                target_text,
                snapshots,
            )
            changed = True
            continue

        # Past target: if it was captured as PendingOfficial on the true
        # snapshot day, keep that exact provisional universe and allow a later
        # official-share audit to finalize it using the original date's close.
        pending_rows = snapshot_rows(
            history,
            target_text,
            "PendingOfficial",
        )

        if not pending_rows.empty:
            provisional_lists = lists_from_snapshot_rows(pending_rows)

            if provisional_lists is None:
                print(
                    f"PendingOfficial snapshot {target_text} is incomplete; "
                    "leaving it unchanged."
                )
                continue

            verified_lists = build_verified_official_lists(
                target,
                provisional_lists,
            )

            if verified_lists is None:
                print(
                    f"PendingOfficial snapshot {target_text} still awaits "
                    "complete official share verification."
                )
                continue

            snapshots = [
                make_snapshot(
                    ranked,
                    list_type,
                    target,
                    target,
                    "Official",
                    "Official shares outstanding x snapshot close",
                )
                for list_type, ranked in verified_lists.items()
            ]

            history = replace_snapshot(
                history,
                target_text,
                snapshots,
            )
            changed = True
            print(
                f"Finalized verified Official snapshot for {target_text}."
            )
            continue

        # Historical dates that predate this verification workflow cannot be
        # reconstructed as truly official without a dedicated historical audit.
        # Preserve any complete existing snapshot; otherwise create a clearly
        # labeled Temporary placeholder instead of pretending it is Official.
        if snapshot_is_complete(history, target_text):
            continue

        if current_lists is None:
            current_lists = fetch_current_lists()

        snapshots = [
            make_snapshot(
                ranked,
                list_type,
                target,
                today,
                "Temporary",
                "StockAnalysis - historical placeholder",
            )
            for list_type, ranked in current_lists.items()
        ]

        history = replace_snapshot(
            history,
            target_text,
            snapshots,
        )
        changed = True
        print(
            f"Saved temporary placeholder for past target {target_text} "
            f"using data from {today.isoformat()}."
        )

    if not changed:
        print("Quarterly Top 125 history is already up to date.")
        return

    history["Rank"] = pd.to_numeric(
        history["Rank"],
        errors="coerce",
    )

    history["MarketCap"] = pd.to_numeric(
        history["MarketCap"],
        errors="coerce",
    )

    history = history.sort_values(
        ["SnapshotDate", "ListType", "Rank"],
        ascending=[True, True, True],
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(OUTPUT_FILE, index=False)

    print("\nSUCCESS")
    print(f"Snapshot dates stored: {history['SnapshotDate'].nunique()}")
    print(f"List types stored: {history['ListType'].nunique()}")
    print(f"Rows stored: {len(history)}")


if __name__ == "__main__":
    main()
