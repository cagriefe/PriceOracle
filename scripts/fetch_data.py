"""Download Binance spot klines into a CSV, or append only the new candles to an existing one.

Uses Binance's public market-data endpoint, so no API key is required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.helpers import cli_path, resolve_path
from utils.logger import get_logger

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
MAX_LIMIT = 1000
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
KLINE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

logger = get_logger(__name__)


def make_session() -> requests.Session:
    retry = Retry(total=5, backoff_factor=1, status_forcelist=(418, 429, 500, 502, 503, 504), allowed_methods=("GET",))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def to_milliseconds(value) -> int:
    """Convert a date string or timestamp (UTC unless it carries a timezone) to epoch milliseconds."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.timestamp() * 1000)


def fetch_klines(symbol, interval, start_ms, end_ms=None, session=None, pause=0.2) -> list:
    """Page through ``/api/v3/klines`` from ``start_ms`` and keep only candles that closed before ``end_ms``."""
    session = session or make_session()
    end_ms = end_ms or int(time.time() * 1000)
    rows = []
    while start_ms < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": MAX_LIMIT}
        response = session.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        if len(batch) < MAX_LIMIT:
            break
        time.sleep(pause)
    return [row for row in rows if row[6] < end_ms]


def klines_to_frame(rows) -> pd.DataFrame:
    """Klines as strings, exactly as Binance returns them, so rewriting the CSV leaves old rows untouched."""
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms").dt.strftime(TIMESTAMP_FORMAT)
    return frame.astype(str)


def update_csv(
    path, symbol="BTCUSDT", interval="1h", start="2017-01-01", full_refresh=False, session=None, end_ms=None
) -> int:
    """Create or extend the CSV at ``path`` and return the number of candles fetched."""
    path = Path(path)
    existing = None
    start_ms = to_milliseconds(start)
    if path.exists() and not full_refresh:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not existing.empty:
            # Re-fetch the last stored candle too: it may have been saved before it closed.
            start_ms = to_milliseconds(existing["timestamp"].iloc[-1])

    fetched = klines_to_frame(fetch_klines(symbol, interval, start_ms, end_ms=end_ms, session=session))
    combined = fetched if existing is None else pd.concat([existing, fetched], ignore_index=True)
    combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp", kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return len(fetched)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance spot symbol")
    parser.add_argument("--interval", default="1h", help="Kline interval, e.g. 15m, 1h, 4h, 1d")
    parser.add_argument("--start", default="2017-01-01", help="Start date (UTC) for a fresh download")
    parser.add_argument("--output", help="CSV to create or update (default: data/btc_usdt.csv)")
    parser.add_argument("--full-refresh", action="store_true", help="Ignore the existing CSV and download everything")
    args = parser.parse_args(argv)

    output = cli_path(args.output) or resolve_path("data/btc_usdt.csv")
    count = update_csv(output, args.symbol, args.interval, args.start, args.full_refresh)
    logger.info("Fetched %d %s %s candles into %s", count, args.symbol, args.interval, Path(output).name)


if __name__ == "__main__":
    main()
