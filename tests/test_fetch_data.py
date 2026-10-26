import pandas as pd

from scripts import fetch_data
from scripts.fetch_data import KLINE_COLUMNS, fetch_klines, klines_to_frame, to_milliseconds, update_csv

HOUR_MS = 3_600_000
START_MS = to_milliseconds("2024-01-01")


def make_kline(index: int, volume: str = "10.00000000") -> list:
    """A Binance-style kline row for the ``index``-th hourly candle after ``START_MS``."""
    open_ms = START_MS + index * HOUR_MS
    price = f"{100 + index:.8f}"
    return [open_ms, price, price, price, price, volume, open_ms + HOUR_MS - 1, "1000.0", 42, "5.0", "500.0", "0"]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Serves klines like ``/api/v3/klines``: open time within [startTime, endTime], at most ``limit`` rows."""

    def __init__(self, klines):
        self.klines = klines
        self.requests = []

    def get(self, url, params, timeout):
        self.requests.append(params)
        rows = [row for row in self.klines if params["startTime"] <= row[0] <= params["endTime"]]
        return FakeResponse(rows[: params["limit"]])


def test_to_milliseconds_defaults_to_utc():
    assert to_milliseconds("1970-01-01 00:00:01") == 1000
    assert to_milliseconds("1970-01-01T03:00:00+03:00") == 0


def test_klines_to_frame_keeps_binance_formatting():
    frame = klines_to_frame([make_kline(0)])
    assert list(frame.columns) == KLINE_COLUMNS
    assert frame["timestamp"].iloc[0] == "2024-01-01 00:00:00"
    assert frame["close"].iloc[0] == "100.00000000"


def test_fetch_klines_paginates_and_skips_the_open_candle(monkeypatch):
    monkeypatch.setattr(fetch_data, "MAX_LIMIT", 2)
    klines = [make_kline(i) for i in range(5)]
    session = FakeSession(klines)
    rows = fetch_klines("BTCUSDT", "1h", START_MS, end_ms=START_MS + 4 * HOUR_MS + 10, session=session, pause=0)
    assert rows == klines[:4]
    assert len(session.requests) == 3


def test_update_csv_refreshes_the_last_row_and_appends(tmp_path):
    path = tmp_path / "btc.csv"
    stale = [make_kline(i) for i in range(3)] + [make_kline(3, volume="0.50000000")]
    klines_to_frame(stale).to_csv(path, index=False)
    original_lines = path.read_text().splitlines()

    fetched = update_csv(path, session=FakeSession([make_kline(i) for i in range(6)]), end_ms=START_MS + 6 * HOUR_MS)

    lines = path.read_text().splitlines()
    assert fetched == 3
    assert lines[:4] == original_lines[:4]
    assert len(lines) == 1 + 6
    assert pd.read_csv(path)["volume"].iloc[3] == 10.0


def test_update_csv_without_closed_candles_leaves_the_file_untouched(tmp_path):
    path = tmp_path / "btc.csv"
    klines = [make_kline(i) for i in range(3)]
    klines_to_frame(klines).to_csv(path, index=False)
    before = path.read_text()
    assert update_csv(path, session=FakeSession(klines), end_ms=START_MS + 2 * HOUR_MS + 5) == 0
    assert path.read_text() == before


def test_update_csv_creates_a_new_file(tmp_path):
    path = tmp_path / "new" / "btc.csv"
    klines = [make_kline(i) for i in range(4)]
    assert update_csv(path, start="2024-01-01", session=FakeSession(klines), end_ms=START_MS + 4 * HOUR_MS) == 4
    assert len(pd.read_csv(path)) == 4
