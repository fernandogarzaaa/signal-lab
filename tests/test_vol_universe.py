"""Tests for vol.universe: frozen S&P 100 + SPY list."""

import pandas as pd

from signal_lab.vol import universe


def test_universe_count():
    tickers = universe.universe_tickers()
    assert len(tickers) == 102
    assert tickers[-1] == "SPY"
    assert len([t for t in tickers if t != "SPY"]) == 101


def test_universe_no_duplicates():
    tickers = universe.universe_tickers()
    assert len(set(tickers)) == len(tickers)


def test_yfinance_override():
    assert universe.yfinance_ticker("BRK.B") == "BRK-B"
    assert universe.yfinance_ticker("AAPL") == "AAPL"


def test_csv_round_trip(tmp_path):
    path = tmp_path / "universe_vol.csv"
    universe.write_universe_csv(path)
    df = pd.read_csv(path)
    assert list(df.columns) == ["ticker"]
    assert len(df) == 102
    assert universe.load_universe(path) == universe.universe_tickers()


def test_csv_drift_rejected(tmp_path):
    path = tmp_path / "universe_vol.csv"
    universe.write_universe_csv(path)
    df = pd.read_csv(path)
    df.loc[0, "ticker"] = "FAKE"
    df.to_csv(path, index=False)
    try:
        universe.load_universe(path)
    except ValueError:
        return
    raise AssertionError("drifted universe CSV was accepted")
