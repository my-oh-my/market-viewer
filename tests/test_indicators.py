"""Tests for the indicators module."""

import pandas as pd
import pytest
from src.analysis import indicators


@pytest.fixture
def _sample_data():
    """Creates sample market data for testing."""
    data = {
        "High": [10, 12, 15, 14, 13],
        "Low": [8, 9, 11, 10, 9],
        "Close": [9, 11, 14, 12, 10],
        "Volume": [100, 200, 150, 120, 130],
    }
    return pd.DataFrame(data)


def test_calculate_vwap(_sample_data):
    """Tests VWAP calculation."""
    vwap_df = indicators.calculate_vwap(_sample_data)

    assert "VWAP" in vwap_df.columns
    assert "VWAP_Upper" in vwap_df.columns
    assert "VWAP_Lower" in vwap_df.columns
    assert not vwap_df.empty

    # Check first value: VWAP = Typical Price (since it's cumulative from start)
    tp = (
        _sample_data["High"][0] + _sample_data["Low"][0] + _sample_data["Close"][0]
    ) / 3
    assert vwap_df["VWAP"].iloc[0] == pytest.approx(tp)


def test_calculate_atr(_sample_data):
    """Tests ATR calculation."""
    atr = indicators.calculate_atr(_sample_data, window=2)

    assert isinstance(atr, pd.Series)
    assert len(atr) == len(_sample_data)
    # First value should be NaN due to rolling window
    assert pd.isna(atr.iloc[0])
    # Third value should be present (window=2, so index 1 is first valid if min_periods=None,
    # but rolling usually requires window size)
    # Actually rolling(2) produces result at index 1
    assert not pd.isna(atr.iloc[1])


def test_empty_data():
    """Tests behavior with empty data."""
    empty_df = pd.DataFrame()
    assert indicators.calculate_vwap(empty_df).empty
    assert indicators.calculate_atr(empty_df).empty
