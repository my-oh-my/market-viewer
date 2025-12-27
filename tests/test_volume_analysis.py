"""Unit tests for volume analysis module."""

import pandas as pd
from src.analysis import volume


def test_calculate_volume_profile_basic():
    """Test basic volume profile calculation."""
    # Create sample data
    data = pd.DataFrame(
        {
            "Open": [100, 102, 101, 99, 100],
            "Close": [102, 101, 99, 100, 103],
            "Volume": [100, 200, 150, 50, 300],
        }
    )

    # Expected behavior:
    # Row 0: 100->102 (Bullish), Vol 100
    # Row 1: 102->101 (Bearish), Vol 200
    # Row 2: 101->99 (Bearish), Vol 150
    # Row 3: 99->100 (Bullish), Vol 50
    # Row 4: 100->103 (Bullish), Vol 300

    # Run calculation with small number of bins to force grouping
    vp = volume.calculate_volume_profile(data, bins=5)

    assert not vp.empty
    assert "Bullish_Volume" in vp.columns
    assert "Bearish_Volume" in vp.columns
    assert "Total_Volume" in vp.columns
    assert "POC" in vp.columns

    # Check total volume matches
    assert vp["Total_Volume"].sum() == data["Volume"].sum()

    # Check POC
    assert vp["POC"].sum() == 1  # Only one POC

    # Check Bullish/Bearish split
    # Total Bullish Volume = 100 + 50 + 300 = 450
    # Total Bearish Volume = 200 + 150 = 350
    assert vp["Bullish_Volume"].sum() == 450
    assert vp["Bearish_Volume"].sum() == 350


def test_calculate_volume_profile_empty():
    """Test volume profile calculation with empty data."""
    data = pd.DataFrame()
    vp = volume.calculate_volume_profile(data)
    assert vp.empty


def test_calculate_volume_profile_no_movement():
    """Test volume profile calculation with no price movement."""
    data = pd.DataFrame({"Open": [100, 100], "Close": [100, 100], "Volume": [100, 100]})
    vp = volume.calculate_volume_profile(data, bins=10)
    assert not vp.empty
    assert vp["Total_Volume"].sum() == 200
